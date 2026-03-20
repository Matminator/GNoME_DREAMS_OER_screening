from datetime import datetime
import subprocess
from pysqa import QueueAdapter
from pathlib import Path
from typing import Literal
from ase.io import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
import pandas as pd

from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    calculational_setup, read_vasp_results, clean_up_vasp_directory)
from gnome_dreams_oer_screening.explog.relational_frame import (
    RelationalFrame)
from gnome_dreams_oer_screening import get_my_jobs_status
    

class Job_handler():

    def __init__(self, 
        rframe: RelationalFrame,
        vasp_calculation_dir: Path,
        mode: Literal["test", "slurm_test", 'MLIP_test',
                      "production_test", "production"],
        prefix_vasp_dirs_with_prefix: bool = True,
        sleep_min: int = 2
        ) -> None:

        # Create the directory if it doesn't exist
        if not vasp_calculation_dir.exists():
            vasp_calculation_dir.mkdir(parents=True)

        self.vasp_dir = vasp_calculation_dir
        self.rframe = rframe
        self.mode = mode
        self.calc_setup = calculational_setup(mode=mode)

        current_dir = Path(__file__).resolve().parent
        pysqa_yaml_path = current_dir / 'pysqa_setup/'
        self.queueadapter = QueueAdapter(directory=pysqa_yaml_path)

        if prefix_vasp_dirs_with_prefix:
            now = datetime.now()
            # Format as string: YYYYMMDD_HHMMSS
            timestamp = now.strftime("%Y%m%d_%H%M%S") + "_"
        else:
            timestamp = ""
        self.vasp_dir_prefix = timestamp

        if self.mode == "slurm_test":
            self.sleep_min = sleep_min

    def enter_process(
        self,
        candidate_id,
        job_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        termination_index: int | None = None,
        site_index: int | None = None,
        note: str | None = None
        ) -> str | list[str]:
        """
        Register a new calculation process for a candidate.

        A new process row is added to the relational-frame `processes`
        table with status set to `'un-submitted'`, a generated
        `process_id`, and a unique VASP working directory path.

        The method rejects creating a new process when a non-failed
        process already exists for the same candidate and the same
        `(job_type, termination_index, site_index)` combination.

        Args:
            candidate_id:
                Candidate identifier in the `candidates` table.
            job_type:
                Type of calculation to register.
            termination_index:
                Surface termination index for surface/adsorption jobs.
            site_index:
                Adsorption-site index for adsorption jobs.
            note:
                Optional process note stored in `processNote`.

        Returns:
            str | list[str]:
                For non-OH jobs, the newly assigned process id.
                For `OH_adsorption`, a list of process ids (one for each
                OH initialization structure).

        Raises:
            RuntimeError:
                If the generated process id already exists, or if a
                non-failed matching process already exists for the
                candidate.
            KeyError:
                If `candidate_id` does not exist in the candidates
                table.
        """

        # Get the candidates and processes rows from:
        cand_row = self.rframe.candidates[candidate_id]
        cand_processes = cand_row.processes

        # --- Test if calculation is appropriate -----------------------
        pdf = cand_processes.df.copy()

        # Base filters
        mask = (
            (pdf["status"] != "failed") &
            (pdf["job_type"] == job_type)
        )

        # Termination filter
        if termination_index is None:
            termination_mask = pdf["termination_index"].isna()
        else:
            termination_mask = pdf["termination_index"] \
                == termination_index

        # Site filter
        if site_index is None:
            site_mask = pdf["site_index"].isna()
        else:
            site_mask = pdf["site_index"] == site_index

        pdf = pdf[mask & termination_mask & site_mask]

        # If non-finished or completed calculation exists: non-valied:
        non_valid_calc = not pdf.empty 
        if non_valid_calc:
            states = pdf["status"].values
            message = f"Candidate {candidate_id} has a non-failed " \
            f"calculation of type {job_type} with status in " \
            f"{states}. This calculation is therefore invalid."
            raise RuntimeError(message)
        # --------------------------------------------------------------

        # --- Sanity checks against study object state -----------------
        study_obj = cand_row.df["study_obj"].values[0]

        if job_type == "bulk_relaxation":
            # Test that termination and site indices are not provided:
            if termination_index is not None or site_index is not None:
                raise ValueError(
                    "bulk_relaxation does not use termination_index "
                    "or site_index."
                )
            # Test if the relaxed bulk already is set for the study:
            if study_obj.get_bulk_atoms() is not None:
                raise RuntimeError(
                    f"Candidate {candidate_id} already has a relaxed "
                    "bulk structure set. A new bulk_relaxation process "
                    "must not be created."
                )

        elif job_type == "surface_relaxation":
            # Test termination index is provided:
            if termination_index is None:
                raise ValueError(
                    "surface_relaxation requires termination_index."
                )
            # Test site index is not provided:
            if site_index is not None:
                raise ValueError(
                    "surface_relaxation does not use site_index."
                )

            # Test if surface study is initialized for the study:
            surface_studies = study_obj.get_surface_studies()
            if termination_index not in surface_studies:
                raise ValueError(
                    "No initialized surface study exists for "
                    f"termination_index={termination_index}."
                )

            # Test if relaxed surface already is set for the study:
            surface_study = surface_studies[termination_index]
            if surface_study.base_surface_relaxed is not None:
                raise RuntimeError(
                    "A relaxed surface is already set for "
                    f"termination_index={termination_index}. "
                    "A new surface_relaxation process must not be "
                    "created."
                )

        elif job_type in ["O_adsorption", "OH_adsorption"]:

            # Test that termination index is provided:
            if termination_index is None:
                raise ValueError(
                    f"{job_type} requires termination_index."
                )
            
            # Test that site index is provided:
            if site_index is None:
                raise ValueError(
                    f"{job_type} requires site_index."
                )

            # Test if surface study is initialized for the study:
            surface_studies = study_obj.get_surface_studies()
            if termination_index not in surface_studies:
                raise ValueError(
                    "No initialized surface study exists for "
                    f"termination_index={termination_index}."
                )
            surface_study = surface_studies[termination_index]

            # Test if ad-site study is initialized for the study:
            adsite_studies = surface_study.\
                get_adsorption_site_studies_dict()
            if site_index not in adsite_studies:
                raise ValueError(
                    "No initialized adsorption-site study exists for "
                    f"site_index={site_index} on "
                    f"termination_index={termination_index}."
                )
            ad_study = adsite_studies[site_index]

            if job_type == "O_adsorption" and \
                    ad_study.relaxed_O_atoms is not None:
                raise RuntimeError(
                    "A relaxed O structure has already been accepted "
                    f"for site_index={site_index} on "
                    f"termination_index={termination_index}. "
                    "A new O_adsorption process must not be created."
                )
            
            # TODO: What should be done about the OH calculations???
            # if job_type == "OH_adsorption" and \
            #         ad_study.relaxed_OH_atoms is not None:


        # --------------------------------------------------------------

        def _add_single_process(single_note: str | None = None) -> str:
            # Format the new process row:
            new_process_row = {
                "job_type": job_type,
                "candidate_id": candidate_id,
                "status": "un-submitted",
                "termination_index": termination_index,
                "site_index": site_index,
                "processNote": single_note,
            }

            # Get total number of processes, and assign as id:
            new_p_id = str(len(self.rframe.processes.df))  # new process id

            # Test if new process id already exists (should not happen if
            # process ids are assigned correctly):
            if new_p_id in self.rframe.processes.df["process_id"].values:
                raise RuntimeError(
                    f"Process id {new_p_id} already exists. "
                    "This should not be possible. Critical!!!"
                )

            new_process_row["process_id"] = new_p_id
            new_process_row["VASP_dir"] = str(
                self.vasp_dir
                / f"{self.vasp_dir_prefix}{candidate_id}_"
                f"{job_type}_{new_process_row['process_id']}"
            )

            self.rframe.processes.add_row(new_process_row,
                                          allow_update=True)
            return new_p_id

        # For OH adsorption, create one process per OH initialization.
        if job_type == "OH_adsorption":
            oh_inits = ad_study.get_OH_adsorbed_atoms()
            process_ids = []
            for oh_init_index in range(len(oh_inits)):
                if note is None:
                    process_note = f"OH_init_index={oh_init_index}"
                else:
                    process_note = (
                        f"{note} | OH_init_index={oh_init_index}"
                    )
                process_ids.append(_add_single_process(process_note))
            return process_ids

        return _add_single_process(note)
    
    def submit_vasp_calculation(
        self,
        process_id: str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'],
        # vasp_cmd: list = ["mpiexec", "vasp_std"],
        # max_errors: int = 20
        ) -> None:
        
        # Convert process_id to string if it is an integer:
        if isinstance(process_id, int):
            process_id = str(process_id)
        # Test if process_id is now a string:
        if not isinstance(process_id, str):
            raise ValueError("process_id must be a string or an " \
                "integer.")

        # Get the process row for the candidate and process id:
        process_row = self.rframe.processes.df.loc[
            (self.rframe.processes.df['process_id'] == process_id)
        ]
        # Test if process row is empty:
        if process_row.empty:
            raise ValueError(f"No process found for process_id \
                             {process_id}.")
        # Test if un-submitted (must be un-submitted to be submitted):
        if process_row['status'].values[0] != 'un-submitted':
            raise RuntimeError(f"Process {process_id}, is not an " \
                f"un-submitted process and cannot be submitted.")

        # Get the candidate id from the process row:
        candidate_id = process_row['candidate_id'].values[0]

        # Get the study object for the candidate:
        study_obr = self.rframe.candidates[candidate_id].df["study_obj"]
        study = study_obr.values[0]        

        # Get the calculation directory from the process row:
        calc_dir = Path(process_row['VASP_dir'].values[0])
    
        # Get the job type from the process row:    
        job_type = process_row['job_type'].values[0]

        # --- Get the initial structure for the calculation ------------
        # --- This inludes checks on the requsted calculation
        if job_type == 'bulk_relaxation': 
            # Test if relaxed structure already exists for the study:   
            if study.get_bulk_atoms() is not None:
                raise RuntimeError(f"Candidate {candidate_id} already "\
                        f"has set a relaxed structure. Cannot submit " \
                        f"this bulk relaxation job again.")

            # Get init structure for bulk relaxation:
            init_structure = study.get_init_structure()
            self.calc_setup.bulk_relaxation(
                structure = init_structure, directory = calc_dir)

        elif job_type in ["surface_relaxation", "O_adsorption", 
                          'OH_adsorption']:

            # Need termination index for surface and adsorption jobs:
            termination_index = \
                process_row['termination_index'].values[0]

            # Test if surface studies have been initialized:
            surface_studies = study.get_surface_studies()
            if surface_studies is None:
                raise RuntimeError(f"Surface studies have not been " \
                    f"initialized for candidate  {candidate_id}. " \
                    f"Cannot submit surface relaxation job.")
            
            # Test if termination index is in surface studies:
            if termination_index not in surface_studies.keys():
                raise ValueError(f"Termination index has not been " \
                    f"initialized for candidate {candidate_id}. " \
                    f"Cannot submit surface relaxation job.")
            
            surface_study = surface_studies[termination_index]

            if job_type == "surface_relaxation":

                # Test if relaxed surface already exists for the study:
                if surface_study.get_relaxed_surface() is not None:
                    raise RuntimeError(f"Candidate {candidate_id} " \
                        f"already has a relaxed surface. Cannot submit"\
                            f"this surface relaxation job again.")

                # Get init structure for surface relaxation:
                init_atoms = surface_study.get_non_relaxed_surface()
                init_structure \
                    = AseAtomsAdaptor.get_structure(init_atoms)

            elif job_type in ["O_adsorption", 'OH_adsorption']:

                # Get site index for adsorption jobs:
                site_index = process_row['site_index'].values[0]
                if pd.isna(site_index):
                    raise ValueError(
                        f"{job_type} requires a valid site_index in "
                        "the process row before submission."
                    )
                site_index = int(site_index)

                adsite_studies = surface_study.\
                    get_adsorption_site_studies_dict()
                if site_index not in adsite_studies:
                    raise ValueError(
                        "No initialized adsorption-site study exists "
                        f"for site_index={site_index} on "
                        f"termination_index={termination_index}."
                    )
                ad_study = adsite_studies[site_index]
                
                if job_type == "O_adsorption":
                    # Defensive re-check in case process creation checks
                    # were bypassed or state changed after creation.
                    if ad_study.relaxed_O_atoms is not None:
                        raise RuntimeError(
                            "A relaxed O structure has already been "
                            "accepted for this adsorption site. "
                            "Cannot submit O_adsorption again."
                        )
                    init_atoms = ad_study.get_atoms_for_O_calc()
                    init_structure \
                        = AseAtomsAdaptor.get_structure(init_atoms)
                elif job_type == 'OH_adsorption':
                    oh_candidates = ad_study.get_OH_adsorbed_atoms()
                    if len(oh_candidates) == 0:
                        raise RuntimeError(
                            "No OH initialization structures are "
                            "available for submission."
                        )
                    process_note = process_row["processNote"].values[0]
                    oh_init_index = None
                    if isinstance(process_note, str):
                        token = "OH_init_index="
                        if token in process_note:
                            tail = process_note.split(token, 1)[1]
                            tail = tail.split("|", 1)[0].strip()
                            oh_init_index = int(tail)
                    if oh_init_index is None:
                        raise ValueError(
                            "OH_adsorption process row is missing "
                            "OH_init_index in processNote."
                        )
                    if oh_init_index < 0 or oh_init_index >= len(oh_candidates):
                        raise ValueError(
                            f"OH_init_index={oh_init_index} is out of "
                            "range for available OH initializations "
                            f"(n={len(oh_candidates)})."
                        )
                    init_atoms = oh_candidates[oh_init_index]
                    init_structure \
                        = AseAtomsAdaptor.get_structure(init_atoms)

            self.calc_setup.surface_relaxation(
                structure = init_structure, directory = calc_dir
            )
        # --------------------------------------------------------------

        # Dummy initialization for test mode, should be overwriten when
        # not in a-test-mode:
        slurm_id = 42 + int(process_id) 

        if self.mode == "MLIP_test":
            mlip_input_path = calc_dir / "mlip_input.traj"
            mlip_input_atoms = AseAtomsAdaptor.get_atoms(init_structure)
            input_traj = Trajectory(mlip_input_path, "w")
            input_traj.write(mlip_input_atoms)
            input_traj.close()

        if self.mode != 'test':
            package_root = Path(__file__).resolve().parents[1]

            # Set up of submission command and parameters for slurm 
            # submission. Needed when not in "test" mode:
            if self.mode == 'slurm_test':
                sleep_script = package_root / "niflheim" / "sleep_job.py"
                if not sleep_script.exists():
                    raise FileNotFoundError(
                        f"Sleep-job script not found: {sleep_script}"
                    )
                partition = 'xeon24el8_test'
                cores, nodes = 1, 1
                walltime = f"00:10:00"
                submit_command = (
                    f'python3 "{sleep_script}" {self.sleep_min}'
                )

            elif self.mode == 'MLIP_test': # TODO NOTE NOTE NOTE -- need to reconsider this path
                mlip_script = package_root / "niflheim" / "mlip_job.py"
                if not mlip_script.exists():
                    raise FileNotFoundError(
                        f"MLIP-job script not found: {mlip_script}"
                    )
                partition = 'xeon24el8_test'
                cores, nodes = 1, 1
                walltime = f"00:10:00"
                submit_command = (
                    f'python3 "{mlip_script}" "{calc_dir.resolve()}"'
                )

            else:
                run_vasp_script = package_root / "vasp" / "run_vasp.py"
                if not run_vasp_script.exists():
                    raise FileNotFoundError(
                        f"run_vasp script not found: {run_vasp_script}"
                    )

                if partition == 'auto':
                    # TODO, could make a smart auto-partition
                    if len(init_structure) < 100:
                        partition = 'xeon40el8'
                    else:
                        partition = 'xeon56'
                
                if partition == 'xeon56':
                    cores, nodes = 56, 1
                elif partition == 'xeon40el8':
                    cores, nodes = 40, 1
                elif partition == 'xeon24el8':
                    cores, nodes = 24, 1
                else:
                    raise ValueError(f"Partition {partition} not " \
                                     f"recognized.")

                walltime = "50:00:00"
                submit_command = (
                    f'python3 "{run_vasp_script}" "{calc_dir.resolve()}"'
                )

            slurm_id = self.queueadapter.submit_job(
                command = submit_command,
                slurm_out_name = calc_dir / "job_log",
                partition = partition,
                cores = cores,
                nodes = nodes,
                walltime = walltime,
            )

        self.rframe.candidates[candidate_id].processes[process_id]\
            .status = 'pending'
        self.rframe.candidates[candidate_id].processes[process_id]\
            .slurmID = str(slurm_id)

    def _update_ideal_over_pot(self, candidate_id) -> None:
        cand_row = self.rframe.candidates[candidate_id]
        study = cand_row.df["study_obj"].values[0]
        sur_studies = study.get_surface_studies()

        # Gather all currently available adsorption-site rows across
        # surfaces.
        frames = []
        for sur_study in sur_studies.values():
            sur_study.update_adsorption_energies()
            ad_sites_df = sur_study.get_adsorption_sites_df()
            if ad_sites_df is None:
                continue
            if len(ad_sites_df) == 0:
                continue
            frames.append(ad_sites_df.copy())

        ad_df = pd.concat(frames, ignore_index=True
                          ) if len(frames) > 0 else None
        cand_cols = set(self.rframe.candidates.df.columns.tolist())

        metric_to_candidate = {
            "ideal overpotential": "idealOverPotential",
            "overpotential from OH-OOH scaling relation":
                "Overpotential_from_scaling",
        }
        for metric_col, cand_col in metric_to_candidate.items():
            if cand_col not in cand_cols:
                continue

            value = np.nan
            if ad_df is not None and metric_col in ad_df.columns:
                metric_series = pd.to_numeric(
                    ad_df[metric_col], errors="coerce"
                )
                min_value = metric_series.min(skipna=True)
                if pd.notna(min_value):
                    value = float(np.round(min_value, 2))

            self.rframe.candidates.set_value(candidate_id, cand_col, 
                                             value)

        
    def update_log(self) -> dict:

        updates = {} # Save all updates in dict

        # --- Identify calculations to update --------------------------
        if self.mode == 'test':
            # For the test mode, all pending can be set to updating
            # (no real job are run, and no real status exists)
            for idx, pdf_row in self.rframe.processes.df.iterrows():
                if pdf_row['status'] == 'pending':
                    self.rframe.processes.df.at[idx, 'status'] = \
                        'updating'
                    
        else:
            # if not in test mode, need to check the real status of the 
            # jobs in the slurm queue, via get_my_jobs_status:
            slurm_queue = get_my_jobs_status()
            reduced_pdf = self.rframe.processes.df.copy()
            reduced_pdf = reduced_pdf[reduced_pdf['status'].isin(
                                                ['running', 'pending'])]

            # Update all statuses, based on current slurm queue
            for idx, row in reduced_pdf.iterrows():
                slurm_id = row['slurmID']
                if slurm_id in slurm_queue.keys():
                    new_status = slurm_queue[slurm_id].lower()
                    
                    # If node is powering on, the job set to pending:
                    # (ignoring configuring as it is irrelevant)
                    if new_status == 'configuring':
                        new_status = 'pending'

                    # set current status:
                    updates[row['process_id']] = new_status
                else:
                    # If job is not in the slurm queue, an update can
                    # be made (either completed or failed considered 
                    # later):
                    new_status = 'updating'
                self.rframe.processes.df.at[idx, 'status'] = new_status

        # --- Read results and mark as completed/failed ----------------
        if self.mode in ['test', 'slurm_test']:
            for idx, pdf_row in self.rframe.processes.df.iterrows():
                if pdf_row['status'] == 'updating':
                    
                    vasp_dir = Path(pdf_row['VASP_dir'])
                    traj = Trajectory(vasp_dir / 'fake_result.traj')
                    atoms = traj[-1]
                    E = atoms[0].charge # Not needed-at least currently

                    candt_id = pdf_row['candidate_id']
                    cand_row = self.rframe.candidates[candt_id]
                    study = cand_row.df['study_obj'].values[0]
                    
                    job_type = pdf_row['job_type']
                    termination_index = pdf_row['termination_index']
                    site_index = pdf_row['site_index']

                    if job_type == 'bulk_relaxation':
                        study.set_relaxed_structure(atoms)
                    else:
                        surface_studies = study.get_surface_studies()
                        surface_so = surface_studies[termination_index]
                        
                        if job_type == 'surface_relaxation':                            
                            surface_so.set_relaxed_base_surface(atoms,E)
                        else:                                
                            ad_site_dict = surface_so.\
                                get_adsorption_site_studies_dict()
                            ad_study = ad_site_dict[site_index]
                            if job_type == 'O_adsorption':
                                ad_study.set_relaxed_O_calc(atoms, E)
                            elif job_type == 'OH_adsorption':
                                ad_study.set_relaxed_OH_calc(
                                    atoms, energies=[E]
                                )
                       
                            self._update_ideal_over_pot(candt_id)

                    self.rframe.processes.df.at[idx, 'status'] = \
                        'completed'
                    
                    updates[pdf_row['process_id']] = 'completed'

        else:
            r_pdf = self.rframe.processes.df.copy()
            r_pdf = r_pdf[r_pdf['status'] == 'updating']
            for idx, pdf_row in r_pdf.iterrows():
                vasp_dir = Path(pdf_row['VASP_dir'])

                if self.mode == 'MLIP_test':
                    try:
                        traj = Trajectory(vasp_dir / "mlip_result.traj")
                        if len(traj) == 0:
                            raise RuntimeError(
                                "MLIP result trajectory is empty."
                            )
                        atoms = traj[-1]
                        traj.close()
                        clean_up_vasp_directory(vasp_dir)

                    except Exception:
                        self.rframe.processes.df.at[idx, 'status'] \
                            = 'failed'
                        updates[pdf_row['process_id']] = 'failed'
                        continue

                else:
                    try:
                        atoms = read_vasp_results(vasp_dir)[-1]
                        clean_up_vasp_directory(vasp_dir)
                    except Exception:
                        self.rframe.processes.df.at[idx, 'status'] \
                            = 'failed'
                        updates[pdf_row['process_id']] = 'failed'
                        continue
                
                job_type = pdf_row['job_type']
                termination_index = pdf_row['termination_index']
                site_index = pdf_row['site_index']
                candt_id = pdf_row['candidate_id']
                cand_row = self.rframe.candidates[candt_id]
                study = cand_row.df['study_obj'].values[0]
                unrecoverable_reason = None

                if job_type == 'bulk_relaxation':
                    study.set_relaxed_structure(atoms)
                else:
                    surface_studies = study.get_surface_studies()
                    surface_study = surface_studies[termination_index]
                    
                    if job_type == 'surface_relaxation':                            
                        success = surface_study.\
                            set_relaxed_base_surface(atoms)
                        if not success:
                            unrecoverable_reason = (
                                "Relaxed surface dissociated and this" \
                                " surface can therefore not be used " \
                                "for adsorption studies."
                            )
                    else:                                
                        ad_site_dict = surface_study.\
                            get_adsorption_site_studies_dict()
                        ad_study = ad_site_dict[site_index]
                        if job_type == 'O_adsorption':
                            result = ad_study.set_relaxed_O_calc(atoms)
                            if isinstance(result, tuple):
                                success, message = result
                            else:
                                success, message = bool(result), None
                            if not success:
                                unrecoverable_reason = (
                                    message
                                    if message is not None
                                    else "Relaxed O structure rejected "
                                    "by study sanity checks."
                                )
                        elif job_type == 'OH_adsorption':
                            result = ad_study.set_relaxed_OH_calc(atoms)
                            if isinstance(result, tuple):
                                success, message = result
                            else:
                                success, message = bool(result), None
                            if not success:
                                unrecoverable_reason = (
                                    message
                                    if message is not None
                                    else "Relaxed OH structure rejected"
                                    "by study sanity checks."
                                )
                    
                        if unrecoverable_reason is None:
                            self._update_ideal_over_pot(candt_id)

                if unrecoverable_reason is not None:
                    self.rframe.processes.df.at[idx, 'status'] = \
                        "unrecoverable: " + unrecoverable_reason
                    updates[pdf_row['process_id']] = 'unrecoverable'
                    continue

                self.rframe.processes.df.at[idx, 'status'] = \
                    'completed'
                updates[pdf_row['process_id']] = 'completed'

        return updates
