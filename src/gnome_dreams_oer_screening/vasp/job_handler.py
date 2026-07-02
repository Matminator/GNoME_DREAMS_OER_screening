from datetime import datetime
from pysqa import QueueAdapter
from pathlib import Path
from typing import Literal
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
import pandas as pd

from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    calculational_setup, read_vasp_results, clean_up_vasp_directory)
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
    extract_mag_config_index,
    MagneticEnumerationError,
)
from gnome_dreams_oer_screening.vasp.job_validation import (
    validate_enter_process)
from gnome_dreams_oer_screening.explog.relational_frame import (
    RelationalFrame)
from gnome_dreams_oer_screening import get_my_jobs_status
    

class Job_handler():
    PROCESS_METRIC_COLUMNS = [
        "G(O)",
        "G(O) deviation",
        "G(OH)",
        "G(OH) deviation",
        "G(OOH) from scaling relation",
        "G(OOH, scaling) deviation",
        "ideal overpotential",
        "overpotential from OH-OOH scaling relation",
    ]

    def __init__(self,
        rframe: RelationalFrame,
        vasp_calculation_dir: Path,
        mode: Literal["test", "slurm_test", 'MLIP_test',
                      "production_test", "production"],
        prefix_vasp_dirs_with_prefix: bool = True,
        sleep_min: int = 2,
        reject_if_failed_exists: bool = False,
        require_relaxed_o_for_oh: bool = False,
        magnetic_bulk_n_max: int = 4,
        use_dipole_correction: bool = True,
        legacy_disposition_exempt_ids: set | None = None,
        disposition_decisions=None,
        ) -> None:

        # Create the directory if it doesn't exist
        if not vasp_calculation_dir.exists():
            vasp_calculation_dir.mkdir(parents=True)

        self.vasp_dir = vasp_calculation_dir
        self.rframe = rframe
        self.mode = mode
        self.use_dipole_correction = use_dipole_correction
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
        self.default_reject_if_failed_exists = \
            reject_if_failed_exists
        self.default_require_relaxed_o_for_oh = \
            require_relaxed_o_for_oh
        self.magnetic_bulk_n_max = magnetic_bulk_n_max
        # Process ids exempt from the disposition coverage check (the legacy
        # backlog, frozen at the disposition-gate rollout; see var.py). Consulted
        # only by candidate_outstanding -- the agent always sees all ids.
        self.legacy_disposition_exempt_ids = set(
            int(x) for x in (legacy_disposition_exempt_ids or ())
        )
        # Allowed Decision vocabulary (sourced from var.py, passed at init);
        # used by Explog.update_disposition_info validation + message.
        self.disposition_decisions = tuple(disposition_decisions or ())

    def enter_process(
        self,
        candidate_id,
        job_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        termination_index: int | None = None,
        site_index: int | None = None,
        note: str | None = None,
        reject_if_failed_exists: bool | None = None,
        require_relaxed_o_for_oh: bool | None = None,
        ) -> int | list[int]:
        """
        Register a new calculation process for a candidate.

        A new process row is added to the relational-frame `processes`
        table with status set to `'un-submitted'`, a generated
        `process_id`, and a unique VASP working directory path.

        Validation is delegated to `validate_enter_process(...)` before
        creating process rows. Duplicate policy is controlled by
        `reject_if_failed_exists` (or the handler default when `None`).

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
            reject_if_failed_exists:
                If True, duplicate matching processes are rejected even
                when their status is `failed`. If `None`, the handler
                default from initialization is used.
            require_relaxed_o_for_oh:
                If True, `OH_adsorption` is rejected unless a relaxed O
                structure has already been accepted for the site. If
                `None`, the handler default from initialization is used.

        Returns:
            int | list[int]:
                For non-OH jobs, the newly assigned process id.
                For `OH_adsorption`, a list of process ids (one for each
                OH initialization structure).

        Raises:
            RuntimeError:
                If the generated process id already exists, if a
                duplicate matching process is disallowed by policy, or
                if study state disallows the requested process.
            ValueError:
                If required indices are missing/invalid for the
                requested `job_type`, or required study objects are not
                initialized.
            KeyError:
                If `candidate_id` does not exist in the candidates
                table.
        """
        if reject_if_failed_exists is None:
            reject_if_failed_exists = \
                self.default_reject_if_failed_exists
        if require_relaxed_o_for_oh is None:
            require_relaxed_o_for_oh = \
                self.default_require_relaxed_o_for_oh

        # Shared pre-checks from validator (non-mutating). Keep this
        # method raise-based by converting failed results to exceptions.
        validation_result = validate_enter_process(
            self.rframe,
            candidate_id=candidate_id,
            job_type=job_type,
            termination_index=termination_index,
            site_index=site_index,
            reject_if_failed_exists=reject_if_failed_exists,
            require_relaxed_o_for_oh=require_relaxed_o_for_oh,
        )
        if not validation_result.ok:
            validation_result.raise_for_error()


        ad_study = validation_result.context.get("ad_study")
        if job_type in ["O_adsorption", "OH_adsorption"] and \
                ad_study is None:
            raise RuntimeError(
                "Internal error: validation contract violation. "
                "Missing ad_study in validate_enter_process(...) "
                "context for adsorption job_type="
                f"{job_type}. This indicates an internal "
                "implementation bug."
            )

        def _add_single_process(single_note: str | None = None) -> int:
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
            new_p_id = len(self.rframe.processes.df)  # new process id

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

        # For bulk relaxation, enumerate magnetic configs and create one
        # process per config. Configs are stored once on the study object.
        if job_type == "bulk_relaxation":
            study_obj = self.rframe.candidates[candidate_id].df["study_obj"].values[0]
            try:
                configs = enumerate_magnetic_structures(
                    study_obj.get_init_structure(), self.magnetic_bulk_n_max
                )
            except Exception as e:
                # The structure cannot be set up for the magnetic bulk batch -- pymatgen
                # raises a generic ValueError (e.g. "Too many magnetic sites to sensibly
                # perform enumeration."). Re-raise as a SPECIFIC error so the submit tool
                # can mark this candidate failed, WITHOUT keying on a generic exception
                # type that the unrelated add_process failures (duplicate / validation /
                # bad id) would also match.
                raise MagneticEnumerationError(
                    f"Magnetic-ordering enumeration failed for candidate "
                    f"{candidate_id}: {e}"
                ) from e
            study_obj.store_mag_bulk_configs(configs)
            process_ids = []
            for i, config in enumerate(configs):
                process_note = (
                    f"mag_config_index={i} | ordering={config['ordering']}"
                )
                if note is not None:
                    process_note = f"{note} | {process_note}"
                process_ids.append(_add_single_process(process_note))
            return process_ids

        # For OH adsorption, create one process per OH initialization.
        if job_type == "OH_adsorption":
            oh_inits = ad_study.get_OH_adsorbed_atoms()
            ad_study.set_oh_batch_finalized(False)
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
        process_id: int,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'],
        # vasp_cmd: list = ["mpiexec", "vasp_std"],
        # max_errors: int = 20
        ) -> None:

        # NOTE (planned validation refactor):
        # Replace the inline submission checks below with:
        #   result = self.can_submit_process(process_id)
        #   if not result.ok:
        #       result.raise_for_error()
        # Then read candidate/study/job_type/indices from
        # `result.context` to avoid duplicated lookups and to keep the
        # same validation logic reusable outside Job_handler.

        # Enforce integer process id:
        if isinstance(process_id, np.integer):
            process_id = int(process_id)
        if not isinstance(process_id, int):
            raise ValueError("process_id must be an integer.")

        # Get the process row for the candidate and process id:
        process_row = self.rframe.processes.df.loc[
            self.rframe.processes.df['process_id'] == process_id
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
            # get_bulk_atoms() returns the RELAXED bulk — not the init structure.
            # If already set, the workflow has gone wrong.
            if study.get_bulk_atoms() is not None:
                raise RuntimeError(
                    f"Candidate {candidate_id} already has a relaxed bulk "
                    "structure set. Cannot submit bulk_relaxation again."
                )
            process_note = process_row['processNote'].values[0]
            mag_idx = extract_mag_config_index(process_note)
            if mag_idx is None:
                raise RuntimeError(
                    f"Process {process_id}: bulk_relaxation is missing "
                    "mag_config_index in processNote."
                )
            configs = study.get_mag_bulk_configs()
            if configs is None:
                raise RuntimeError(
                    f"Candidate {candidate_id}: magnetic bulk configs not found. "
                    "enter_process must be called before submit_vasp_calculation."
                )
            config = configs[mag_idx]
            init_structure = config["structure"]
            # config['structure'] already has site_properties['magmom'] set by
            # enumerate_magnetic_structures; VaspInputSet writes MAGMOM automatically.
            self.calc_setup.bulk_relaxation(
                structure=init_structure, directory=calc_dir
            )

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
                init_structure = AseAtomsAdaptor.get_structure(init_atoms)
                # Inject bulk-derived slab magmoms so VASP starts from a
                # sensible magnetic state.
                slab_magmom = surface_study.get_slab_magmom()
                if slab_magmom is not None:
                    for site, m in zip(init_structure.sites, slab_magmom):
                        site.properties["magmom"] = m

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
                    init_structure = AseAtomsAdaptor.get_structure(init_atoms)
                    o_magmom = ad_study.get_O_init_magmom()
                    if o_magmom is not None:
                        for site, m in zip(init_structure.sites, o_magmom):
                            site.properties["magmom"] = m
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
                    init_structure = AseAtomsAdaptor.get_structure(init_atoms)
                    oh_magmom = ad_study.get_OH_init_magmom()
                    if oh_magmom is not None:
                        for site, m in zip(init_structure.sites, oh_magmom):
                            site.properties["magmom"] = m

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

            elif self.mode == 'MLIP_test':
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
                if job_type == 'bulk_relaxation':
                    run_vasp_script = package_root / "vasp" / "run_vasp.py"
                    if not run_vasp_script.exists():
                        raise FileNotFoundError(
                            f"run_vasp script not found: {run_vasp_script}"
                        )
                    submit_command = (
                        f'python3 "{run_vasp_script}" "{calc_dir.resolve()}"'
                    )
                else:  # surface_relaxation, O_adsorption, OH_adsorption
                    run_vasp_surface_script = package_root / "vasp" / "run_vasp_surface.py"
                    if not run_vasp_surface_script.exists():
                        raise FileNotFoundError(
                            f"run_vasp_surface script not found: {run_vasp_surface_script}"
                        )
                    submit_command = (
                        f'python3 "{run_vasp_surface_script}" "{calc_dir.resolve()}" '
                        f'{self.use_dipole_correction}'
                    )

                if partition == 'auto':
                    # NOTE, could make a smart auto-partition
                    if len(init_structure) < 85:
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

    def _update_candidate_oer_metrics(self, candidate_id) -> None:
        """
        Update candidate-level OER summary metrics from site studies.

        The method recomputes adsorption metrics on all initialized
        surface studies and stores best available values in the
        candidate table (minimum deviations/overpotentials).
        """
        cand_row = self.rframe.candidates[candidate_id]
        study = cand_row.df["study_obj"].values[0]
        sur_studies = study.get_surface_studies()

        # Gather all currently available adsorption-site rows across
        # surfaces.
        all_ad_site_dfs = []
        for sur_study in sur_studies.values():
            ad_sites_df = sur_study.get_adsorption_sites_df()
            if ad_sites_df is None:
                continue

            sur_study.update_adsorption_energies()
            ad_sites_df = sur_study.get_adsorption_sites_df()
            if ad_sites_df is None:
                continue
            if len(ad_sites_df) == 0:
                continue
            all_ad_site_dfs.append(ad_sites_df.copy())

        # Concatenate all site dfs into one df:
        ad_df = pd.concat(all_ad_site_dfs, ignore_index=True
                          ) if len(all_ad_site_dfs) > 0 else None
        
        # Get all cols:
        cand_cols = set(self.rframe.candidates.df.columns.tolist())

        # Map source metric names in adsorption-site DataFrames (ad_df)
        # to summary columns in the candidates table.
        # This keeps detailed site-level labels while supporting stable
        # candidate-level names used elsewhere:
        metric_to_candidate = {
            "G(O) deviation": "G(O) deviation",
            "ideal overpotential": "idealOverPotential",
            "overpotential from OH-OOH scaling relation":
                "Overpotential_from_scaling",
        }

        # For each mapping pair:
        # - metric_col: source metric in ad_df
        # - cand_col: summary field in candidates
        # Write the best available value across all sites
        # (minimum non-NaN) so candidates can be compared/ranked.
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

    @staticmethod
    def _is_terminal_process_status(status) -> bool:
        if not isinstance(status, str):
            return False
        return status in {"completed", "failed"} or status.startswith(
            "unrecoverable"
        )

    def _recompute_candidate_progress(self) -> None:
        """
        Refresh the per-candidate progress projection on the candidates table.

        For every candidate, derive the per-stage started/finalized counts and
        the active/failed ``state`` purely from the processes table, in a single
        whole-frame pass. Reads only lightweight scalar columns of the processes
        df and writes 9 scalar columns onto the candidates df in place — it never
        reads or copies the heavy ``study_obj``. Called as the final step of
        ``update_log``; idempotent and self-healing (it creates the columns if a
        resumed checkpoint predates them).

        Counts are per LOGICAL UNIT, not per raw DFT sub-job:
          - bulk:        0/1 per candidate (the magnetic batch is one unit)
          - surface, O:  one DFT job each -> per-row counts
          - OH:          per adsorption (termination, site); its 3 sub-jobs
                         collapse to one unit
        "started" = work initiated; "finalized" = terminal (failures included).
        surface/O/OH counters stay <NA> until the candidate is eligible for that
        stage (the previous stage produced a usable result), so
        "n_X_started == 0" means "ready for stage X". ``state`` latches: once
        'failed' it is never downgraded back to 'active'.
        """
        cdf = self.rframe.candidates.df          # write target (candidates)
        pdf = self.rframe.processes.df           # read source (processes)
        cid = cdf["candidate_id"]                # one id per candidate-row
        jt = pdf["job_type"]                     # job_type per process-row

        # (0) Per-process-row masks. `terminal` mirrors
        # _is_terminal_process_status (vectorized); `completed` = succeeded.
        terminal = (
            pdf["status"].isin(["completed", "failed"])
            | pdf["status"].fillna("").str.startswith("unrecoverable")
        )
        completed = pdf["status"] == "completed"

        # Count process-rows-per-candidate for a row-mask, aligned to the
        # candidates table (candidates with no match -> 0), as nullable Int64.
        def per_cand(mask) -> pd.Series:
            counts = pdf.loc[mask, "candidate_id"].value_counts()
            return cid.map(counts).fillna(0).astype("Int64")

        # nullable/object bool Series -> Int64 0/1 (NA -> 0).
        def to01(bool_series) -> pd.Series:
            return (
                bool_series.fillna(False).astype(bool).astype("int64").astype("Int64")
            )

        # (1) INTERNAL success signals (not exposed): a stage is eligible only
        # when the PREVIOUS stage produced a usable (completed) result.
        succ_bulk = per_cand((jt == "bulk_relaxation") & completed)
        succ_surf = per_cand((jt == "surface_relaxation") & completed)
        succ_O = per_cand((jt == "O_adsorption") & completed)

        # (2) Bulk magnetic batch: the relaxed-bulk "winner" exists only once all
        # bulk sub-jobs are terminal and at least one succeeded.
        bulk_mask = jt == "bulk_relaxation"
        all_bulk_terminal = (
            cid.map(
                terminal[bulk_mask]
                .groupby(pdf.loc[bulk_mask, "candidate_id"])
                .all()
            )
            .astype("boolean")  # object {True,False,NaN} -> nullable boolean
            .fillna(False)      # fill on a real dtype -> no silent object downcast
            .astype(bool)
        )
        has_bulk = per_cand(bulk_mask) >= 1
        bulk_winner = all_bulk_terminal & (succ_bulk >= 1)
        failed_now = all_bulk_terminal & has_bulk & (succ_bulk == 0)

        # (3) Eligibility cascade. `gate` keeps the count where eligible, else
        # <NA> (preserving the nullable Int64 dtype).
        surface_open = bulk_winner
        O_open = surface_open & (succ_surf >= 1)
        OH_open = O_open & (succ_O >= 1)

        def gate(series, mask) -> pd.Series:
            return series.where(mask.fillna(False))

        # (4) EXPOSED per-logical-unit counts.
        # bulk: collapse the magnetic batch to a single 0/1 (never gated).
        cdf["n_bulk_started"] = to01(has_bulk)
        cdf["n_bulk_finalized"] = to01(all_bulk_terminal)
        # surface & O: one DFT job each -> per-row counts are already per-unit.
        cdf["n_surface_started"] = gate(
            per_cand(jt == "surface_relaxation"), surface_open
        )
        cdf["n_surface_finalized"] = gate(
            per_cand((jt == "surface_relaxation") & terminal), surface_open
        )
        cdf["n_O_started"] = gate(per_cand(jt == "O_adsorption"), O_open)
        cdf["n_O_finalized"] = gate(
            per_cand((jt == "O_adsorption") & terminal), O_open
        )
        # OH: collapse the 3 sub-jobs per (termination, site) into one unit.
        #   sites_started   = # distinct sites with OH jobs    (per candidate)
        #   sites_finalized = # of those sites with ALL OH sub-jobs terminal
        oh = jt == "OH_adsorption"
        if bool(oh.any()):
            grp = terminal[oh].groupby(
                [
                    pdf.loc[oh, "candidate_id"],
                    pdf.loc[oh, "termination_index"],
                    pdf.loc[oh, "site_index"],
                ],
                dropna=False,
            )
            sites_started = grp.size().groupby(level=0).size()
            sites_finalized = grp.all().groupby(level=0).sum()
        else:
            sites_started = pd.Series(dtype="int64")
            sites_finalized = pd.Series(dtype="int64")
        cdf["n_OH_started"] = gate(
            cid.map(sites_started).fillna(0).astype("Int64"), OH_open
        )
        cdf["n_OH_finalized"] = gate(
            cid.map(sites_finalized).fillna(0).astype("Int64"), OH_open
        )

        # (5) state: 'active' unless the bulk batch settled with no success.
        # Latches to 'failed' by reading the prior value (which may be absent
        # when resuming a pre-change checkpoint -> treat as NA).
        prior = (
            cdf["state"]
            if "state" in cdf.columns
            else pd.Series(pd.NA, index=cdf.index)
        )
        state = pd.Series("active", index=cdf.index, dtype="string")
        state[failed_now.fillna(False)] = "failed"                       # newly failed
        state[(prior.astype("string") == "failed").fillna(False)] = "failed"  # latch
        cdf["state"] = state

    # ------------------------------------------------------------------
    # Disposition coverage engine (single source of truth for Gate 1,
    # needs_disposition_update, and the get_disposition_info tool).
    # ------------------------------------------------------------------

    @staticmethod
    def _na_to_none_int(v):
        """pandas NA/NaN -> None; otherwise a plain int (JSON-friendly units)."""
        if v is None or pd.isna(v):
            return None
        return int(v)

    def _candidate_dispositions(self, candidate_id) -> list:
        """The candidate's disposition_record as a list (empty if absent/NA)."""
        cdf = self.rframe.candidates.df
        if "disposition_record" not in cdf.columns:
            return []
        mask = cdf["candidate_id"] == candidate_id
        if not mask.any():
            return []
        rec = cdf.loc[mask, "disposition_record"].iloc[0]
        return rec if isinstance(rec, list) else []

    def _candidate_finalized_units(self, candidate_id) -> list:
        """Enumerate the candidate's batch-collapsed FINALIZED units.

        Each unit is {job_type, termination_index, site_index, ids:[int]}.
        A unit is finalized iff all its (batch) sub-jobs are terminal:
          bulk    -> one unit (all magnetic sub-jobs collapse together);
          surface -> one unit per terminal row;
          O       -> one unit per terminal row;
          OH      -> one unit per (termination, site) with all 3 sub-jobs
                     terminal (collapsed inline, side-effect-free).
        "finalized"/"terminal" includes failures and unrecoverable.
        """
        pdf = self.rframe.processes.df
        sub = pdf[pdf["candidate_id"] == candidate_id]
        units: list = []
        if sub.empty:
            return units

        # bulk: a single collapsed unit, finalized iff the whole magnetic batch
        # is terminal (reuses the existing batch-terminality check).
        bulk = sub[sub["job_type"] == "bulk_relaxation"]
        if not bulk.empty and self._all_magnetic_bulk_terminal(candidate_id):
            units.append({
                "job_type": "bulk_relaxation",
                "termination_index": None,
                "site_index": None,
                "ids": sorted(int(x) for x in bulk["process_id"].tolist()),
            })

        # surface, O: one DFT job each -> one unit per terminal row.
        for jt in ("surface_relaxation", "O_adsorption"):
            for _, r in sub[sub["job_type"] == jt].iterrows():
                if self._is_terminal_process_status(r["status"]):
                    units.append({
                        "job_type": jt,
                        "termination_index": self._na_to_none_int(
                            r["termination_index"]
                        ),
                        "site_index": self._na_to_none_int(r["site_index"]),
                        "ids": [int(r["process_id"])],
                    })

        # OH: collapse the 3 sub-jobs per (termination, site); finalized iff all
        # terminal. Inline (NOT _refresh_oh_batch_finalization_state, which would
        # mutate the study object) to keep this helper pure for the tools.
        oh = sub[sub["job_type"] == "OH_adsorption"]
        if not oh.empty:
            for (term, site), grp in oh.groupby(
                ["termination_index", "site_index"], dropna=False
            ):
                if grp["status"].apply(self._is_terminal_process_status).all():
                    units.append({
                        "job_type": "OH_adsorption",
                        "termination_index": self._na_to_none_int(term),
                        "site_index": self._na_to_none_int(site),
                        "ids": sorted(int(x) for x in grp["process_id"].tolist()),
                    })
        return units

    def candidate_outstanding(self, candidate_id) -> dict:
        """Outstanding disposition state for one candidate -- the single source
        of truth shared by the disposition tools and _recompute_disposition_state.

        Returns {must_cover, legacy_optional, latest_disposition, has_finalized}:
          - must_cover:      finalized units neither covered nor exempt (the
                             agent must cite >=1 id of each before it may wait);
          - legacy_optional: finalized units uncovered but exempt via the legacy
                             set L (citing them is optional);
          - latest_disposition: last disposition dict, or None;
          - has_finalized:   whether any finalized unit exists at all.
        A unit is COVERED iff a disposition's Summarized_process_id intersects
        its ids (citing any one sub-job covers the batch); EXEMPT iff all its
        ids are in the legacy set L.
        """
        units = self._candidate_finalized_units(candidate_id)
        L = set(getattr(self, "legacy_disposition_exempt_ids", set()) or set())
        recs = self._candidate_dispositions(candidate_id)
        covered: set = set()
        for rec in recs:
            for pid in (rec.get("Summarized_process_id") or []):
                covered.add(int(pid))
        must_cover, legacy_optional = [], []
        for u in units:
            ids = set(u["ids"])
            if ids & covered:
                continue
            if ids <= L:
                legacy_optional.append(u)
            else:
                must_cover.append(u)
        return {
            "must_cover": must_cover,
            "legacy_optional": legacy_optional,
            "latest_disposition": recs[-1] if recs else None,
            "has_finalized": len(units) > 0,
        }

    def _ensure_reduced_formula_column(self) -> None:
        """Self-heal the "Reduced Formula" column (named to match AQ-GNoME's
        own field name exactly, for consistency with OER_data_analasis_v2/
        get_candidate_data/browse_df): create it (NA-filled) if a resumed
        checkpoint predates this feature. This package has no aq_gnome
        dependency, so it can only guarantee the column EXISTS -- actual
        values are (re)populated by the caller (src/tools.py, which has
        AQ-GNoME access) via its own sync step. Never overwrites existing
        values. Called as a step of update_log(), alongside
        _recompute_candidate_progress and _recompute_disposition_state.
        """
        cdf = self.rframe.candidates.df
        if "Reduced Formula" not in cdf.columns:
            cdf["Reduced Formula"] = pd.Series(
                [pd.NA] * len(cdf), index=cdf.index, dtype="string"
            )

    def _recompute_disposition_state(self) -> None:
        """Refresh needs_disposition_update on the candidates table.

        Runs as a step of update_log(), right after
        _recompute_candidate_progress. Self-healing: creates the four
        disposition columns if a resumed checkpoint predates them. Writes ONLY
        needs_disposition_update; `decision` and `ready_for_disposition_update`
        are owned by the disposition tools.
            needs = (must_cover non-empty)
                    OR (no disposition yet AND the candidate has finalized work)
        """
        cdf = self.rframe.candidates.df
        idx = cdf.index
        if "disposition_record" not in cdf.columns:
            cdf["disposition_record"] = pd.Series(
                [[] for _ in range(len(cdf))], index=idx, dtype="object"
            )
        if "decision" not in cdf.columns:
            cdf["decision"] = pd.Series(
                [pd.NA] * len(cdf), index=idx, dtype="string"
            )
        if "ready_for_disposition_update" not in cdf.columns:
            cdf["ready_for_disposition_update"] = pd.Series(
                [False] * len(cdf), index=idx, dtype="boolean"
            )
        if "needs_disposition_update" not in cdf.columns:
            cdf["needs_disposition_update"] = pd.Series(
                [False] * len(cdf), index=idx, dtype="boolean"
            )
        if len(cdf) == 0:
            return
        needs = []
        for cid in cdf["candidate_id"].tolist():
            out = self.candidate_outstanding(cid)
            need = bool(out["must_cover"]) or (
                out["latest_disposition"] is None and out["has_finalized"]
            )
            needs.append(bool(need))
        cdf["needs_disposition_update"] = pd.Series(
            needs, index=idx, dtype="boolean"
        )

    def count_pending(self) -> int:
        """Number of processes in the 'pending' (queued, not yet running) state.

        Gate 2 of wait_for_update compares this against var.QUEUE_MIN_PENDING --
        a deliberate queued backlog so HPC nodes never drain.
        """
        pdf = self.rframe.processes.df
        if "status" not in pdf.columns or len(pdf) == 0:
            return 0
        return int((pdf["status"] == "pending").sum())

    def finalized_exit_ids(self, process_ids) -> set:
        """Batch-aware wait-exit filter for wait_for_update.

        Given the process ids that just reached a terminal status this poll,
        return the FULL id set of every finalized batch unit they belong to:
          - surface/O: a single-job unit -> the id maps to itself;
          - bulk/OH:   ANY one terminal sub-job of a fully-terminal batch
                       expands to that whole unit's ids (so the exit report is
                       complete no matter which poll finished which sub-job);
          - a sub-job whose batch is NOT yet fully terminal contributes nothing,
            so an all-partial poll returns an empty set and the wait continues.
        Reuses _candidate_finalized_units, so it matches the coverage /
        disposition semantics exactly.
        """
        pdf = self.rframe.processes.df
        want = {int(p) for p in process_ids}
        if not want or "process_id" not in pdf.columns:
            return set()
        cands = pdf.loc[pdf["process_id"].isin(want), "candidate_id"].unique()
        exit_ids: set = set()
        for cid in cands:
            for u in self._candidate_finalized_units(cid):
                unit_ids = set(u["ids"])
                if want & unit_ids:        # this poll touched this finalized unit
                    exit_ids |= unit_ids   # -> surface the WHOLE unit
        return exit_ids

    def _get_adsorption_site_study(
        self,
        candidate_id: str,
        termination_index,
        site_index,
    ):
        if pd.isna(termination_index) or pd.isna(site_index):
            raise ValueError(
                "Adsorption-site lookup requires non-missing "
                "termination_index and site_index."
            )

        term_idx = int(termination_index)
        site_idx = int(site_index)
        study = self.rframe.candidates[candidate_id].df[
            "study_obj"
        ].values[0]
        surface_study = study.get_surface_studies()[term_idx]
        return surface_study.get_adsorption_site_studies_dict()[site_idx]

    def _get_adsorption_site_metrics_row(
        self,
        candidate_id: str,
        termination_index,
        site_index,
    ) -> pd.Series | None:
        """
        Return the adsorption-site metrics row for one site, or None.

        The row is taken from the site's updated adsorption-sites
        DataFrame for the given candidate, termination index, and site
        index.
        """
        
        # If nan are passed, no mestrics can be retrieved, return None:
        if pd.isna(termination_index) or pd.isna(site_index):
            return None

        # Get study object from candidate row:
        term_idx = int(termination_index)
        site_idx = int(site_index)
        study = self.rframe.candidates[candidate_id].df[
            "study_obj"
        ].values[0]

        # Get surface study dict:
        surface_studies = study.get_surface_studies()
        # Test that the specific surface study exists:
        if term_idx not in surface_studies:
            return None
        surface_study = surface_studies[term_idx]

        # Get adsorption sites dataframe from surface study:
        ad_sites_df = surface_study.get_adsorption_sites_df()
        if ad_sites_df is None:
            return None
        
        # Make sure the lates data is in the df:
        surface_study.update_adsorption_energies()
        # Get updated df:
        ad_sites_df = surface_study.get_adsorption_sites_df()

        site_rows = ad_sites_df[ad_sites_df["Site index"] == site_idx]
        if site_rows.empty:
            return None

        return site_rows.iloc[0]

    def _set_process_oer_metrics_from_site_row(
        self,
        process_id,
        site_metrics_row: pd.Series | None,
    ) -> None:
        """
        Write site-level OER metric values into one process row.

        Missing metrics (or missing site row) are stored as NaN.
        """
        
        # Ensure integer process id for indexing:
        if isinstance(process_id, np.integer):
            process_id = int(process_id)
        if not isinstance(process_id, int):
            raise ValueError("process_id must be an integer.")

        # Iterate over process cols:
        process_cols = set(self.rframe.processes.df.columns.tolist())
        for metric_col in self.PROCESS_METRIC_COLUMNS:
            # If col not in process df, skip:
            if metric_col not in process_cols:
                continue

            # Init value (to be used in the metric_col) as nan:
            value = np.nan
            if site_metrics_row is not None:
                raw = site_metrics_row.get(metric_col, np.nan)
                if pd.notna(raw):
                    value = float(raw)

            # Update process row with the metric value:
            self.rframe.processes.set_value(
                process_id,
                metric_col,
                value,
            )

    def _update_process_oer_metrics(
        self,
        process_id,
        candidate_id: str,
        termination_index,
        site_index,
    ) -> None:
        """
        Refresh one process row from the current site metrics.

        The method fetches the matching site metrics row and writes its
        OER metric values to the process table.
        """

        # Get the adsorption-site metrics row for the process:
        site_metrics_row = self._get_adsorption_site_metrics_row(
            candidate_id,
            termination_index,
            site_index,
        )
        
        # Set the process OER metrics:
        self._set_process_oer_metrics_from_site_row(
            process_id,
            site_metrics_row,
        )

    def _update_oh_batch_process_oer_metrics(
        self,
        candidate_id: str,
        termination_index: int | str,
        site_index: int | str,
    ) -> None:
        """
        Refresh process-level OER metrics for all OH jobs on one site.

        Each OH process row for the given candidate/termination/site is
        updated from the current site metrics.
        """
        
        # Return if identifiers are not provided:
        if pd.isna(termination_index) or pd.isna(site_index):
            return

        term_idx = int(termination_index)
        site_idx = int(site_index)
        pdf = self.rframe.processes.df
        oh_rows = pdf[
            (pdf["candidate_id"] == candidate_id)
            & (pdf["job_type"] == "OH_adsorption")
            & (pdf["termination_index"] == term_idx)
            & (pdf["site_index"] == site_idx)
        ]

        # Populate only the first completed row; leave the others unpopulated.
        completed_ids = oh_rows.loc[
            oh_rows["status"] == "completed", "process_id"
        ]
        if completed_ids.empty:
            return
        self._update_process_oer_metrics(
            completed_ids.iloc[0],
            candidate_id,
            termination_index,
            site_index,
        )

    def _refresh_oh_batch_finalization_state(
        self,
        candidate_id: str,
        termination_index,
        site_index,
    ) -> bool:
        ad_study = self._get_adsorption_site_study(
            candidate_id, termination_index, site_index
        )
        pdf = self.rframe.processes.df
        term_idx = int(termination_index)
        site_idx = int(site_index)
        oh_rows = pdf[
            (pdf["candidate_id"] == candidate_id)
            & (pdf["job_type"] == "OH_adsorption")
            & (pdf["termination_index"] == term_idx)
            & (pdf["site_index"] == site_idx)
        ]

        all_terminal = (
            not oh_rows.empty
            and oh_rows["status"].apply(self._is_terminal_process_status).all()
        )
        ad_study.set_oh_batch_finalized(all_terminal)
        return bool(all_terminal)

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
                    # Keep COMPLETING jobs in a running-like state
                    # until they disappear from the queue and can be
                    # post-processed.
                    elif new_status == 'completing':
                        new_status = 'running'

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
                    process_id = int(pdf_row['process_id'])
                    study = cand_row.df['study_obj'].values[0]
                    
                    job_type = pdf_row['job_type']
                    termination_index = pdf_row['termination_index']
                    site_index = pdf_row['site_index']

                    if job_type == 'bulk_relaxation':
                        mag_idx = extract_mag_config_index(
                            pdf_row['processNote']
                        )
                        if mag_idx is None:
                            raise RuntimeError(
                                f"Process {process_id}: bulk_relaxation "
                                "processNote is missing mag_config_index."
                            )
                        # In test/slurm_test mode there is no real VASP run,
                        # so the atoms carry no meaningful magnetic moments.
                        # Use the initial enumerator magmoms from the stored
                        # configs instead.
                        magmom = self._get_config_magmom(study, mag_idx)
                        study.store_mag_bulk_candidate(
                            mag_idx, atoms, E, magmom
                        )
                    else:
                        surface_studies = study.get_surface_studies()
                        surface_so = surface_studies[termination_index]
                        
                        if job_type == 'surface_relaxation':
                            # Test-mode atoms carry no VASP calculator;
                            # require_magmom=False avoids ValueError.
                            # Slab magmoms (bulk-derived, site_properties)
                            # are unaffected — they propagate via
                            # get_slab_magmom(), not _relaxed_surface_magmom.
                            surface_so.set_relaxed_base_surface(
                                atoms, E, require_magmom=False
                            )
                            # Test-mode: no VASP calculator → _relaxed_surface_magmom
                            # stays None. Inject slab magmoms as stand-in so
                            # get_O/OH_init_magmom work in test/slurm_test mode.
                            if surface_so._relaxed_surface_magmom is None:
                                _slab_m = surface_so.get_slab_magmom()
                                if _slab_m is not None:
                                    surface_so._relaxed_surface_magmom = _slab_m
                        else:
                            ad_site_dict = surface_so.\
                                get_adsorption_site_studies_dict()
                            ad_study = ad_site_dict[site_index]
                            if job_type == 'O_adsorption':
                                ad_study.set_relaxed_O_calc(atoms, E)
                                # Test-mode atoms carry no VASP calculator so
                                # _relaxed_O_magmom stays None after the call
                                # above. Inject the initial O magmom so that
                                # get_OH_init_magmom (on-top sites) is
                                # exercisable in test mode. Mirrors the
                                # bulk_relaxation fix (_get_config_magmom).
                                if ad_study._relaxed_O_magmom is None:
                                    o_init = ad_study.get_O_init_magmom()
                                    if o_init is not None:
                                        ad_study._relaxed_O_magmom = o_init
                            elif job_type == 'OH_adsorption':
                                ad_study.set_relaxed_OH_calc(
                                    atoms, energies=[E]
                                )

                    self.rframe.processes.df.at[idx, 'status'] = \
                        'completed'

                    updates[process_id] = 'completed'

                    if job_type == 'bulk_relaxation':
                        if self._all_magnetic_bulk_terminal(candt_id):
                            self._finalize_mag_bulk_batch(candt_id)
                    elif job_type == 'O_adsorption':
                        self._update_process_oer_metrics(
                            process_id, candt_id, termination_index, site_index
                        )
                        self._update_candidate_oer_metrics(candt_id)
                    elif job_type == 'OH_adsorption':
                        oh_batch_ready = \
                            self._refresh_oh_batch_finalization_state(
                                candt_id, termination_index, site_index
                            )
                        if oh_batch_ready:
                            self._update_oh_batch_process_oer_metrics(
                                candt_id, termination_index, site_index
                            )
                            self._update_candidate_oer_metrics(candt_id)
                            # Keep OH process rows unchanged until the
                            # full OH batch is terminal.

        else:
            r_pdf = self.rframe.processes.df.copy()
            r_pdf = r_pdf[r_pdf['status'] == 'updating']
            for idx, pdf_row in r_pdf.iterrows():
                vasp_dir = Path(pdf_row['VASP_dir'])
                job_type = pdf_row['job_type']
                termination_index = pdf_row['termination_index']
                site_index = pdf_row['site_index']
                candt_id = pdf_row['candidate_id']
                process_id = int(pdf_row['process_id'])

                if self.mode == 'MLIP_test':
                    try:
                        traj = Trajectory(vasp_dir / "mlip_result.traj")
                        if len(traj) == 0:
                            raise RuntimeError(
                                "MLIP result trajectory is empty."
                            )
                        atoms = traj[-1]
                        traj.close()
                        # Normalise the calculator so magmoms are always
                        # accessible downstream, while preserving the MLIP
                        # energy. Energy is mandatory — raises if missing.
                        # Forces are not needed. Magmoms fall back to zeros
                        # if the MLIP model provides no spin moments.
                        energy = atoms.get_potential_energy()
                        try:
                            magmom = atoms.get_magnetic_moments().tolist()
                        except Exception:
                            magmom = [0.0] * len(atoms)
                        atoms.calc = SinglePointCalculator(
                            atoms, energy=energy, magmoms=magmom
                        )
                        clean_up_vasp_directory(vasp_dir)

                    except Exception:
                        self.rframe.processes.df.at[idx, 'status'] \
                            = 'failed'
                        updates[process_id] = 'failed'
                        if job_type == 'bulk_relaxation':
                            if self._all_magnetic_bulk_terminal(candt_id):
                                self._finalize_mag_bulk_batch(candt_id)
                        elif job_type == 'O_adsorption':
                            self._update_process_oer_metrics(
                                process_id, candt_id, termination_index,
                                site_index
                            )
                        elif job_type == 'OH_adsorption':
                            oh_batch_ready = \
                                self._refresh_oh_batch_finalization_state(
                                    candt_id,
                                    termination_index,
                                    site_index,
                                )
                            if oh_batch_ready:
                                self._update_oh_batch_process_oer_metrics(
                                    candt_id, termination_index,
                                    site_index
                                )
                                self._update_candidate_oer_metrics(
                                    candt_id
                                )
                                # Keep OH process rows unchanged until
                                # all OH jobs for this site are
                                # terminal.
                        continue

                else:
                    try:
                        atoms = read_vasp_results(vasp_dir)[-1]
                        energy = atoms.get_potential_energy()
                        if np.isnan(energy):
                            raise ValueError(
                                "NaN energy from VASP result."
                            )
                        magmom = atoms.get_magnetic_moments().tolist()
                        if np.any(np.isnan(magmom)):
                            raise ValueError(
                                "NaN magnetic moments from VASP result."
                            )
                        clean_up_vasp_directory(vasp_dir)
                    except Exception:
                        import traceback
                        print(
                            '\n\n' + '-'*80 + '\n',
                            f"\n[update_log] process {process_id} "
                            f"({job_type}) FAILED while reading results "
                            f"from {vasp_dir}:\n"
                            + traceback.format_exc(),
                            '\n\n' + '-'*80 + '\n',
                            flush=True,
                        )
                        self.rframe.processes.df.at[idx, 'status'] \
                            = 'failed'
                        updates[process_id] = 'failed'
                        if job_type == 'bulk_relaxation':
                            if self._all_magnetic_bulk_terminal(candt_id):
                                self._finalize_mag_bulk_batch(candt_id)
                        elif job_type == 'O_adsorption':
                            self._update_process_oer_metrics(
                                process_id, candt_id, termination_index,
                                site_index
                            )
                        elif job_type == 'OH_adsorption':
                            oh_batch_ready = \
                                self._refresh_oh_batch_finalization_state(
                                    candt_id,
                                    termination_index,
                                    site_index,
                                )
                            if oh_batch_ready:
                                self._update_oh_batch_process_oer_metrics(
                                    candt_id, termination_index,
                                    site_index
                                )
                                self._update_candidate_oer_metrics(
                                    candt_id
                                )
                            else:
                                # NOTE:
                                # Keep OH process rows unchanged until
                                # all OH jobs for this site are
                                # terminal. Re-enable if per-process
                                # partial snapshots are needed.

                                # self._update_process_oer_metrics(
                                #     process_id,
                                #     candt_id,
                                #     termination_index,
                                #     site_index,
                                # )
                                pass
                        continue

                cand_row = self.rframe.candidates[candt_id]
                study = cand_row.df['study_obj'].values[0]
                unrecoverable_reason = None

                if job_type == 'bulk_relaxation':
                    mag_idx = extract_mag_config_index(
                        pdf_row['processNote']
                    )
                    if mag_idx is None:
                        raise RuntimeError(
                            f"Process {process_id}: bulk_relaxation "
                            "processNote is missing mag_config_index."
                        )
                    study.store_mag_bulk_candidate(
                        mag_idx, atoms, energy, magmom
                    )
                else:
                    surface_studies = study.get_surface_studies()
                    surface_study = surface_studies[termination_index]

                    if job_type == 'surface_relaxation':
                        # Defensive: base_surface_non_relaxed has tags=0
                        # today, but restore explicitly so the invariant
                        # holds if slab tagging is ever added.
                        atoms.set_tags(
                            surface_study.get_non_relaxed_surface().get_tags()
                        )
                        success = surface_study.\
                            set_relaxed_base_surface(atoms, energy=energy)
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
                            # Restore tags lost in VASP round-trip
                            # (POSCAR/vasprun.xml strip ASE tags).
                            atoms.set_tags(
                                ad_study.get_atoms_for_O_calc().get_tags()
                            )
                            result = ad_study.set_relaxed_O_calc(
                                atoms, energy=energy
                            )
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
                            # Restore tags lost in VASP round-trip.
                            atoms.set_tags(
                                ad_study.OH_init_atoms[0].get_tags()
                            )
                            result = ad_study.set_relaxed_OH_calc(
                                atoms, energies=[energy]
                            )
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

                if unrecoverable_reason is not None:
                    self.rframe.processes.df.at[idx, 'status'] = \
                        "unrecoverable: " + unrecoverable_reason
                    updates[process_id] = 'unrecoverable'
                    if job_type == 'O_adsorption':
                        self._update_process_oer_metrics(
                            process_id, candt_id, termination_index,
                            site_index
                        )
                    elif job_type == 'OH_adsorption':
                        oh_batch_ready = \
                            self._refresh_oh_batch_finalization_state(
                                candt_id, termination_index, site_index
                            )
                        if oh_batch_ready:
                            self._update_oh_batch_process_oer_metrics(
                                candt_id, termination_index, site_index
                            )
                            self._update_candidate_oer_metrics(candt_id)
                            # Keep OH process rows unchanged until the
                            # full OH batch is terminal.
                    continue

                self.rframe.processes.df.at[idx, 'status'] = \
                    'completed'
                updates[process_id] = 'completed'
                if job_type == 'bulk_relaxation':
                    if self._all_magnetic_bulk_terminal(candt_id):
                        self._finalize_mag_bulk_batch(candt_id)
                elif job_type == 'O_adsorption':
                    self._update_process_oer_metrics(
                        process_id, candt_id, termination_index,
                        site_index
                    )
                    self._update_candidate_oer_metrics(candt_id)
                elif job_type == 'OH_adsorption':
                    oh_batch_ready = self._refresh_oh_batch_finalization_state(
                        candt_id, termination_index, site_index
                    )
                    if oh_batch_ready:
                        self._update_oh_batch_process_oer_metrics(
                            candt_id, termination_index, site_index
                        )
                        self._update_candidate_oer_metrics(candt_id)
                        # Keep OH process rows unchanged until the full
                        # OH batch is terminal.

        # Refresh the per-candidate progress projection (started/finalized
        # counts + active/failed state) now that all process statuses for this
        # cycle are settled. Whole-frame derived pass over the processes table.
        self._recompute_candidate_progress()
        # Refresh the disposition coverage projection (needs_disposition_update)
        # from the now-settled process statuses + the dispositions on record.
        self._recompute_disposition_state()
        # Self-heal reduced_formula (schema only -- see method docstring).
        self._ensure_reduced_formula_column()
        return updates

    # ------------------------------------------------------------------
    # Magnetic bulk helpers
    # ------------------------------------------------------------------

    def _get_config_magmom(self, study, mag_idx: int) -> list[float]:
        """Return the initial enumerator magmoms for config index mag_idx.
        Reads from configs stored on the study by store_mag_bulk_configs —
        never re-enumerates. Raises if configs are missing or index is out
        of range."""
        configs = study.get_mag_bulk_configs()
        if configs is None:
            raise RuntimeError(
                "Magnetic bulk configs not found on study object. "
                "enter_process must be called before update_log."
            )
        if mag_idx >= len(configs):
            raise RuntimeError(
                f"mag_config_index={mag_idx} out of range "
                f"(only {len(configs)} configs stored)."
            )
        return configs[mag_idx]["magmom"]

    def _all_magnetic_bulk_terminal(self, candidate_id: str) -> bool:
        """True when every bulk_relaxation process for this candidate has
        reached a terminal status (completed, failed, or unrecoverable)."""
        pdf = self.rframe.processes.df
        bulk_rows = pdf[
            (pdf["candidate_id"] == candidate_id) &
            (pdf["job_type"] == "bulk_relaxation")
        ]
        if bulk_rows.empty:
            return False
        return bulk_rows["status"].apply(
            self._is_terminal_process_status
        ).all()

    def _finalize_mag_bulk_batch(self, candidate_id: str) -> bool:
        """Pick the lowest-energy completed bulk candidate and set it as
        the relaxed bulk on the study. Returns True if a winner was found."""
        study = self.rframe.candidates[candidate_id].df[
            "study_obj"
        ].values[0]
        return study.finalize_mag_bulk_batch()
