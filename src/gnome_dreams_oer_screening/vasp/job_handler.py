from datetime import datetime
from pysqa import QueueAdapter
from pathlib import Path
from typing import Literal
from ase.io import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
import pandas as pd

from custodian.vasp.handlers import ( 
    VaspErrorHandler,
    MeshSymmetryErrorHandler,
    UnconvergedErrorHandler,
    NonConvergingErrorHandler,
    PotimErrorHandler,
    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler,
    StdErrHandler,
    LargeSigmaHandler,
    IncorrectSmearingHandler,
    KspacingMetalHandler,
    PotimErrorHandler, # only relaxation handler
    DriftErrorHandler, # only relaxation handler
    WalltimeHandler
)

from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    calculational_setup)
from gnome_dreams_oer_screening.explog.relational_frame import (
    RelationalFrame)
from gnome_dreams_oer_screening import get_my_jobs_status
    

class Job_handler():

    def __init__(self, 
        rframe: RelationalFrame,
        vasp_calculation_dir: Path,
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
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
        ) -> None:

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

        # Format the new process row:
        new_process_row = {"job_type": job_type, 
                          "candidate_id": candidate_id,
                          "status": "un-submitted",
                          "termination_index": termination_index,
                          "site_index": site_index,
                          "processNote": note
                          }
        
        p_df = cand_processes.df
        p_N = len(p_df)
        new_process_row['process_id'] = str(p_N)
        new_process_row['VASP_dir'] = str(self.vasp_dir / 
            f"{self.vasp_dir_prefix}{candidate_id}_" \
            f"{job_type}_{new_process_row['process_id']}")
                
        self.rframe.processes.add_row(new_process_row, 
                                      allow_update=True)
        
        return str(p_N)

    def submit_vasp_calculation(
        self,
        process_id: str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'],
        vasp_cmd: list = ["mpiexec", "vasp_std"],
        max_errors: int = 20
        ) -> None:

        if isinstance(process_id, int):
            process_id = str(process_id)
        if not isinstance(process_id, str):
            raise ValueError("process_id must be a string or an " \
                "integer.")

        # Get the process row for the candidate and process id:s
        process_row = self.rframe.processes.df.loc[
            (self.rframe.processes.df['process_id'] == process_id)
        ]

        # Get the candidate id from the process row:
        candidate_id = process_row['candidate_id'].values[0]

        # Get the study object for the candidate:
        study_obr = self.rframe.candidates[candidate_id].df["study_obj"]
        study = study_obr.values[0]        

        # Test if process row exists and is un-submitted:
        if len(process_row) == 0:
            raise ValueError(f"No process found for candidate_id "
                        f"{candidate_id} and process_id {process_id}.")
        if process_row['status'].values[0] != 'un-submitted':
            raise RuntimeError(f"Process {process_id}, is not an " \
                f"un-submitted process and cannot be submitted.")
        
        # Get the calculation directory from the process row:
        calc_dir = Path(process_row['VASP_dir'].values[0])
    
        # Get the job type from the process row:    
        job_type = process_row['job_type'].values[0]

        if job_type == 'bulk_relaxation':
            init_structure = study.get_init_structure()
            self.calc_setup.bulk_relaxation(
                structure = init_structure, directory = calc_dir)

        elif job_type in ["surface_relaxation", "O_adsorption", 
                          'OH_adsorption']:

            termination_index = \
                process_row['termination_index'].values[0]
            site_index = process_row['site_index'].values[0]

            # Test if surface studies have been initialized:
            surface_studies = study.get_surface_studies()
            if surface_studies is None:
                raise RuntimeError(f"Surface studies have not been " \
                    f"initialized for candidate  {candidate_id}. " \
                    f"Cannot submit surface relaxation job.")

            if termination_index not in surface_studies.keys():
                raise ValueError(f"Termination index has not been " \
                    f"initialized for candidate {candidate_id}. " \
                    f"Cannot submit surface relaxation job.")
            
            surface_so = surface_studies[termination_index]

            if job_type == "surface_relaxation":
                init_atoms = surface_so.get_non_relaxed_surface()
                init_structure \
                    = AseAtomsAdaptor.get_structure(init_atoms)

            elif job_type in ["O_adsorption", 'OH_adsorption']:
                adsite_studies = surface_so.\
                    get_adsorption_site_studies_dict()
                ad_study = adsite_studies[site_index]
                
                if job_type == "O_adsorption":
                    init_atoms = ad_study.get_initial_surface_with_O()
                    init_structure \
                        = AseAtomsAdaptor.get_structure(init_atoms)
                elif job_type == 'OH_adsorption':
                    raise NotImplementedError(f"Submission for job type"
                        f" {job_type} not implemented yet.")

            self.calc_setup.surface_relaxation(
                structure = init_structure, directory = calc_dir
            )

        slurm_id = 42 + int(process_id) # Dummy initialization for test mode
        if self.mode != 'test':
            
            if self.mode == 'slurm_test':
                python_sp = 'src/gnome_dreams_oer_screening/niflheim/'+ \
                    f'sleep_job.py {self.sleep_min}'
                partition = 'xeon24el8_test'
                cores, nodes = 1, 1
                walltime = f"00:0{self.sleep_min}:00"

            else:

                python_sp = 'src/gnome_dreams_oer_screening/vasp/'+ \
                    f'run_vasp.py' + f" {calc_dir.resolve()}"

                if partition == 'auto':
                    partition = 'xeon40el8' # <<< ---------------------- CAN MAKE THIS MORE INTELLIGENT LATER
                
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


                # curently overwriting for testing purposes:
                partition = 'xeon24el8_test'
                cores, nodes = 1, 1
                walltime = f"00:01:00"
            

            slurm_id = self.queueadapter.submit_job(
                command = f"python3 {python_sp} {calc_dir.resolve()}",
                slurm_out_name = calc_dir / "slurm_output",
                partition = partition,
                cores= cores,
                nodes= nodes,
                walltime=walltime,
            )

        self.rframe.candidates[candidate_id].processes[process_id]\
            .status = 'submitted'
        self.rframe.candidates[candidate_id].processes[process_id]\
            .slurmID = slurm_id

    def _update_ideal_over_pot(self, candidate_id) -> None:
        # <<< ---------------------------------------------------------- NOT A PROPOR IMPLIMENTATION...
        cand_row = self.rframe.candidates[candidate_id]
        study = cand_row.df['study_obj'].values[0]
        sur_studies = study.get_surface_studies()

        over_pots = []
        for sur_study in sur_studies.values():
            sur_study.update_adsorption_energies()
            ad_sites_df = sur_study.get_adsorption_sites_df()
            ad_sites_O_Gs = ad_sites_df['G_approx(O)'].values
            ad_sites_O_Gs = np.array(ad_sites_O_Gs, dtype=np.float64)
            ad_sites_O_Gs = ad_sites_O_Gs[~np.isnan(ad_sites_O_Gs)]
            over_pots.append(np.round(
                                   (np.abs(ad_sites_O_Gs - 2.46)), 2))
        
        self.rframe.candidates[candidate_id].idealOverPotential \
            = min(over_pots)

        
    def update_log(self) -> dict:

        # --- Identify calculations to update --------------------------
        if self.mode == 'test':
            for idx, pdf_row in self.rframe.processes.df.iterrows():
                if pdf_row['status'] == 'submitted':
                    self.rframe.processes.df.at[idx, 'status'] = \
                        'updating'

        else:
            slurm_queue = get_my_jobs_status()

            # IMPLIMENT UPDATING OF LOG BASED ON SLURM QUEUE STATUS:    <<< --- CAN SPECIFY FURTHER LATER

        # --- Read results and mark as compleat or failed --------------
        updates = {}

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
                                ad_study.set_relaxed_surface_with_O(
                                    atoms, E)
                            elif job_type == 'OH_adsorption':
                                raise NotImplementedError(f"Result processing "
                                    f"for job type {job_type} not implemented ")
                       
                            self._update_ideal_over_pot(candt_id)

                    self.rframe.processes.df.at[idx, 'status'] = \
                        'completed'
                    
                    updates[pdf_row['process_id']] = 'completed'

        return updates