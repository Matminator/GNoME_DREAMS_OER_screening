from datetime import datetime
from pysqa import QueueAdapter
from pathlib import Path
from typing import Literal
from ase.io import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

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
    run_vasp_via_custodian, calculational_setup)
from gnome_dreams_oer_screening.explog.relational_frame import (
    RelationalFrame)
from gnome_dreams_oer_screening import get_my_jobs_status





class queue_factory():
    def __init__(self, mode: Literal["test", "slurm_test",
                        "production_test", "production"] ):
        self.mode = mode
        self.partition_configs = {
            "xeon24el8_test": {
                "nodes": 1,
                "cores": 1,
                "run_time_max": "00:10:00",
                "mail-type": "NONE"
            },
            "xeon24el8": {
                "nodes": 1,
                "cores": 24,
                "run_time_max": "50:00:00",
                "mail-type": "NONE"
            },
            "xeon40el8": {
                "nodes": 1,
                "cores": 40,
                "run_time_max": "50:00:00",
                "mail-type": "NONE"
            },
            "xeon56": {
                "nodes": 1,
                "cores": 56,
                "run_time_max": "50:00:00",
                "mail-type": "NONE"
            },
        }

    def create_queue(self, partition: Literal["xeon24el8", "xeon40el8", 
                                "xeon56", "auto"]) -> QueueAdapter:
        
        if self.mode in ["test", "slurm_test"]:
            partition = "xeon24el8_test"
        else:
            if partition == "auto":
                # Placeholder for auto resource selection logic: <<< --- CAN SPECIFY FURTHER LATER
                partition = "xeon40el8"

        if partition not in self.partition_configs:
            raise ValueError(f"Unknown partition: {partition}")
        
        config = self.partition_configs[partition]

        return QueueAdapter(
            scheduler_type="slurm",
            job_name="A_vasp_job",
            queue_name=partition,
            # stderr="./slabord.err",      # corresponds to -e
            # stdout="./slabord.out",      # corresponds to -o
            **config
            )
    

class Job_handler():

    def __init__(self, 
        rframe: RelationalFrame,
        vasp_calculation_dir: Path,
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
        prefix_vasp_dirs_with_prefix: bool = True
        ) -> None:

        # Create the directory if it doesn't exist
        if not vasp_calculation_dir.exists():
            vasp_calculation_dir.mkdir(parents=True)

        self.vasp_dir = vasp_calculation_dir
        self.rframe = rframe
        self.mode = mode
        self.queue_factory = queue_factory(mode=mode)
        self.calc_setup = calculational_setup(mode=mode)

        if prefix_vasp_dirs_with_prefix:
            now = datetime.now()
            # Format as string: YYYYMMDD_HHMMSS
            timestamp = now.strftime("%Y%m%d_%H%M%S") + "_"
        else:
            timestamp = ""
        self.vasp_dir_prefix = timestamp

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

    def submit_vasp_calculation(
        self,
        candidate_id: str,
        process_id: str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'],
        custodian_handlers: list = [VaspErrorHandler(),
        MeshSymmetryErrorHandler(), NonConvergingErrorHandler(),
        UnconvergedErrorHandler(), FrozenJobErrorHandler(),
        StdErrHandler(), PositiveEnergyErrorHandler()
        # LargeSigmaHandler(), IncorrectSmearingHandler(), 
        ],
        vasp_cmd: list = ["mpiexec", "vasp_std"],
        max_errors: int = 20
        ) -> None:

        if isinstance(process_id, int):
            process_id = str(process_id)
        if not isinstance(process_id, str):
            raise ValueError("process_id must be a string or an " \
                "integer.")

        # Get the study object for the candidate:
        study_obr = self.rframe.candidates[candidate_id].df["study_obj"]
        study = study_obr.values[0]
        
        # Get the process row for the candidate and process id:s
        process_row = self.rframe.processes.df.loc[
            (self.rframe.processes.df['candidate_id'] == candidate_id) &
            (self.rframe.processes.df['process_id'] == process_id)
        ]

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

        if self.mode != 'test':
            # Get partion settings from queue factory:
            if self.mode == 'slurm_test':
                queue_settings = self.queue_factory.create_queue(
                    partition='xeon24el8_test')
            else:
                queue_settings = self.queue_factory.create_queue(
                    partition=partition)

            # Define python function
            if self.mode == 'slurm_test':
                job_type = 'gnome_dreams_oer_screening.niflheim.' + \
                    'sleep_job.py'
            else:   
                raise NotImplementedError(f"Job submission not implemented for mode {self.mode} yet.")

            job_id_tmp = self.qa.submit(
                command = f"python3 {job_type} {calc_dir.resolve()}",
                additional_script_lines=[
                    "module purge",
                    "module use /home/modules/energy/modules/all",
                    "module load VASP/6.5.1-intel-2023a",
                    "module load impi/2021.9.0-intel-compilers-2023.1.0",
                    "ml Python/3.11.3-GCCcore-12.3.0",
                    "source venv/bin/activate",
                ]
                **queue_settings
            )
            
        self.rframe.candidates[candidate_id].processes[process_id].status = 'submitted'
        
    def update_log(self) -> None:

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
        if self.mode in ['test', 'slurm_test']:
            for idx, pdf_row in self.rframe.processes.df.iterrows():
                if pdf_row['status'] == 'updating':
                    
                    vasp_dir = Path(pdf_row['VASP_dir'])
                    traj = Trajectory(vasp_dir / 'fake_result.traj')
                    atoms = traj[-1]
                    E = atoms[0].charge # Not needed-at least currently

                    candt_id = pdf_row['candidate_id']
                    cnad_row = self.rframe.candidates[candt_id]
                    study = cnad_row.df['study_obj'].values[0]
                    
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
                       
                    self.rframe.processes.df.at[idx, 'status'] = \
                        'completed'
        

        
        




        return self.rframe.processes