from pathlib import Path
from typing import Literal

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
from gnome_dreams_oer_screening.explog.relational_frame import RelationalFrame

from pysqa import QueueAdapter


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
                                "xeon56","auto"]) -> QueueAdapter:
        
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
        explog: RelationalFrame, 
        exlib_cache_dir: Path,
        vasp_calculation_dir: Path,
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
        overwrite_existing_vasp_dirs: bool = False
        ) -> None:

        if vasp_calculation_dir.exists():
            if not overwrite_existing_vasp_dirs:
                raise FileExistsError(f"The specified VASP" \
                    f" calculation root directory " \
                    f"{vasp_calculation_dir} already exists. " \
                    f"To overwrite, set " \
                    f"overwrite_existing_vasp_dirs=True.")
        else:
            vasp_calculation_dir.mkdir(parents=True, exist_ok=True)

        self.vasp_dir = vasp_calculation_dir
        self.exlib_cache_dir = exlib_cache_dir
        self.explog = explog
        self.mode = mode
        self.queue_factory = queue_factory(mode=mode)
        self.calc_setup = calculational_setup(mode=mode)

    def enter_in_log(
        self,
        candidate_id, # <------- DO WE KNOW WHAT TYPE THIS IS?
        calculation_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        note: str | None = None
        ) -> None:

        N_candidates = len(self.explog.candidates.df)

        new_process_row = {"job_type": calculation_type, 
                       "candidate_id": candidate_id,
                       "status": "un-submitted"
        }
        if note is not None:
            new_process_row["processNote"] = note

        # Get the candidate row from the ExpLog:
        cand_row = self.explog.candidates[candidate_id]
        cand_processes = cand_row.processes

        # Sanetity checks before entering the calculation in the log:
        p_df = cand_processes.df
        p_N = len(p_df)
        new_process_row['process_id'] = str(N_candidates) + str(p_N)
        if p_N > 0 and calculation_type == "bulk_relaxation":
            sub_df = p_df[(p_df['job_type'] == calculation_type) & 
                             (p_df['status'] == 'completed')]
            if len(sub_df) > 0:
                raise RuntimeError(f"Bulk relaxation calculation has " \
                f"already been completed for candidate {candidate_id}.")
        
        new_process_row['VASP_dir'] = str(self.vasp_dir / 
            f"{candidate_id}_{calculation_type}_"\
                f"{new_process_row['process_id']}")
                
        self.explog.processes.add_row(new_process_row, 
                                      allow_update=True)

    def submit_vasp_calculation(
        self,
        candidate_id,
        process_id,
        resource: Literal['xeon56', 'xeon40el8', 'xeon24el8', # <<< ---- CAN SPECIFY FURTHER LATER
                           'auto'],
        custodian_handlers: list = [VaspErrorHandler(),
        MeshSymmetryErrorHandler(), NonConvergingErrorHandler(),
        UnconvergedErrorHandler(), FrozenJobErrorHandler(),
        StdErrHandler(), PositiveEnergyErrorHandler()
        # LargeSigmaHandler(), IncorrectSmearingHandler(), 
        ],
        vasp_cmd: list = ["mpiexec", "vasp_std"],
        max_errors: int = 20
        ) -> None:

        study = self.explog.candidates[candidate_id].df["study_obj"].values[0]
        
        if self.mode != 'test':
            raise NotImplementedError("Only test mode is implemented " \
            "currently.")
        
        process_row = self.explog.processes.df.loc[
            (self.explog.processes.df['candidate_id'] == candidate_id) &
            (self.explog.processes.df['process_id'] == process_id)
        ]
        if len(process_row) == 0:
            raise ValueError(f"No process found for candidate_id "
                        f"{candidate_id} and process_id {process_id}.")

        if process_row['status'].values[0] != 'un-submitted':
            raise RuntimeError(f"Process {process_id}, is not an " \
                f"un-submitted process and cannot be submitted.")
        
        calc_dir = Path(process_row['VASP_dir'].values[0])

        if process_row['job_type'].values[0] == 'bulk_relaxation':
            init_structure = study.get_init_structure()
            self.calc_setup.bulk_relaxation(
                structure = init_structure, directory = calc_dir
            )

        elif process_row['job_type'].values[0] in \
            ["surface_relaxation", "O_adsorption", 'OH_adsorption']:

            raise NotImplementedError(f"Calculation type "\
                f"{process_row['job_type'].values[0]} not " \
                    f"implemented yet.")
        
            self.calc_setup.surface_relaxation(
                structure = init_structure, directory = calc_dir
            )
        

        if self.mode == 'slurm_test':
            slurm_template = 'TEMP TEMP'

        elif self.mode in ['production_test', 'production']:
            if resource == 'auto':
                # Placeholder for auto resource selection logic:
                # slurm_template = read_template('xeon40el8')
                slurm_template = {
                    "job_name": "vasp_job",
                    "cores_max": 24,
                    "nodes_max": 1,
                    "memory_max": 2000,
                    "partition": "venkvis-cpu",
                }
            else:
                # slurm_template = read_template(resource)
                slurm_template = {
                    "job_name": "vasp_job",
                    "cores_max": 48,
                    "nodes_max": 1,
                    "memory_max": 2000,
                    "partition": "venkvis-asdfasdfpu",
                }

        if self.mode != 'test':

            if self.mode == 'slurm_test':
                job_type = 'TEMP TEMP TEMP'
            else:   
                job_type = 'TEMP TEMP TEMP'

            job_id_tmp = self.qa.submit(
                node_max = slurm_template["nodes_max"],
                core_max = slurm_template["cores_max"],
                memory_max = slurm_template["memory_max"],
                command = f"python {job_type} {calc_dir.resolve()}",
            )
            self.explog 

        self.explog.candidates[candidate_id].processes[process_id].status = 'submitted'
        
    def update_log(self) -> None:
        return self.explog.processes
    
    