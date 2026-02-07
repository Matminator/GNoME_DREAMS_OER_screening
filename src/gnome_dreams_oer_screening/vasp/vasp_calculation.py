
import os
from pathlib import Path
from time import time
import numpy as np
from typing import Literal

from ase import Atoms
from ase.io import read, Trajectory
from pymatgen.core.surface import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from executorlib import SlurmClusterExecutor

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
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


from gnome_dreams_oer_screening.vasp.pre_defined_vasp_sets import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set, TestRelaxSet)

from gnome_dreams_oer_screening import RelationalFrame, read_template

def run_vasp_via_custodian(
                directory, 
                handlers = [VaspErrorHandler()],
                vasp_cmd = ["mpiexec", "vasp_std"],
                auto_npar = True,
                max_errors = 20
                ):
    """
    Run a VASP calculation in a specified directory using Custodian.

    Parameters:
    directory (str): The directory where the VASP calculation will be 
    run.
    handlers (list): A list of Custodian error handlers to use.
    vasp_cmd (list): The command to run VASP.
    auto_npar (bool): Whether to automatically adjust NPAR.
    max_errors (int): Maximum number of errors to tolerate by Custodian
    before terminating the calculation.
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not" \
                                 " exist.")

    # save current working directory
    cwd = os.getcwd() 

    # change to the VASP directory
    os.chdir(directory)

    # instantiate VASP job
    job = VaspJob(vasp_cmd=vasp_cmd, auto_npar = auto_npar)

    # run with custodian
    c = Custodian(handlers=handlers, jobs=[job], 
                  max_errors = max_errors)
    c.run()

    # change back to the original directory
    os.chdir(cwd)

def read_vasp_results(directory, return_only_final=True) -> list[Atoms]:
    """
    Read VASP calculation results from a specified directory.

    Parameters:
    directory (str): The directory where the VASP calculation was run.
    return_only_final (bool): Whether to return only the final structure
    or all intermediate structures.

    Returns:
    list[Atoms]: A list of ASE Atoms objects representing the
    structures from the VASP calculation.
    """

    directory = Path(directory)  # Ensure directory is a Path object

    atoms_list = read(directory / "vasprun.xml", index=":")

    if return_only_final:
        out = [atoms_list[-1]]
    else:
        out = atoms_list
    return out

def clean_up_vasp_directory(
    directory: str, files_to_remove: list[str] = None,
    purge_all_files: bool = False
    ) -> None:
    
    dir_path =  Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if purge_all_files:
        targets = list(dir_path.iterdir())
    else:
        if files_to_remove is None:
            files_to_remove = ["POTCAR", "WAVECAR", "CHGCAR"]
        # Collect all files that match any of the specified substrings
        targets = [
            file for file in dir_path.iterdir()
            if any(token in file.name for token in files_to_remove)
        ]

    for file in targets:
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Could not remove {file}: {e}")

def exlib_run_slurm_test(**kwargs):
    time.sleep(8*60)

def exlib_run_vasp(
    directory, 
    return_vasp_out = True,
    handlers = [VaspErrorHandler()],
    vasp_cmd = ["mpiexec", "vasp_std"],
    auto_npar = True,
    max_errors = 20
    ):
    """
    Function for running a VASP calculation via Custodian and returning
    results.

    Parameters:
    directory (PathLike): directory where VASP calculation will be run.
    return_vasp_out (bool): Whether to return the content of vasp.out
    file.
    handlers (list): A list of Custodian error handlers to use.
    vasp_cmd (list): The command to run VASP.
    auto_npar (bool): Whether Custodian should automatically adjust 
    NPAR.
    max_errors (int): Maximum number of errors to tolerate by Custodian
    before terminating the calculation.
    """
    
    try:
        run_vasp_via_custodian(
            directory=directory,
            handlers=handlers,
            vasp_cmd=vasp_cmd,
            auto_npar=auto_npar,
            max_errors=max_errors
        )

        atoms_result = read_vasp_results(
            directory=directory,
            return_only_final=True
        )[0]
        E = atoms_result.get_total_energy()
        out = {"atoms_result": atoms_result, "E": E, "error": None}

        clean_up_vasp_directory(directory=directory)

    except Exception as e:
        out = {"atoms_result": None, "E": None, "error": str(e)}

    if return_vasp_out:
        with open(directory / "vasp.out", "r") as f:
            vasp_text = f.read()
        out["vasp_out"] = vasp_text
    else:
        out["vasp_out"] = None

    return out

def save_fake_results_to_traj(directory: Path, atoms: Atoms):
    out_traj = Trajectory(directory / "fake_result.traj", "w")
    out_traj.write(atoms)
    out_traj.close()
    

class calculational_setup():

    def __init__(self, 
                 mode: Literal['test', 'production_test', 'production'] 
                 = 'production',
                 bulk_set = RPBE_relax_bulk_set,
                 surface_set = RPBE_relax_surface_set
                 ) -> None:
        self.mode = mode

        # Setup of test modes:
        if mode == 'production_test':
            self.bulk_set = TestRelaxSet
            self.surface_set = TestRelaxSet
        else:
            self.bulk_set = bulk_set
            self.surface_set = surface_set

    def bulk_relaxation(self, structure: Structure, directory: Path
                        ) -> None:
        vasp_set = self.bulk_set(structure = structure)
        vasp_set.write_input(directory, potcar_spec = False if 
                             self.mode =='production' else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            fake_energy = len(atoms)
            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)
        
    def surface_relaxation(self, structure: Structure, directory: Path
                           ) -> None:
        vasp_set = self.surface_set(structure = structure)
        vasp_set.write_input(directory, potcar_spec = False if 
                             self.mode =='production' else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            
            # Fake energy calculation:
            fake_energy = len(atoms) * 2.46 + 2*np.random.rand() - 1
            if 'H' in atoms.symbols():
                fake_energy += -3.69            
            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)


class Job_submitter():

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
        self.calc_setup = calculational_setup(mode=mode)

    def enter_in_log(
        self,
        candidate_id, # <------- DO WE KNOW WHAT TYPE THIS IS?
        calculation_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        note: str | None = None
        ) -> None:

        new_process_row = {"job_type": calculation_type, 
                       "candidate_id": candidate_id,
                       "status": "un-submitted"
        }
        if note is not None:
            new_process_row["processNote"] = note

        # Get the candidate row from the ExpLog:
        cand_row = self.explog.candidates[candidate_id]

        processes = cand_row.processes

        # Sanetity checks before entering the calculation in the log:
        p_df = processes.df
        p_N = len(p_df)
        new_process_row['process_id'] = p_N
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
        print(study)
        
        process_row = self.explog.processes.df.loc[
            (self.explog.processes.df['candidate_id'] == candidate_id) &
            (self.explog.processes.df['process_id'] == process_id)
        ]

        print(process_row['job_type'].values[0])

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
            slurm_template = read_template('test')

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
                job_type = exlib_run_slurm_test
            else:   
                job_type = exlib_run_vasp

            # with SlurmClusterExecutor(
            #     cache_directory=self.exlib_cache_dir) as exe:
            #     job = exe.submit( # <------------------------------------- MAY BE EASIER TO USE F LATER????
            #             job_type,
            #             calc_dir.resolve(),
            #             resource_dict={
            #                 "submission_template": slurm_template,
            #             },
            #         )
                
            #     job_id = job.job_id
            #     self.explog.candidates[candidate_id].processes[process_id].slurmID = job_id
                
            job_id_tmp = self.qa.submit(
                node_max = slurm_template["nodes_max"],
                core_max = slurm_template["cores_max"],
                memory_max = slurm_template["memory_max"],
                command = f"python {job_type} {calc_dir.resolve()}",
            )
            self.explog 
            

        self.explog.candidates[candidate_id].processes[process_id].status = 'submitted'
        
    def update_log(self) -> None:
        self.explog.processes

    

"""
NOTE FOR FUTURE:

Initial structure
       |
       v
Static SCF (ISMEAR=0, SIGMA=0.05)
       |
       v
Handlers check:
   - LargeSigmaHandler
   - IncorrectSmearingHandler
       |
       v
Determine electronic type:
   - Metal -> ISMEAR=1/2, SIGMA=0.2
   - Insulator -> ISMEAR=-5
       |
       v
Relaxation calculation (NSW>0)

"""




"""
TEMP....


# --- Setup of VASP directories ---
vasp_calc_main_dir = Path("VASP_calculations/test_run_" + run_id)
calc_paths = [vasp_calc_main_dir / f"calc_{i}" for i in range(5)]

for path in calc_paths:
    vasp_set = RPBE_relax_bulk_set(test_structure)
    vasp_set.write_input(output_dir = path) 

# --- Cache location (for executorlib) ---


slurm_template = Path("slurm_templates/test_template.sh").read_text()

# --- Submit job ---
print("--- Submitting job to Slurm cluster... ---", flush=True)
with SlurmClusterExecutor(cache_directory=cache_dir) as exe:
    futures = []

    for i, calc_path in enumerate(calc_paths):
        f = exe.submit(
            exlib_run_vasp,
            calc_path.resolve(),
            resource_dict={
                "submission_template": slurm_template,
            },
        )
        futures.append(f)
        print(f"Submitted job {i}, future: {f}", flush=True)
    print("--- All jobs submitted ---", flush=True)

    results = [f.result() for f in futures]
    print("Results:", results)




for result in results:
    print("VASP run result:", result['E'])




"""