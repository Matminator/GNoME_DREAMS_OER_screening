
import os
from pathlib import Path
from time import time
import numpy as np
from typing import Literal

from ase import Atoms
from ase.io import read, Trajectory
from pymatgen.core.surface import Structure
from pymatgen.io.ase import AseAtomsAdaptor

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
            pass


def save_fake_results_to_traj(directory: Path, atoms: Atoms):
    out_traj = Trajectory(directory / "fake_result.traj", "w")
    out_traj.write(atoms)
    out_traj.close()
    

class calculational_setup():

    def __init__(self, 
                 mode: Literal['dummy_test', 'test', 'production_test', 
                    'production'] = 'production',
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
        vasp_set = self.bulk_set(structure = structure, 
                                 sort_structure = False)
        vasp_set.write_input(directory, potcar_spec = False if 
                             'production' in self.mode else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            fake_energy = len(atoms)
            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)
        
    def surface_relaxation(self, structure: Structure, directory: Path
                           ) -> None:
        vasp_set = self.surface_set(structure = structure, 
                                    sort_structure = False)
        vasp_set.write_input(directory, potcar_spec = False if 
                             'production' in self.mode else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)

            K_O = (-14.217) - 0.070 - (-6.77)
            K_OH = 0.361 + (-6.77)/2 - (-14.217)
            x = 2.46 + K_O
            y = 1.23 - 2*x + K_OH
            
            if 'dummy' in self.mode:
                random_factor = 0
            else:
                random_factor = np.random.normal(loc = 0, scale = 1)

            # Fake energy calculation:
            if 'H' in atoms.symbols:
                fake_energy = x * len(atoms) - y + random_factor
            else:
                fake_energy = x * len(atoms) + random_factor

            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)


    



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
