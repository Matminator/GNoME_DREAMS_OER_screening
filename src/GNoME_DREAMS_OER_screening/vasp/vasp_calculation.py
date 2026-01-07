import os
from pathlib import Path

from ase import Atoms
from ase.io import read

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

def exlib_run_vasp(
    directory, 
    retrun_vasp_out = True,
    handlers = [VaspErrorHandler()],
    vasp_cmd = ["mpiexec", "vasp_std"],
    auto_npar = True,
    max_errors = 20
    ):
    """
    Function for running a VASP calculation via Custodian and returning
    results.

    Parameters:
    directory: directory where VASP calculation will be run.
    retrun_vasp_out (bool): Whether to return the content of vasp.out
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

    if retrun_vasp_out:
        with open("vasp.out", "r") as f:
            vasp_text = f.read()
        out["vasp_out"] = vasp_text
    else:
        out["vasp_out"] = None

    return out
