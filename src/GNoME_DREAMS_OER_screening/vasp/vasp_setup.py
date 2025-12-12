from pathlib import Path
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.core import Structure

def create_vasp_folder(structure: Structure, directory: str, user_incar_settings=None):
    """
    Create a VASP input directory with POSCAR, INCAR, KPOINTS, and POTCAR
    using pymatgen's MPRelaxSet.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The crystal structure for the calculation.
    directory : str
        Path to the directory to create VASP inputs in.
    user_incar_settings : dict, optional
        Custom INCAR settings to override defaults.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Create a default relaxation input set
    vasp_input_set = MPRelaxSet(
        structure,
        user_incar_settings=user_incar_settings or {}
    )

    # Write VASP input files to folder
    vasp_input_set.write_input(str(directory))

    print(f"VASP input folder created at: {directory.resolve()}")
