from pathlib import Path
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import VaspInputSet, DictSet



def vasp_calculation_setup(
    structure: Structure | Atoms,
    directory: str,
    incar_settings: dict,
    kpoints: dict,
    potcar_symbols: list,
    potcar_functional: str,
    potcar_spec: bool = False,
):
    """
    Create a VASP calculation folder with fully custom parameters using
    pymatgen.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Material structure.
    directory : str
        Folder to write input files to.
    incar_settings : dict
        Dictionary of INCAR parameters.
    kpoints_settings : dict
        KPOINTS dict (style, grid, subdivisions, etc).
    potcar_symbols : list of str
        POTCAR symbols, e.g. ["Si", "O"].
    potcar_functional : str
        Which version of POTCAR pseudopotentials to use, e.g. "PBE".
    """

    if isinstance(structure, Atoms):
        structure = AseAtomsAdaptor.get_structure(structure)
    if not isinstance(structure, Structure):
        raise TypeError("structure must be a pymatgen Structure "
        "or ASE Atoms object.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Create a general VaspInputSet using DictSet
    vasp_set = DictSet(
        incar = incar_settings,
        kpoints = kpoints,
        potcar_symbols = potcar_symbols,
        UserPotcarFunctional = potcar_functional,
        structure=structure,
    )

    # Write the VASP inputs
    vasp_set.write_input(str(directory), potcar_spec = potcar_spec)

    print(f"Custom VASP input folder created at: {directory.resolve()}")
