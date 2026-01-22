import numpy as np
import math
from dataclasses import dataclass
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Kpoints as VaspKpoints

from gnome_dreams_oer_screening.vasp.dummy_sets import get_dummy_settings # material_agent.GNoME_DREAMS_OER_screening.src. ...
from gnome_dreams_oer_screening.vasp.rpbe_relax_sets import (
    get_RPBE_bulk_settings, get_RPBE_surface_settings
    )   

def heuristic_compute_kpoints(structure, structure_type, 
                              kpts_nominator = -1):
    atoms = AseAtomsAdaptor.get_atoms(structure)

    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0,:])
    b = np.linalg.norm(cell[1,:])
    c = np.linalg.norm(cell[2,:])

    if kpts_nominator > 0:
        k_points = (
            math.ceil(kpts_nominator/a),
            math.ceil(kpts_nominator/b),
            math.ceil(kpts_nominator/c)
        )
    elif structure_type == 'bulk':
        k_points = (
            math.ceil(50/a), 
            math.ceil(50/b), 
            math.ceil(50/c))
    elif structure_type == 'surface':
        k_points = (
            math.ceil(30/a),
            math.ceil(30/b),
            1
        )

    # build a gamma-centered grid in VASP style
    return VaspKpoints.gamma_automatic(k_points)


@dataclass
class RPBE_relax_bulk_set(VaspInputSet):
    """
    Custom VASP input set for RPBE bulk relaxation using Python. This
    set is build specifically for this project.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """
    CONFIG = get_RPBE_bulk_settings()
    def __init__(self, structure, **kwargs):
        self.CONFIG["KPOINTS"] = heuristic_compute_kpoints(
            structure, structure_type='bulk'
            )
        super().__init__(structure=structure, **kwargs)

@dataclass
class RPBE_relax_surface_set(VaspInputSet):
    """
    Custom VASP input set for RPBE bulk relaxation using Python. This
    set is build specifically for this project.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """
    CONFIG = get_RPBE_surface_settings()
    def __init__(self, structure, **kwargs):
        self.CONFIG["KPOINTS"] = heuristic_compute_kpoints(
            structure, structure_type='surface'
            )
        super().__init__(structure=structure, **kwargs)


@dataclass
class TestRelaxSet(VaspInputSet):
    """
    Custom VASP input set using Python dictionary instead of YAML.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """
    CONFIG = get_dummy_settings()
    def __init__(self, structure, **kwargs):
        self.CONFIG["KPOINTS"] = heuristic_compute_kpoints(
            structure, structure_type='N/A', kpts_nominator=0.01
            )
        super().__init__(structure=structure, **kwargs)