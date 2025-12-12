import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab

import numpy as np
import pandas as pd

def get_atoms_and_normal_from_input(input_surface: Atoms | Slab, 
                normal_vector: np.ndarray | None = None
                ) -> tuple[Atoms, np.ndarray]:
        
    if isinstance(input_surface, Slab):
        atoms = AseAtomsAdaptor.get_atoms(input_surface)
        if normal_vector is None:
            normal_vector = input_surface.normal
    else:
        if normal_vector is None:
            raise ValueError("If input is Atoms, normal_vector must " \
            "be provided.")
        atoms = input_surface

    return (atoms, normal_vector)