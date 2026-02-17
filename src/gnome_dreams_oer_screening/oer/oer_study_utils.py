from typing import Literal
import numpy as np
import warnings
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab, Structure


def get_atoms_and_normal_from_input(
        input_surface: Atoms | Structure | Slab, 
        normal_vector: np.ndarray | None = None
        ) -> tuple[Atoms, np.ndarray]:
        
    if isinstance(input_surface, (Structure, Slab)):
        atoms = AseAtomsAdaptor.get_atoms(input_surface)
        if normal_vector is None:
            normal_vector = input_surface.normal
    else:
        if normal_vector is None:
            raise ValueError("If input is Atoms, normal_vector must " \
            "be provided.")
        atoms = input_surface

    return (atoms, normal_vector)

def get_atom_projections_on_normal(input: Atoms | Slab,
                            normal_vector: np.ndarray | None = None
                            ) -> float:
    
    atoms, normal_vector = get_atoms_and_normal_from_input(input,
                                                        normal_vector)
    
    pos = atoms.get_positions()
    return np.dot(pos, normal_vector)

def compute_slab_thickness(input: Atoms | Slab, 
                            normal_vector: np.ndarray | None = None
                            ) -> float:
    projections = get_atom_projections_on_normal(
        input, normal_vector)

    slab_thickness = projections.max() - projections.min()
    return slab_thickness

def get_top_half_atom_projections(input: Atoms | Slab,
        normal_vector: np.ndarray | None = None,
        ) -> np.ndarray:
    
    projections = get_atom_projections_on_normal(
        input, normal_vector)
    
    median_projection = np.median(projections)
    indices = np.where(projections >= \
                        median_projection)[0].astype(int)
    
    return indices

# --- Coordination determination ---------------------------------------
def determine_coordination_numbers(atoms: Atoms,
            method: Literal['all', 'disregard-coord-of-O', 
            'O-coordination']) -> list[int]:
    """
    Determine coordination numbers for all atoms in the given ASE
    Atoms object, using the specified method. The method "all" gives
    gives the corrdination numbers for all atoms given any neighbor,
    "disregard-coord-of-O" gives the coordination numbers for all 
    atoms excluding oxygen (sets oxygen coordination numbers to 
    zero). "O-coordination" gives the coordination numbers for all
    atoms considering only oxygen neighbors (also sets oxygen 
    coordination numbers to zero).
    Args:
        atoms (Atoms): ASE Atoms object representing the structure.
        method (str): Method to compute coordination numbers.
                        Currently only 'coordination' is implemented.
    Returns:
        list[int]: List of coordination numbers for each atom.
    """
    
    # Build neighbor list:
    atoms_copy = atoms.copy()
    atoms_copy.set_pbc(True)
    # The only neighborlist that will be needed:
    atoms_copy.set_pbc(True)
    cutoffs = natural_cutoffs(atoms_copy)
    nl = NeighborList(cutoffs, self_interaction=False, 
                        bothways=True, skin = 1e-4)
    nl.update(atoms_copy)

    coord_nums = []
    for index, atom in enumerate(atoms_copy):
        if atom.symbol == 'O' and method in ['disregard-coord-of-O', 
                                                'O-coordination']:
            coord_nums.append(0) # O atoms are ignored
        elif method == 'O-coordination':

            indices, offsets = nl.get_neighbors(index)
            indices = [i for i in indices if atoms_copy[i].symbol == 'O']

            coord_nums.append(len(indices)) # len(indices) = CN
        else:
            indices, offsets = nl.get_neighbors(index)
            coord_nums.append(len(indices)) # len(indices) = CN
            
    return coord_nums
# --- End Coordination determination -----------------------------------

# --- Stoichiometry utils ----------------------------------------------
def get_stoichiometry_dict(input: Atoms | Structure) -> dict:
    if isinstance(input, Structure):
        atoms = AseAtomsAdaptor.get_atoms(input)
    elif isinstance(input, Atoms):
        atoms = input
    else:
        raise ValueError("Input must be either a pymatgen Structure " \
        "or an ASE Atoms object.")
        
    elements, counts = np.unique(atoms.get_chemical_symbols(),
                                 return_counts=True)

    N_total = np.sum(counts)
    fractions = counts / N_total
    return dict(zip(elements, fractions))

def get_stoichiometry_diff(dict1: dict, dict2: dict) -> dict:
    """
    Calculate the stoichiometry difference between two stoichiometry 
    dictionaries. 
    NOTE that negative values indicate that an element is missing in one
    of the dictionaries.

    Args:
        dict1 (dict): First stoichiometry dictionary.
        dict2 (dict): Second stoichiometry dictionary.

    Returns:
        dict: Dictionary with absolute differences in stoichiometry.
        Negative values indicate missing elements in one of the
        dictionaries.
    """

    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    diff_dict = {}
    for key in all_keys:
        val1 = dict1.get(key, 0)
        val2 = dict2.get(key, 0)
        diff = np.abs(val1 - val2)
        if 0 in (val1, val2):
            diff_dict[key] = -diff
        else:
            diff_dict[key] = diff
    return diff_dict

def eval_stoichiometry_diff(
        config_1: Atoms | Structure, config_2: Atoms | Structure,
        max_diff: float = .1,
        max_diff_dict: dict = {},
        all_species_present: bool = True,
        ignore_species: list[str] = [],
        ) -> bool:
    """
    Evaluate whether the stoichiometry difference between two fullfills
    given criteria. 

    Args:
        config_1 (Atoms | Structure): First configuration.
        config_2 (Atoms | Structure): Second configuration.
        max_diff (float, optional): Maximum allowed stoichiometry 
            difference for species not in 
            stoichiometric_match_species. Defaults to .1.
        all_species_present (bool, optional): Whether all species
            in the stoichiometry difference dict must be present
            (non-negative) in both configurations. Defaults to True.
        max_diff_dict (dict | None, optional): Maximum allowed 
            stoichiometry difference for specific species. Defaults to 
            {}.
        ignore_species (list[str], optional): List of species to ignore
            when evaluating stoichiometry difference. Defaults to [].

    Returns:
        bool: True if the stoichiometry difference meets the criteria,
        False otherwise.
    """
    
    # Check that no species is both in max_diff_dict and
    # ignore_species:
    for species in max_diff_dict.keys():
        if species in ignore_species:
            warnings.warn("Putting a species in both max_diff_dict" \
            " and ignore_species may lead to unintended " \
            "behavior.", UserWarning, stacklevel=2)
    
    # get_stoichiometry_dict: agnostic to atoms or structure:
    dict1 = get_stoichiometry_dict(config_1) 
    dict2 = get_stoichiometry_dict(config_2)

    # Get dict which contains stoichiometry differences, negative values
    # indicate element is not present (at all) in one of the structures:
    diff_dict = get_stoichiometry_diff(dict1, dict2)

    # Remove species which are ignored for stoichiometry matching:
    for species in ignore_species:
        diff_dict.pop(species, None)

    # --- Starting checkes which may return False early: ---------------
    # Check that no species is missing if all_species_present:
    if np.min(list(diff_dict.values())) < -1e-4 and all_species_present:
        return False

    # Check species in max_diff_dict:
    for species, max_species_diff in max_diff_dict.items():
        if species in diff_dict:
            if np.abs(diff_dict[species]) > max_species_diff:
                return False
            # Remove species from diff_dict to avoid checking later
            diff_dict.pop(species, None)
    
    if len(diff_dict) != 0:
        if np.max(np.abs(list(diff_dict.values()))) > max_diff:
            return False

    return True
# --- End Stoichiometry utils ------------------------------------------