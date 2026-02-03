import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab, Structure


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
        species_must_be_present: bool = True,
        stoichiometric_match_species: list[str] = [],
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
        species_must_be_present (bool, optional): Whether all species
            in the stoichiometry difference dict must be present
            (non-negative) in both configurations. Defaults to True.
        stoichiometric_match_species (list[str], optional): List of 
            species which must match exactly between the two 
            configurations. Defaults to [].
        ignore_species (list[str], optional): List of species to ignore
            when evaluating stoichiometry difference. Defaults to [].

    Returns:
        bool: True if the stoichiometry difference meets the criteria,
        False otherwise.
    """
    
    # Check that no species is both in stoichiometric_match_species and 
    # ignore_species:
    for species in stoichiometric_match_species:
        if species in ignore_species:
            raise ValueError("Species cannot be both in \
            stoichiometric_match_species and ignore_species.")
    
    # get_stoichiometry_dict: agnostic to atoms or structure:
    dict1 = get_stoichiometry_dict(config_1) 
    dict2 = get_stoichiometry_dict(config_2)

    # Get dict which contains stoichiometry differences, negative values
    # indicate element is not present (at all) in one of the structures:
    diff_dict = get_stoichiometry_diff(dict1, dict2)

    # Remove species which are ignored for stoichiometry matching:
    for species in ignore_species:
        diff_dict.pop(species, None)

    # Check that species in stoichiometric_match_species
    for species in stoichiometric_match_species:
        if species in diff_dict:
            if np.abs(diff_dict[species]) > 1e-4:
                return False
        
    if min(list(diff_dict.values())) < 0 and species_must_be_present:
        return False
    elif max(np.abs(list(diff_dict.values()))) <= max_diff:
        return True
    else:
        return False
# --- End Stoichiometry utils ------------------------------------------