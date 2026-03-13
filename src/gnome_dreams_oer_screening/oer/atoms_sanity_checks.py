import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs

from gnome_dreams_oer_screening.oer.oer_adsorption_site_study import (
    OER_adsorption_site_study,
)


def are_all_atoms_connected(
    atoms: Atoms, cutoff_multiplier: float = 1.2
) -> bool:
    """
    Test whether all atoms belong to one connected component based on
    ASE's natural cutoffs and neighbor list.
    """

    if len(atoms) <= 1:
        return True

    cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
    nl = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True,
        skin=1e-4,
    )
    nl.update(atoms)

    visited = {0}
    stack = [0]

    while stack:
        index = stack.pop()
        neighbors, _ = nl.get_neighbors(index)
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return len(visited) == len(atoms)


# def _get_adsorbed_O_index(
#     study: OER_adsorption_site_study, atoms: Atoms
# ) -> int:
#     if study.site_type == "on-top":
#         target_tag = 4
#     elif study.site_type == "lattice O":
#         target_tag = 3
#     else:
#         raise ValueError(f"Unsupported adsorption site type: {study.site_type}")

#     O_indices = [
#         i
#         for i, atom in enumerate(atoms)
#         if atom.symbol == "O" and atom.tag == target_tag
#     ]
#     if len(O_indices) != 1:
#         raise ValueError(
#             "Could not identify a unique adsorbed O atom "
#             f"with tag={target_tag}."
#         )

#     return int(O_indices[0])


# def is_adsorbate_near_site(
#     study: OER_adsorption_site_study,
#     atoms: Atoms,
#     max_site_drift: float = 1.5,
#     mic: bool = True,
# ) -> bool:
#     """
#     Test if the adsorbed O atom has drifted too far from the initial
#     adsorption position for the site study.
#     """

#     if max_site_drift <= 0:
#         raise ValueError("max_site_drift must be positive.")

#     O_index = _get_adsorbed_O_index(study, atoms)
#     displacement = atoms[O_index].position - np.asarray(
#         study.init_add_position, dtype=float
#     )

#     if mic:
#         displacement, _ = find_mic(
#             displacement[None, :], cell=atoms.cell, pbc=atoms.pbc
#         )
#         displacement = displacement[0]

#     drift_distance = float(np.linalg.norm(displacement))
#     return drift_distance <= max_site_drift


# def is_OH_adsorbate_intact(
#     study: OER_adsorption_site_study,
#     atoms: Atoms,
#     min_OH_distance: float = 0.6,
#     max_OH_distance: float = 1.5,
#     mic: bool = True,
# ) -> bool:
#     """
#     Test if O and H in an OH adsorbate remain bonded.
#     """

#     if min_OH_distance <= 0:
#         raise ValueError("min_OH_distance must be positive.")
#     if max_OH_distance <= min_OH_distance:
#         raise ValueError(
#             "max_OH_distance must be larger than min_OH_distance."
#         )

#     O_index = _get_adsorbed_O_index(study, atoms)
#     H_indices = [
#         i
#         for i, atom in enumerate(atoms)
#         if atom.symbol == "H" and atom.tag == 4
#     ]
#     if len(H_indices) != 1:
#         raise ValueError(
#             "Could not identify a unique OH adsorbate H atom with tag=4."
#         )
#     H_index = int(H_indices[0])

#     d_OH = float(atoms.get_distance(O_index, H_index, mic=mic))
#     return min_OH_distance <= d_OH <= max_OH_distance
