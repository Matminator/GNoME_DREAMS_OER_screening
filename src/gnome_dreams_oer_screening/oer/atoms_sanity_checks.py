import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs


class StructureConsistencyError(ValueError):
    """Raised when a relaxed structure is inconsistent with its reference."""


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

def are_systems_equal(atoms1: Atoms, atoms2: Atoms
    ) -> tuple[bool, str]:
    """
    Test if two ASE Atoms objects represent the same system, ignoring
    atomic positions, but comparing atomic symbols, tags and ordering -
    the latter at an superficial level. Intented to be used for testing 
    if the correct system is provided after a relaxation - and if the 
    ordering has not changed (important for tag-based identification of 
    adsorption sites).

    Args:
        atoms1 (Atoms): First Atoms object to compare.
        atoms2 (Atoms): Second Atoms object to compare.

    Returns:
        tuple[bool, str]: A tuple where the bool indicates if the
                           systems are equal, and the str provides a
                           message in case they are not.
    """

    equal = True
    message_parts = []

    if len(atoms1) != len(atoms2):
        equal = False
        message_parts.append(
            f"Number of atoms differ: {len(atoms1)} vs {len(atoms2)}"
        )
    
    for i, (atom1, atom2) in enumerate(zip(atoms1, atoms2)):
        if atom1.symbol != atom2.symbol:
            equal = False
            message_parts.append(
                f"Atom {i} has different symbols: "
                f"{atom1.symbol} vs {atom2.symbol}"
            )
        if atom1.tag != atom2.tag:
            equal = False
            message_parts.append(
                f"Atom {i} has different tags: {atom1.tag} "
                f"vs {atom2.tag}"
            )

    return equal, "; ".join(message_parts)

