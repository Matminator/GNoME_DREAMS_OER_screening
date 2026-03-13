import pathlib
import pytest
from ase.io import Trajectory

from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    are_all_atoms_connected, are_systems_equal)

@pytest.fixture
def atoms_adsorbed_oh():
    dir_path = pathlib.Path(__file__).parent
    path = dir_path / 'test_structures/TiO2_adsorbed_OH.traj'
    traj = Trajectory(path.resolve())
    return traj[-1]

@pytest.mark.mini
def test_all_atoms_connected_connected_Ti_OH(atoms_adsorbed_oh):
    assert are_all_atoms_connected(atoms_adsorbed_oh)

@pytest.mark.mini
def test_all_atoms_connected_Ti_displaced_H_in_OH():
    dir_path = pathlib.Path(__file__).parent
    path = dir_path / 'test_structures/TiO2_split_O-H_adsorbed.traj'
    traj = Trajectory(path.resolve())
    atoms_adrobed_OH = traj[-1]
        
    # must yield True, since the H and O are on the surface
    assert are_all_atoms_connected(atoms_adrobed_OH)

@pytest.mark.mini
def test_all_atoms_connected_Ti_desorbed_OH():
    dir_path = pathlib.Path(__file__).parent
    path = dir_path / 'test_structures/TiO2_desorbed_OH.traj'
    traj = Trajectory(path.resolve())
    atoms_adrobed_OH = traj[-1]
        
    # must yield False, since the OH is desorbed/disconnected
    assert not are_all_atoms_connected(atoms_adrobed_OH)

@pytest.mark.mini
def test_if_systems_are_equal_two_equal_systems(atoms_adsorbed_oh):    
    atoms_1 = atoms_adsorbed_oh.copy()
    atoms_2 = atoms_adsorbed_oh.copy()

    are_equal, message = are_systems_equal(atoms_1, atoms_2)
    assert are_equal
    assert message == ""

@pytest.mark.mini
def test_if_systems_are_equal_different_lengths(atoms_adsorbed_oh):    
    atoms_1 = atoms_adsorbed_oh.copy()
    atoms_2 = atoms_adsorbed_oh.copy()
    atoms_2.pop()  # Remove one atom to make the lengths different

    are_equal, message = are_systems_equal(atoms_1, atoms_2)
    assert not are_equal
    assert "Number of atoms differ" in message

@pytest.mark.mini
def test_if_systems_are_equal_different_tags(atoms_adsorbed_oh):    
    atoms_1 = atoms_adsorbed_oh.copy()
    atoms_2 = atoms_adsorbed_oh.copy()
    atoms_2[0].tag += 1  # Change the tag of the first atom

    are_equal, message = are_systems_equal(atoms_1, atoms_2)
    assert not are_equal
    assert "tags" in message

@pytest.mark.mini
def test_if_systems_are_equal_different_symbols(atoms_adsorbed_oh):    
    atoms_1 = atoms_adsorbed_oh.copy()
    atoms_2 = atoms_adsorbed_oh.copy()
    atoms_2[0].symbol = "C"  # Change the symbol of the first atom

    are_equal, message = are_systems_equal(atoms_1, atoms_2)
    assert not are_equal
    assert "symbols" in message

@pytest.mark.mini
def test_if_systems_are_equal_different_ordering(atoms_adsorbed_oh):
    atoms_1 = atoms_adsorbed_oh.copy()
    atoms_2 = atoms_adsorbed_oh.copy()

    swap_index = next(
        i for i, atom in enumerate(atoms_2) if atom.symbol != atoms_2[0].symbol
    )
    reordered_indices = list(range(len(atoms_2)))
    reordered_indices[0], reordered_indices[swap_index] = (
        reordered_indices[swap_index],
        reordered_indices[0],
    )
    atoms_2 = atoms_2[reordered_indices]

    are_equal, message = are_systems_equal(atoms_1, atoms_2)
    assert not are_equal
    assert "symbols" in message
