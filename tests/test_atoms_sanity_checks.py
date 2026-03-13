import pathlib
import pytest

from ase.io import Trajectory

from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    are_all_atoms_connected)

@pytest.mark.mini
def test_all_atoms_connected_connected_Ti_OH():
    dir_path = pathlib.Path(__file__).parent
    path = dir_path / 'test_structures/TiO2_adsorbed_OH.traj'
    traj = Trajectory(path.resolve())
    atoms_adrobed_OH = traj[-1]
        
    assert are_all_atoms_connected(atoms_adrobed_OH)

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









