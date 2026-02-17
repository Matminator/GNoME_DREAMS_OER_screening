import numpy as np
import pathlib
import pytest

from ase import Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from ase.neighborlist import NeighborList, natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure

from gnome_dreams_oer_screening import OER_catalyst_study
from gnome_dreams_oer_screening.oer.ad_site_determination import (
    determine_adsorption_sites
)

@pytest.fixture(scope="module")
def test_cands_traj():
    dir_path = pathlib.Path(__file__).parent
    path = dir_path / '../notebooks/test_cands.traj'
    traj = Trajectory(path.resolve())
    if traj is None:
        pytest.fail("Slab generation failed")
    return traj

def test_sites_are_found(test_cands_traj):
    atoms = test_cands_traj[-1]

    # Setup of study object
    study = OER_catalyst_study(atoms,     
    H2O_gas_free_energy = -14.217,
    H2_gas_free_energy = -6.77,
    use_init_as_relaxed_bulk = True
    )

    # Make surfaces:
    study.predict_most_likely_surfaces(max_miller = 1)

    # Get a termination (as a Slab object):
    slab = list(study.terminations.values())[0]

    df, atoms = determine_adsorption_sites(slab,
        only_reduced_coord_O_sites = True
    )

    assert len(df) > 0, "No adsorption sites were determined"
    
def test_site_distances(test_cands_traj):
    atoms = test_cands_traj[-1]

    # Setup of study object
    study = OER_catalyst_study(atoms,     
    H2O_gas_free_energy = -14.217,
    H2_gas_free_energy = -6.77,
    use_init_as_relaxed_bulk = True
    )

    # Make surfaces:
    study.predict_most_likely_surfaces(max_miller = 1)

    # Get a termination (as a Slab object):
    slab = list(study.terminations.values())[0]

    df, atoms = determine_adsorption_sites(slab,
        only_reduced_coord_O_sites = True
    )

    assert len(df) > 2
    
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False,
                      bothways=True)
    nl.update(atoms)
    
    for i, row in df.iterrows():
        if row['site type'] == 'on-top':
            ad_site_pos = row['position']
            atoms_copy = atoms.copy()
            atoms_copy += Atoms('X', positions=[ad_site_pos])
            atoms_copy.center(vacuum = 10.0, axis = 2)
            x_index = len(atoms_copy) - 1
            if atoms_copy[x_index].symbol != 'X':
                raise ValueError('The ad site position was not ' + \
                    'added  as an atom to the copied Atoms object.')
            for atom in atoms_copy:
                if atom.symbol == 'X':
                    continue

                distance = atoms_copy.get_distance(atom.index, x_index,
                    mic=True)

                assert distance > 1.2, f"Adsorption site {i} is too " \
                    "close to atom {atom.index} (distance: " \
                    f"{distance:.2f} Å)"
    
    for i, row in df.iterrows():
        for distance_tb in row['ad site neighboring elements']:
            species, distance = distance_tb
            assert isinstance(species, str), f"Expected string for "\
                f"species, got {type(species)}"
            assert isinstance(distance, (int, float)), f"Expected int "\
                f"or float for distance, got {type(distance)}"
            assert distance > .5, f"Adsorption site {i} has a "\
                  f"neighboring atom at distance {distance:.2f} Å, "\
                    "which is too close."
