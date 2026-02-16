import numpy as np
import pathlib
import pytest

from ase.io import read, Trajectory
from ase.visualize import view
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



def test_temp_name(test_cands_traj):
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
        only_reduced_coord_O_sotes = True
    )

    assert len(df) == 6