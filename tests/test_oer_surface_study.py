# --- standard library imports ---
import os
import time
import numpy as np
import pandas as pd
import pathlib
import pytest
import copy

from ase import Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor

# --- IMPORTS FROM THIS PROJECT ---
from gnome_dreams_oer_screening import (OER_catalyst_study, 
OER_surface_study, OER_adsorption_site_study)
from gnome_dreams_oer_screening import Explog
from gnome_dreams_oer_screening.oer.oer_surface_study import (
    SurfaceStudyLockedError,
)
from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    StructureConsistencyError,
)

@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = pathlib.Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -6.77, 
        H2O_gas_free_energy = -14.217, 
        use_init_as_relaxed_bulk=True)
    if atoms is None:
        pytest.fail("Slab generation failed")
    return atoms, oer_study

# @pytest.fixture(scope="module")
# def TiO2_r_surfaces_predicted_O_coordination(TiO2_r_bulk):
#     atoms, oer_study_not_used = TiO2_r_bulk

#     oer_study = OER_catalyst_study(
#         atoms, H2_gas_free_energy = -6.77, H2O_gas_free_energy = -14.217, 
#         use_init_as_relaxed_bulk=True)

#     atoms_to_relax = oer_study.get_init_atoms()
#     oer_study.set_relaxed_structure(atoms_to_relax)
#     oer_study.predict_most_likely_surfaces(method = 'O-coordination')
#     return oer_study

@pytest.fixture(scope="module")
def TiO2_r_init_surface_study(TiO2_r_bulk):
    atoms, study = TiO2_r_bulk

    # study = OER_catalyst_study(
    #     atoms, H2_gas_free_energy = -6.77, H2O_gas_free_energy = -14.217, 
    #     use_init_as_relaxed_bulk=True)

    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method = 'O-coordination')

    # From the stody obcjet:
    df = study.get_init_adsorption_sites_df(0,
        only_reduced_coord_O_sites=False)

    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    init_surface = surface_study.get_non_relaxed_surface()
    fake_energy = -278.13500000000005 # NOTE: Needed for later test: test_adsorbtion_update_adsorption_energies
    surface_study.set_relaxed_base_surface(init_surface, fake_energy)
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=False)
    
    # From the surface study object:
    df_relaxed = surface_study.get_adsorption_sites_df() 

    return surface_study, df, df_relaxed

def test_adsorbtion_sites_setup(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)
    df = copy.deepcopy(df)
    df_relaxed = copy.deepcopy(df_relaxed)
    
    assert df['reduced coordination'].equals(
        df_relaxed['reduced coordination'])
    assert df['ad site element'].equals(df_relaxed['ad site element'])
    assert df['site type'].equals(df_relaxed['site type'])

def test_adsorbtion_sites_dict(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    i_inits = []
    for i in df_relaxed['Site index']:
        surface_study.initialize_adsorption_site_study(i)
        i_inits.append(i)

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()
    assert set(i_inits) == set(ad_site_dict.keys())


def test_adsorbtion_sites_init(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    for i in df_relaxed['Site index']:
        surface_study.initialize_adsorption_site_study(i)

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()

    for i in df_relaxed['Site index']:
        ad_site_study = ad_site_dict[i]
        # Is correct type:
        assert isinstance(ad_site_study, OER_adsorption_site_study)

        init_atoms = ad_site_study.init_surface

        df_row = df_relaxed[df_relaxed['Site index'] == i].iloc[0]
        site_type = df_row['site type']
        if site_type == 'on-top':
            expected_element = df_row['ad site element']
        elif site_type == 'lattice O':
            expected_element = 'O'
        else:
            raise ValueError("Invalid site type in test dataframe:", 
                             site_type)

        found_site = False
        for atom in init_atoms:
            if atom.tag == 3:
                if atom.symbol == expected_element:
                    found_site = True
                    break

        assert found_site, f'Failed to find expected site element '\
            f"{expected_element} for site type {site_type} in " \
            f"adsorption site study with index {i}"

def test_adsorbtion_update_O_adsorption_energies(
        TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    for i, row in df_relaxed.iterrows():
        surface_study.initialize_adsorption_site_study(i)
        ad_site_study = surface_study.\
            get_adsorption_site_studies_dict()[i]
        O_atoms = ad_site_study.get_atoms_for_O_calc()

        site_type = row['site type']
        if site_type == 'on-top':
            E = -283.19200000000006
        elif site_type == 'lattice O':
            E = -283.19200000000006  -2*(-5.057000000000016)
        else:
            raise ValueError("Invalid site type in test dataframe:", 
                             site_type)
        
        ad_site_study.set_relaxed_O_calc(O_atoms, energy = E)

    surface_study.update_adsorption_energies()
    df = surface_study.get_adsorption_sites_df()
    for i, row in df.iterrows():
        diff = row['G(O)'] - 2.46
        assert abs(diff) < 0.001, f"Expected G(O) = 2.46, got " \
            f"{row['G(O)']} for site {i} of type {row['site type']}"


def test_set_relaxed_base_surface_raises_structure_consistency_error(
        TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method='O-coordination')
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]

    reordered_surface = surface_study.get_non_relaxed_surface()

    swap_index = next(
        i for i, atom in enumerate(reordered_surface)
        if atom.symbol != reordered_surface[0].symbol
    )
    reordered_indices = list(range(len(reordered_surface)))
    reordered_indices[0], reordered_indices[swap_index] = (
        reordered_indices[swap_index],
        reordered_indices[0],
    )
    reordered_surface = reordered_surface[reordered_indices]

    with pytest.raises(
        StructureConsistencyError, match="different symbols"
    ):
        surface_study.set_relaxed_base_surface(
            reordered_surface, energy=-278.13500000000005
        )


def test_set_relaxed_base_surface_can_retry_after_missing_energy_error(
        TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)

    surface_study = study.get_surface_studies()[termination_index]

    relaxed_surface = surface_study.get_non_relaxed_surface()

    # First attempt should raise error due to missing energy:
    with pytest.raises(
        ValueError,
        match="Energy not provided and could not be obtained",
    ):
        surface_study.set_relaxed_base_surface(relaxed_surface)

    success = surface_study.set_relaxed_base_surface(
        relaxed_surface, energy=-278.13500000000005
    )

    # The second attempt should succeed, and the relaxed surface and 
    # energy should be set correctly:
    assert success is True
    assert surface_study.relaxed_surface_energy == -278.13500000000005
    assert surface_study.get_relaxed_surface() is not None


def test_set_relaxed_base_surface_locks_study_for_disconnected_surface(
        TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    # Make a disconnected surface:
    disconnected_surface = surface_study.get_non_relaxed_surface()
    disconnected_surface.set_pbc(False)
    disconnected_surface[-1].position += np.array([50.0, 50.0, 50.0])

    success = surface_study.set_relaxed_base_surface(
        disconnected_surface, energy=-278.13500000000005
    )

    # Success should be False, as the surface should not be set:
    assert success is False
    assert surface_study.study_locked is True
    assert surface_study.relaxed_surface_energy is None
    with pytest.raises(
        ValueError,
        match="No relaxed surface has been provided",
    ):
        surface_study.get_relaxed_surface()

    # A later valid relaxed surface should unlock the study and be
    # accepted:
    valid_surface = surface_study.get_non_relaxed_surface()
    retry_success = surface_study.set_relaxed_base_surface(
        valid_surface,
        energy=-278.13500000000005,
    )

    assert retry_success is True
    assert surface_study.study_locked is False
    assert surface_study.relaxed_surface_energy == -278.13500000000005
    assert surface_study.get_relaxed_surface() == valid_surface


def test_determine_adsorption_sites_raises_specific_error_for_locked_study(
        TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    disconnected_surface = surface_study.get_non_relaxed_surface()
    disconnected_surface.set_pbc(False)
    disconnected_surface[-1].position += np.array([50.0, 50.0, 50.0])

    success = surface_study.set_relaxed_base_surface(
        disconnected_surface, energy=-278.13500000000005
    )

    assert success is False

    with pytest.raises(
        SurfaceStudyLockedError,
        match="surface study is locked due to a previously provided "
        "non-continuous relaxed surface",
    ):
        surface_study.determine_adsorption_sites()

    # Test that a valied surface may be provided after the error:
    success = surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(), energy = 10
    )
    assert success is True
    assert surface_study.study_locked is False


    # Test that the method can be called successfully after providing a 
    # valid relaxed surface
    try:
        surface_study.determine_adsorption_sites()
    except SurfaceStudyLockedError:
        pytest.fail(
            "determine_adsorption_sites should not raise "
            "SurfaceStudyLockedError after providing a valid relaxed " \
            "surface"
        )
