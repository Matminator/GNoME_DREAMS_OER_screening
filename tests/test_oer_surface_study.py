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
OER_surface_study)
from gnome_dreams_oer_screening import Explog

@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = pathlib.Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)
    if atoms is None:
        pytest.fail("Slab generation failed")
    return atoms, oer_study

@pytest.fixture(scope="module")
def TiO2_r_surfaces_predicted_O_coordination(TiO2_r_bulk):
    atoms, oer_study_not_used = TiO2_r_bulk

    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.set_relaxed_structure(atoms_to_relax)
    oer_study.predict_most_likely_surfaces(method = 'O-coordination')
    return oer_study

@pytest.fixture(scope="module")
def TiO2_r_init_surface_study(TiO2_r_bulk):
    atoms, oer_study_not_used = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)

    atoms_to_relax = study.get_init_atoms()
    study.set_relaxed_structure(atoms_to_relax)
    study.predict_most_likely_surfaces(method = 'O-coordination')

    # From the stody obcjet:
    df = study.get_init_adsorption_sites_df(0,
        only_reduced_coord_O_sites=False)

    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    init_surface = surface_study.get_non_relaxed_surface()
    fake_energy = 123.456
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

def test_adsorbtion_sites_init_and_dict(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    i_inits = []
    for i in df_relaxed['Site index']:
        surface_study.initialize_adsorption_site_study(i)
        i_inits.append(i)

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()
    assert set(i_inits) == set(ad_site_dict.keys())

    

    
# print(E_O, E_star)
#     -288.2490000000001 -283.19200000000006