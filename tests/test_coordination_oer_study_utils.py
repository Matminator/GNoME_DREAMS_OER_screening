import pathlib
import pytest
import numpy as np

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import OER_catalyst_study


from gnome_dreams_oer_screening.oer.oer_study_utils import (
    determine_coordination_numbers
    )

# ______________________________________________________________________
# ___ Testing of coordination number methods for study object __________
# --- Testing coordination determination for bulk TiO2 rutile ----------
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

@pytest.mark.mini
def test_determine_coordination_numbers_bulk_TiO2_r_a(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'all'
    coords = determine_coordination_numbers(atoms, method=m)
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 8
        elif atom.symbol == 'O':
            assert coords[atom.index] == 3
        else:
            raise ValueError("Unexpected element in structure")

@pytest.mark.mini
def test_determine_coordination_numbers_bulk_TiO2_r_acoO(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'disregard-coord-of-O'
    coords = determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 8
            i += 1

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_TiO2_r_Oc(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'O-coordination'
    coords = determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 6
            i += 1

# ----------------------------------------------------------------------
# --- Testing coordination determination for bulk ZrO2 monoclinic ------
@pytest.fixture(scope="module")
def ZrO2_m_bulk():
    dir_path = pathlib.Path(__file__).parent
    ZrO2_m_path = dir_path / '../notebooks/ZrO2_m.cif'
    atoms = read(ZrO2_m_path.resolve(), format='cif')
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)
    if atoms is None:
        pytest.fail("Slab generation failed")
    return atoms, oer_study

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_ZrO2_m_a(ZrO2_m_bulk):
    atoms, oer_study = ZrO2_m_bulk
    m = 'all'
    coords = determine_coordination_numbers(atoms, method=m)
    num_O_cords = [0, 0]
    for atom in atoms:
        if atom.symbol == 'Zr':
            assert coords[atom.index] == 13
        elif atom.symbol == 'O':
            if coords[atom.index] == 3:
                num_O_cords[0] += 1
            elif coords[atom.index] == 4:
                num_O_cords[1] += 1
        else:
            raise ValueError("Unexpected element in structure")
        
    assert num_O_cords[0] == 4
    assert num_O_cords[1] == 4

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_ZrO2_m_acoO(ZrO2_m_bulk):
    atoms, oer_study = ZrO2_m_bulk
    m = 'disregard-coord-of-O'
    coords = determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Zr':
            assert coords[atom.index] == 13
            i += 1

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_ZrO2_m_Oc(ZrO2_m_bulk):
    atoms, oer_study = ZrO2_m_bulk
    m = 'O-coordination'
    coords = determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Zr':
            assert coords[atom.index] == 7
            i += 1
# ----------------------------------------------------------------------
# ______________________________________________________________________
