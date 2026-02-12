import pathlib
import pytest

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import OER_catalyst_study
    

@pytest.mark.mini
def test_setup_1():

    atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_study =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True
    )

    assert oer_study.get_bulk_atoms() ==  atoms
    assert oer_study.get_bulk_structure() == \
        AseAtomsAdaptor.get_structure(atoms)
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4

@pytest.mark.mini
def test_setup_2():

    atoms = Atoms('Ti', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_study =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True
    )

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.set_relaxed_structure(atoms_to_relax)

    assert oer_study.get_bulk_atoms() ==  atoms
    assert oer_study.get_bulk_structure() == \
        AseAtomsAdaptor.get_structure(atoms)
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4

def test_error_apon_resetting_relaxed_structure():

    dir_path = pathlib.Path(__file__).parent
    ZrO2_m_path = dir_path / '../notebooks/ZrO2_m.cif'
    atoms = read(ZrO2_m_path.resolve(), format='cif')

    oer_study =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.set_relaxed_structure(atoms_to_relax)

    oer_study.predict_most_likely_surfaces()

    with pytest.raises(RuntimeError):
        oer_study.set_relaxed_structure(atoms_to_relax)

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
def test__determine_coordination_numbers_bulk_TiO2_r_a(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'all'
    coords = oer_study._determine_coordination_numbers(atoms, method=m)
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 8
        elif atom.symbol == 'O':
            assert coords[atom.index] == 3
        else:
            raise ValueError("Unexpected element in structure")

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_TiO2_r_acoO(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'disregard-coord-of-O'
    coords = oer_study._determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 8
            i += 1

@pytest.mark.mini
def test__determine_coordination_numbers_bulk_TiO2_r_Oc(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'O-coordination'
    coords = oer_study._determine_coordination_numbers(atoms, method=m)
    i = 0
    for atom in atoms:
        if atom.symbol == 'Ti':
            assert coords[atom.index] == 6
            i += 1


