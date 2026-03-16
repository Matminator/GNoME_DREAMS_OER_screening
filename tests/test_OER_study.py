import pathlib
import pytest
import numpy as np

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import OER_catalyst_study
import gnome_dreams_oer_screening.oer.oer_study as oer_study_module
from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    StructureConsistencyError,
)
    
# --- Initilization tests of the OER_catalyst_study class: -------------
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

@pytest.mark.mini
def test_error_apon_setting_relaxed_structure():

    atoms_1 = Atoms('Ti', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)
    atoms_2 = Atoms('Fe', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_study =OER_catalyst_study(
        atoms_1, H2_gas_free_energy = -2, H2O_gas_free_energy = -4
    )

    assert oer_study.get_init_atoms() ==  atoms_1    
    with pytest.raises(ValueError):
        oer_study.set_relaxed_structure(atoms_2)


@pytest.mark.mini
def test_set_relaxed_structure_raises_structure_consistency_error():
    atoms = Atoms(
        "TiO",
        positions=[[0, 0, 0], [1, 0, 0]],
        cell=[2, 2, 2],
        pbc=True,
    )
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4
    )

    reordered_atoms = oer_study.get_init_atoms()[[1, 0]]

    with pytest.raises(
        StructureConsistencyError, match="different symbols"
    ):
        oer_study.set_relaxed_structure(reordered_atoms)
# --- End of initilization tests of the OER_catalyst_study class: ------

# --- TEMP TITEL... 
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

# Testing that the correct bulk 'bulk-custom-coord' property is set:
@pytest.mark.mini
def test__set_bulk_coordination(TiO2_r_bulk):
    atoms, oer_study = TiO2_r_bulk
    m = 'all'
    oer_study._set_bulk_coordination(method=m)

    bulk_structure = oer_study.get_bulk_structure()
    for site in bulk_structure.sites:
        if site.species_string == 'Ti':
            assert site.properties['bulk-custom-coord'] == 8
        elif site.species_string == 'O':
            assert site.properties['bulk-custom-coord'] == 3
        else:
            raise ValueError("Unexpected element in structure")

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

def test_error_apon_resetting_relaxed_structure(
        TiO2_r_surfaces_predicted_O_coordination):
    oer_study = TiO2_r_surfaces_predicted_O_coordination

    atoms_to_relax = oer_study.get_init_atoms()
    with pytest.raises(RuntimeError):
        oer_study.set_relaxed_structure(atoms_to_relax)

def test_TiO2_r_110_surface_score_calculation_O_coordination(
        TiO2_r_surfaces_predicted_O_coordination):
    oer_study = TiO2_r_surfaces_predicted_O_coordination

    df = oer_study.get_termination_rankings()
    row = df[df['Miller indices'] == (1,1,0)]
    termination_index = row['Termination index'].values[0]
    score = row['Score'].values[0]
    slab = oer_study.terminations[termination_index]
    slab_atoms = AseAtomsAdaptor.get_atoms(slab)

    cell = slab_atoms.get_cell()
    A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

    # Ti: 1: -1, (O: 1: -1, but ignored in this method)
    expected_reduction = -1
    expected_score = expected_reduction / A_surface

    score_diff = abs(score - expected_score)
    assert score_diff < 1e-6, f"Expected score: {expected_score}, but" \
        f" got {score}"
    
@pytest.fixture(scope="module")
def TiO2_r_surfaces_predicted_all_coordination(TiO2_r_bulk):
    atoms, oer_study_not_used = TiO2_r_bulk

    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.set_relaxed_structure(atoms_to_relax)
    oer_study.predict_most_likely_surfaces(method = 'all')
    return oer_study

def test_TiO2_r_110_surface_score_calculation_all_coordination(
        TiO2_r_surfaces_predicted_all_coordination):
    oer_study = TiO2_r_surfaces_predicted_all_coordination

    df = oer_study.get_termination_rankings()
    row = df[df['Miller indices'] == (1,1,0)]
    termination_index = row['Termination index'].values[0]
    score = row['Score'].values[0]
    slab = oer_study.terminations[termination_index]
    slab_atoms = AseAtomsAdaptor.get_atoms(slab)

    cell = slab_atoms.get_cell()
    A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

    # Ti: 1: -1, O: 1: -1
    expected_reduction = -1 - 1
    expected_score = expected_reduction / A_surface

    score_diff = abs(score - expected_score)
    assert score_diff < 1e-6, f"Expected score: {expected_score}, but" \
        f" got {score}"



# --- Testing timeout and assoriated flag in _build_surfaces: ----------
@pytest.mark.mini
def test_build_surfaces_raises_if_previous_timeout_failed():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
        use_init_as_relaxed_bulk=True,
    )
    study.failed_slab_generation = True

    with pytest.raises(RuntimeError, match="Previous slab generation attempt"):
        study._build_surfaces((1, 0, 0), timeout=1)


@pytest.mark.mini
def test_build_surfaces_timeout_sets_failed_flag(monkeypatch):
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
        use_init_as_relaxed_bulk=True,
    )

    time_values = iter([0.0, 2.0])
    monkeypatch.setattr(oer_study_module.time, "time", lambda: next(time_values))

    with pytest.raises(
        TimeoutError,
        match="Slab generation timed out\\.It is not possible to generate surfaces for this bulk structure\\.",
    ):
        study._build_surfaces((1, 0, 0), timeout=1)

    assert study.failed_slab_generation is True
# --- End of testing timeout and assoriated flag in _build_surfaces ----
