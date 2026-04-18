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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_study(symbols='Fe'):
    """Return a (study, atoms) pair with the relaxed bulk NOT yet set.
    Always produces a single-atom cell — symbols should be one element name."""
    atoms = Atoms(symbols, positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
    )
    return study, atoms


def _finalize(study, atoms, energy=-1.0, magmom=None):
    """Store a single candidate and finalize — convenience for fixtures."""
    if magmom is None:
        magmom = [0.0] * len(atoms)
    study.store_mag_bulk_candidate(0, atoms.copy(), energy, magmom)
    study.finalize_mag_bulk_batch()


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_getters_return_none_when_not_initialized():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
    )

    assert study.get_bulk_atoms() is None
    assert study.get_bulk_structure() is None
    assert study.get_distinct_millers() is None


@pytest.mark.mini
def test_getters_return_copies_after_initialization():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
    )
    study.set_relaxed_structure(atoms.copy())

    # Test init atoms are pobulated and returns copy
    init_atoms = study.get_init_atoms()
    assert init_atoms is not None
    assert init_atoms is not study.init_atoms

    # Test bulk atoms are populated and returns copy
    bulk_atoms = study.get_bulk_atoms()
    assert bulk_atoms is not None
    assert bulk_atoms is not study.bulk_atoms

    # Test modifying returned bulk-does not modify the study bulk atoms
    bulk_atoms.positions[0, 0] += 0.1
    assert not np.isclose(
        bulk_atoms.positions[0, 0], study.bulk_atoms.positions[0, 0]
    )

    bulk_structure = study.get_bulk_structure()
    assert bulk_structure is not None
    assert bulk_structure is not study.bulk_structure


@pytest.mark.mini
def test_setup_1():
    """Bulk is None until finalized via the magnetic workflow."""
    atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )
    assert oer_study.get_bulk_atoms() is None
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4


@pytest.mark.mini
def test_setup_2():
    """Setting the relaxed bulk via the magnetic workflow produces the correct result."""
    atoms = Atoms('Ti', positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    for atom in atoms:
        atom.tag = 0

    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.store_mag_bulk_candidate(0, atoms_to_relax, -1.0, [0.0])
    oer_study.finalize_mag_bulk_batch()

    assert oer_study.get_bulk_atoms() == atoms
    # bulk_structure carries site_properties["magmom"] after finalization, so
    # compare element formula and atom count rather than full structure equality:
    bulk_structure = oer_study.get_bulk_structure()
    assert bulk_structure.formula == AseAtomsAdaptor.get_structure(atoms).formula
    assert len(bulk_structure) == len(atoms)
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4


@pytest.mark.mini
def test_error_apon_setting_relaxed_structure():
    """Wrong-element candidate raises StructureConsistencyError at store time."""
    atoms_1 = Atoms('Ti', positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    atoms_2 = Atoms('Fe', positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)

    oer_study = OER_catalyst_study(
        atoms_1, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )

    assert oer_study.get_init_atoms() == atoms_1
    with pytest.raises(StructureConsistencyError):
        oer_study.store_mag_bulk_candidate(0, atoms_2, -1.0, [0.0])


@pytest.mark.mini
def test_set_relaxed_structure_raises_structure_consistency_error():
    """Foreign element raises StructureConsistencyError at store time."""
    atoms = Atoms(
        "TiO",
        positions=[[0, 0, 0], [1, 0, 0]],
        cell=[2, 2, 2],
        pbc=True,
    )
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )

    foreign_atoms = Atoms(
        "TiCu",
        positions=[[0, 0, 0], [1, 0, 0]],
        cell=[2, 2, 2],
        pbc=True,
    )
    with pytest.raises(StructureConsistencyError, match="not present in"):
        oer_study.store_mag_bulk_candidate(0, foreign_atoms, -1.0, [0.0, 0.0])


# ---------------------------------------------------------------------------
# TiO2 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = pathlib.Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    if atoms is None:
        pytest.fail("Could not read TiO2 CIF")
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )
    _finalize(oer_study, oer_study.get_init_atoms())
    return atoms, oer_study


@pytest.fixture(scope="module")
def TiO2_r_surfaces_predicted_O_coordination(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )
    _finalize(oer_study, oer_study.get_init_atoms())
    oer_study.predict_most_likely_surfaces(method='O-coordination')
    return oer_study


@pytest.fixture(scope="module")
def TiO2_r_surfaces_predicted_all_coordination(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2, H2O_gas_free_energy=-4,
    )
    _finalize(oer_study, oer_study.get_init_atoms())
    oer_study.predict_most_likely_surfaces(method='all')
    return oer_study


# ---------------------------------------------------------------------------
# Bulk coordination test
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test__set_bulk_coordination(TiO2_r_bulk):
    _, oer_study = TiO2_r_bulk
    oer_study._set_bulk_coordination(method='all')

    bulk_structure = oer_study.get_bulk_structure()
    for site in bulk_structure.sites:
        if site.species_string == 'Ti':
            assert site.properties['bulk-custom-coord'] == 8
        elif site.species_string == 'O':
            assert site.properties['bulk-custom-coord'] == 3
        else:
            raise ValueError("Unexpected element in structure")


# ---------------------------------------------------------------------------
# Surface score tests
# ---------------------------------------------------------------------------

def test_TiO2_r_110_surface_score_calculation_O_coordination(
        TiO2_r_surfaces_predicted_O_coordination):
    oer_study = TiO2_r_surfaces_predicted_O_coordination

    df = oer_study.get_termination_rankings()
    row = df[df['Miller indices'] == (1, 1, 0)]
    termination_index = row['Termination index'].values[0]
    score = row['Score'].values[0]
    slab = oer_study.terminations[termination_index]
    slab_atoms = AseAtomsAdaptor.get_atoms(slab)

    cell = slab_atoms.get_cell()
    A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

    expected_score = -1 / A_surface  # Ti: 1 undercoordinated; O ignored
    assert abs(score - expected_score) < 1e-6, \
        f"Expected {expected_score}, got {score}"


def test_TiO2_r_110_surface_score_calculation_all_coordination(
        TiO2_r_surfaces_predicted_all_coordination):
    oer_study = TiO2_r_surfaces_predicted_all_coordination

    df = oer_study.get_termination_rankings()
    row = df[df['Miller indices'] == (1, 1, 0)]
    termination_index = row['Termination index'].values[0]
    score = row['Score'].values[0]
    slab = oer_study.terminations[termination_index]
    slab_atoms = AseAtomsAdaptor.get_atoms(slab)

    cell = slab_atoms.get_cell()
    A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

    expected_score = (-1 - 1) / A_surface  # Ti: -1, O: -1
    assert abs(score - expected_score) < 1e-6, \
        f"Expected {expected_score}, got {score}"


# ---------------------------------------------------------------------------
# Second finalize raises (replaces old "reset after surfaces" test)
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_error_on_second_finalize():
    """Second call to finalize_mag_bulk_batch must raise RuntimeError."""
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [0.0])
    study.finalize_mag_bulk_batch()

    study.store_mag_bulk_candidate(1, atoms.copy(), -2.0, [0.0])
    with pytest.raises(RuntimeError):
        study.finalize_mag_bulk_batch()


# ---------------------------------------------------------------------------
# Timeout test
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_build_surfaces_timeout_returns_empty_list(monkeypatch):
    study, atoms = _make_study()
    _finalize(study, atoms)

    time_values = iter([0.0, 2.0])
    monkeypatch.setattr(oer_study_module.time, "time", lambda: next(time_values))

    slabs = study._build_surfaces((1, 0, 0), timeout=1)
    assert slabs == []


# ---------------------------------------------------------------------------
# Magnetic workflow — initial state
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_mag_bulk_initial_state():
    study, _ = _make_study()
    assert study.is_mag_bulk_batch_finalized() is False
    assert study.get_bulk_magmom() is None
    assert study.get_bulk_atoms() is None


# ---------------------------------------------------------------------------
# store_mag_bulk_candidate
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_store_mag_bulk_candidate():
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), -2.5, [2.0])
    assert 0 in study._mag_bulk_candidates
    _, stored_energy, stored_magmom = study._mag_bulk_candidates[0]
    assert stored_energy == -2.5
    assert stored_magmom == [2.0]


@pytest.mark.mini
def test_store_mag_bulk_candidate_wrong_magmom_length_raises():
    """magmom length mismatch is caught at store time."""
    atoms = Atoms('FeO', positions=[[0, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
    study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-2.0, H2O_gas_free_energy=-4.0,
    )
    with pytest.raises(ValueError):
        study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [2.0])  # len 1 != 2


# ---------------------------------------------------------------------------
# finalize_mag_bulk_batch
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_finalize_mag_bulk_batch_empty_returns_false():
    study, _ = _make_study()
    assert study.finalize_mag_bulk_batch() is False
    assert study.get_bulk_atoms() is None


@pytest.mark.mini
def test_finalize_mag_bulk_batch_single_candidate():
    study, atoms = _make_study()
    magmom = [2.0]
    study.store_mag_bulk_candidate(0, atoms.copy(), -3.0, magmom)
    result = study.finalize_mag_bulk_batch()

    assert result is True
    assert study.is_mag_bulk_batch_finalized() is True
    assert study.get_bulk_atoms() is not None
    assert study.get_bulk_magmom() == magmom


@pytest.mark.mini
def test_finalize_mag_bulk_batch_picks_lowest_energy_per_atom():
    """Candidate with lower energy/atom wins."""
    study, atoms = _make_study()
    # candidate 0: -1.0/atom,  candidate 1: -3.0/atom  ← winner
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [2.0])
    study.store_mag_bulk_candidate(1, atoms.copy(), -3.0, [1.0])
    study.finalize_mag_bulk_batch()

    assert study.get_bulk_magmom() == [1.0]


@pytest.mark.mini
def test_finalize_mag_bulk_batch_smaller_cell_wins_on_epa():
    """1-atom primitive wins when its E/atom is lower than the 2-atom candidate.

    The study is initialised with a 2-atom cell; the 1-atom candidate has fewer
    atoms but contains only elements already in the reference — assert_elements_subset
    must accept it, and finalize_mag_bulk_batch must select it on E/atom grounds.
    """
    atoms_2 = Atoms('FeFe', positions=[[0, 0, 0], [2, 2, 2]], cell=[4, 4, 4], pbc=True)
    study = OER_catalyst_study(atoms_2, H2_gas_free_energy=-2.0, H2O_gas_free_energy=-4.0)
    atoms_1 = Atoms('Fe', positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)

    # 1-atom: E/atom = -3.0 (winner);  2-atom: E/atom = -1.0
    study.store_mag_bulk_candidate(0, atoms_1.copy(), -3.0, [2.0])
    study.store_mag_bulk_candidate(1, atoms_2.copy(), -2.0, [2.0, -2.0])
    study.finalize_mag_bulk_batch()

    assert study.get_bulk_magmom() == [2.0], (
        "1-atom candidate (E/atom=-3) must beat the 2-atom candidate (E/atom=-1)"
    )
    assert len(study.get_bulk_atoms()) == 1


@pytest.mark.mini
def test_finalize_mag_bulk_batch_larger_cell_wins_on_epa():
    """2-atom conventional cell wins when its E/atom is lower than the 1-atom candidate."""
    atoms_2 = Atoms('FeFe', positions=[[0, 0, 0], [2, 2, 2]], cell=[4, 4, 4], pbc=True)
    study = OER_catalyst_study(atoms_2, H2_gas_free_energy=-2.0, H2O_gas_free_energy=-4.0)
    atoms_1 = Atoms('Fe', positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)

    # 1-atom: E/atom = -1.0;  2-atom: E/atom = -3.0 (winner)
    study.store_mag_bulk_candidate(0, atoms_1.copy(), -1.0, [2.0])
    study.store_mag_bulk_candidate(1, atoms_2.copy(), -6.0, [2.0, -2.0])
    study.finalize_mag_bulk_batch()

    assert study.get_bulk_magmom() == [2.0, -2.0], (
        "2-atom candidate (E/atom=-3) must beat the 1-atom candidate (E/atom=-1)"
    )
    assert len(study.get_bulk_atoms()) == 2


@pytest.mark.mini
def test_finalize_mag_bulk_batch_sets_bulk_structure_site_properties():
    """Winning magmom is written into bulk_structure.site_properties for SlabGenerator."""
    study, atoms = _make_study()
    magmom = [2.5]
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, magmom)
    study.finalize_mag_bulk_batch()

    site_magmoms = study.get_bulk_structure().site_properties.get("magmom")
    assert site_magmoms is not None
    assert list(site_magmoms) == magmom


@pytest.mark.mini
def test_finalize_mag_bulk_batch_raises_on_double_call():
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [0.0])
    study.finalize_mag_bulk_batch()

    study.store_mag_bulk_candidate(1, atoms.copy(), -2.0, [0.0])
    with pytest.raises(RuntimeError):
        study.finalize_mag_bulk_batch()


# ---------------------------------------------------------------------------
# get_bulk_magmom — copy safety
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_get_bulk_magmom_returns_copy():
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [2.0])
    study.finalize_mag_bulk_batch()

    magmom = study.get_bulk_magmom()
    magmom[0] = 999.0
    assert study.get_bulk_magmom()[0] != 999.0


# ---------------------------------------------------------------------------
# finalize_mag_bulk_batch — NaN guards (§2b)
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_finalize_mag_bulk_batch_raises_on_nan_energy():
    """NaN DFT energy must raise ValueError before any state is mutated."""
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), float("nan"), [0.0])
    with pytest.raises(ValueError, match="NaN energy"):
        study.finalize_mag_bulk_batch()


@pytest.mark.mini
def test_finalize_mag_bulk_batch_raises_on_nan_magmom():
    """NaN magnetic moments must raise ValueError before any state is mutated."""
    study, atoms = _make_study()
    study.store_mag_bulk_candidate(0, atoms.copy(), -1.0, [float("nan")])
    with pytest.raises(ValueError, match="NaN magnetic moments"):
        study.finalize_mag_bulk_batch()
