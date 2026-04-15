import numpy as np
import pytest

from ase import Atoms

from gnome_dreams_oer_screening import OER_adsorption_site_study


def _build_simple_on_top_study() -> OER_adsorption_site_study:
    init_surface = Atoms(
        "Pt",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    init_surface[0].tag = 3  # adsorption-site marker
    init_add_position = np.array([0.0, 0.0, 1.8])

    return OER_adsorption_site_study(
        init_surface=init_surface,
        init_add_position=init_add_position,
        site_index=0,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        init=("on-top", -1.0),
    )


def _build_simple_lattice_O_study() -> OER_adsorption_site_study:
    init_surface = Atoms(
        "PtO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    init_surface[1].tag = 3  # lattice O site marker

    return OER_adsorption_site_study(
        init_surface=init_surface,
        init_add_position=np.array([0.0, 0.0, 1.8]),
        site_index=1,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        init=("lattice O", -1.0),
    )


def test_get_OH_adsorbed_atoms_requires_relaxed_O():
    study = _build_simple_on_top_study()
    with pytest.raises(RuntimeError, match="Relaxed O-adsorbed structure"):
        study.get_OH_adsorbed_atoms()


def test_get_OH_adsorbed_atoms_returns_three_distinct_initializations():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()
    study.set_relaxed_O_calc(O_atoms, energy=-2.0, require_magmom=False)

    OH_inits = study.get_OH_adsorbed_atoms()
    assert len(OH_inits) == 3

    O_ref = study.relaxed_O_atoms
    O_ads_indices = [
        i for i, atom in enumerate(O_ref)
        if atom.symbol == "O" and atom.tag == 4
    ]
    assert len(O_ads_indices) == 1
    O_ads_index = int(O_ads_indices[0])

    H_positions = []
    normal = study.surface_normal / np.linalg.norm(study.surface_normal)
    for OH_atoms in OH_inits:
        assert isinstance(OH_atoms, Atoms)
        assert len(OH_atoms) == len(O_ref) + 1

        H_atom = OH_atoms[-1]
        assert H_atom.symbol == "H"
        assert H_atom.tag == 4
        H_positions.append(H_atom.position.copy())

        d_OH = np.linalg.norm(
            H_atom.position - OH_atoms[O_ads_index].position
        )
        assert abs(d_OH - 0.98) < 0.10
        assert np.dot(H_atom.position - OH_atoms[O_ads_index].position,
                      normal) > 0.0

    pairwise_distances = []
    for i in range(len(H_positions)):
        for j in range(i + 1, len(H_positions)):
            pairwise_distances.append(
                np.linalg.norm(H_positions[i] - H_positions[j])
            )

    assert min(pairwise_distances) > 0.05


def test_get_OH_adsorbed_atoms_lattice_O_uses_tag_3():
    init_surface = Atoms(
        "PtOO",
        positions=[
            [0.0, 0.0, 0.0],   # surface metal
            [0.0, 0.0, 1.8],   # lattice O (target, tag=3)
            [0.8, 0.0, 2.8],   # another O to ensure selection by tag
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    init_surface[1].tag = 3
    init_surface[2].tag = 4

    study = OER_adsorption_site_study(
        init_surface=init_surface,
        init_add_position=np.array([0.0, 0.0, 1.8]),
        site_index=1,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        init=("lattice O", -1.0),
    )

    OH_inits = study.get_OH_adsorbed_atoms()
    assert len(OH_inits) == 3

    target_O_pos = init_surface[1].position.copy()
    other_O_pos = init_surface[2].position.copy()
    normal = study.surface_normal / np.linalg.norm(study.surface_normal)

    for OH_atoms in OH_inits:
        H_pos = OH_atoms[-1].position
        d_target = np.linalg.norm(H_pos - target_O_pos)
        d_other = np.linalg.norm(H_pos - other_O_pos)
        assert abs(d_target - 0.98) < 0.10
        assert d_other > 0.25
        assert np.dot(H_pos - target_O_pos, normal) > 0.0


def test_is_adsorbate_near_site_accepts_rigid_translation_with_default_limit():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()

    translated_atoms = O_atoms.copy()
    translated_atoms[0].position += np.array([0.9, 0.0, 0.0])
    translated_atoms[1].position += np.array([0.9, 0.0, 0.0])

    assert study.is_adsorbate_near_site(translated_atoms)


def test_is_adsorbate_near_site_rejects_adsorbate_motion_away_from_site_atom():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()

    drifted_atoms = O_atoms.copy()
    drifted_atoms[1].position += np.array([2.0, 0.0, 0.0])

    assert not study.is_adsorbate_near_site(drifted_atoms)


def test_is_adsorbate_near_site_accepts_explicit_distance_threshold():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()

    assert study.is_adsorbate_near_site(
        O_atoms,
        max_adsorbate_site_distance=1.9,
    )
    assert not study.is_adsorbate_near_site(
        O_atoms,
        max_adsorbate_site_distance=1.7,
    )


def test_is_adsorbate_near_site_is_not_applied_for_lattice_o_sites():
    study = _build_simple_lattice_O_study()
    OH_atoms = study.get_OH_adsorbed_atoms()[0]

    drifted_atoms = OH_atoms.copy()
    drifted_atoms[1].position += np.array([1.5, 0.0, 0.0])
    drifted_atoms[2].position += np.array([1.5, 0.0, 0.0])

    assert study.is_adsorbate_near_site(drifted_atoms)


# ---------------------------------------------------------------------------
# set_relaxed_O_calc magmom tests
# ---------------------------------------------------------------------------

class _MockCalc:
    """Minimal ASE calculator stub that returns pre-set magnetic moments."""
    def __init__(self, magmoms):
        self._magmoms = np.array(magmoms)
    def get_magnetic_moments(self, _atoms=None):
        return self._magmoms


def test_set_relaxed_O_calc_raises_without_magmom_by_default():
    """Default require_magmom=True raises ValueError when no calculator is attached."""
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()  # no calculator attached
    with pytest.raises(ValueError, match="require_magmom"):
        study.set_relaxed_O_calc(O_atoms, energy=-2.0)


def test_set_relaxed_O_calc_stores_magmom_when_calculator_attached():
    """Magmoms from a mock calculator are stored and returned via get_O_calc_magmom()."""
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()
    n = len(O_atoms)
    fake_magmoms = [0.3 * (i % 4 + 1) for i in range(n)]  # all non-zero
    O_atoms.calc = _MockCalc(fake_magmoms)
    ok, msg = study.set_relaxed_O_calc(O_atoms, energy=-2.0)
    assert ok is True
    result = study.get_O_calc_magmom()
    assert result is not None
    assert len(result) == n
    assert result == pytest.approx(fake_magmoms, abs=1e-6)
