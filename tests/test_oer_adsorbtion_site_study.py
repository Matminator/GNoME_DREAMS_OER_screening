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
    study.set_relaxed_O_calc(O_atoms, energy=-2.0)

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
    study = _build_simple_lattice_O_study()

    relaxed_atoms = Atoms(
        "PtOO",
        positions=[
            [0.0, 0.0, 0.0],   # surface metal
            [0.0, 0.0, 1.8],   # lattice O (target, tag=3)
            [0.8, 0.0, 2.8],   # another O to ensure selection by tag
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    relaxed_atoms[1].tag = 3
    relaxed_atoms[2].tag = 4
    study.set_relaxed_O_calc(relaxed_atoms, energy=-2.0)

    OH_inits = study.get_OH_adsorbed_atoms()
    assert len(OH_inits) == 3

    target_O_pos = relaxed_atoms[1].position.copy()
    other_O_pos = relaxed_atoms[2].position.copy()
    normal = study.surface_normal / np.linalg.norm(study.surface_normal)

    for OH_atoms in OH_inits:
        H_pos = OH_atoms[-1].position
        d_target = np.linalg.norm(H_pos - target_O_pos)
        d_other = np.linalg.norm(H_pos - other_O_pos)
        assert abs(d_target - 0.98) < 0.10
        assert d_other > 0.25
        assert np.dot(H_pos - target_O_pos, normal) > 0.0
