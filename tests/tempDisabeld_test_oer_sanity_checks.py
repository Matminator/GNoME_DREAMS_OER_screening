import numpy as np

from ase import Atoms

from gnome_dreams_oer_screening import (
    OER_adsorption_site_study,
    are_all_atoms_connected,
    is_adsorbate_near_site,
    is_OH_adsorbate_intact,
)


def _build_simple_on_top_study() -> OER_adsorption_site_study:
    init_surface = Atoms(
        "Pt",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    init_surface[0].tag = 3

    return OER_adsorption_site_study(
        init_surface=init_surface,
        init_add_position=np.array([0.0, 0.0, 1.8]),
        site_index=0,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        init=("on-top", -1.0),
    )


def test_are_all_atoms_connected_detects_desorbed_atom():
    connected_atoms = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    assert are_all_atoms_connected(connected_atoms)

    desorbed_atoms = connected_atoms.copy()
    desorbed_atoms[1].position = np.array([0.0, 0.0, 10.0])
    assert not are_all_atoms_connected(desorbed_atoms)


def test_is_adsorbate_near_site_detects_drift():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()

    assert is_adsorbate_near_site(study, O_atoms, max_site_drift=0.2)

    drifted_O_atoms = O_atoms.copy()
    drifted_O_atoms[1].position += np.array([0.0, 0.0, 2.0])
    assert not is_adsorbate_near_site(
        study, drifted_O_atoms, max_site_drift=1.0
    )


def test_is_OH_adsorbate_intact_detects_OH_drift_apart():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()
    study.set_relaxed_O_calc(O_atoms, energy=-2.0)

    OH_atoms = study.get_OH_adsorbed_atoms()[0]
    assert is_OH_adsorbate_intact(
        study, OH_atoms, min_OH_distance=0.6, max_OH_distance=1.5
    )

    separated_OH_atoms = OH_atoms.copy()
    H_index = [
        i
        for i, atom in enumerate(separated_OH_atoms)
        if atom.symbol == "H" and atom.tag == 4
    ][0]
    separated_OH_atoms[H_index].position += np.array([0.0, 0.0, 2.0])

    assert not is_OH_adsorbate_intact(
        study, separated_OH_atoms, min_OH_distance=0.6,
        max_OH_distance=1.5
    )
