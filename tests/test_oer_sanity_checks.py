import numpy as np

from ase import Atoms

from gnome_dreams_oer_screening.oer.oer_adsorption_site_study import (
    OER_adsorption_site_study
) 

from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    are_all_atoms_connected
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

    assert study.is_adsorbate_near_site(O_atoms)

    drifted_O_atoms = O_atoms.copy()
    drifted_O_atoms[1].position += np.array([0.0, 0.0, 2.0])
    assert not study.is_adsorbate_near_site(drifted_O_atoms)


def test_is_OH_adsorbate_intact_detects_OH_drift_apart():
    study = _build_simple_on_top_study()
    O_atoms = study.get_atoms_for_O_calc()
    study.set_relaxed_O_calc(
        O_atoms, 
        energy=-2.0, 
        require_magmom = False
        )

    OH_atoms = study.get_OH_adsorbed_atoms()[0]
    assert study.is_OH_adsorbate_intact(
        OH_atoms, max_OH_distance=1.5
    )

    separated_OH_atoms = OH_atoms.copy()
    H_index = [
        i
        for i, atom in enumerate(separated_OH_atoms)
        if atom.symbol == "H" and atom.tag == 4
    ][0]
    separated_OH_atoms[H_index].position += np.array([0.0, 0.0, 2.0])

    assert not study.is_OH_adsorbate_intact(
       separated_OH_atoms, max_OH_distance=1.5
    )
