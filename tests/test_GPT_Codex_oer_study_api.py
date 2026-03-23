import pytest
from ase import Atoms

from gnome_dreams_oer_screening import OER_catalyst_study


def _make_atoms() -> Atoms:
    return Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)


def test_gas_and_energy_corrections_expose_only_active_keys():
    study = OER_catalyst_study(
        _make_atoms(),
        H2O_gas_free_energy=-4.0,
        H2_gas_free_energy=-2.0,
    )

    energies = study.get_gas_and_energy_corrections()

    assert set(energies.keys()) == {
        "H2O_gas_free_energy",
        "H2_gas_free_energy",
        "O_ad_correction",
        "OH_ad_correction",
        "OOH_ad_correction",
    }
    assert "O2_gas_free_energy" not in energies


def test_set_energies_updates_all_active_energy_and_correction_fields():
    study = OER_catalyst_study(
        _make_atoms(),
        H2O_gas_free_energy=-4.0,
        H2_gas_free_energy=-2.0,
    )

    study.set_energies(
        H2O_gas_free_energy=-10.0,
        H2_gas_free_energy=-5.0,
        O_ad_correction=0.11,
        OH_ad_correction=0.22,
        OOH_ad_correction=0.33,
    )

    energies = study.get_gas_and_energy_corrections()

    assert energies["H2O_gas_free_energy"] == pytest.approx(-10.0, abs=1e-12)
    assert energies["H2_gas_free_energy"] == pytest.approx(-5.0, abs=1e-12)
    assert energies["O_ad_correction"] == pytest.approx(0.11, abs=1e-12)
    assert energies["OH_ad_correction"] == pytest.approx(0.22, abs=1e-12)
    assert energies["OOH_ad_correction"] == pytest.approx(0.33, abs=1e-12)


def test_constructor_rejects_removed_o2_gas_free_energy_keyword():
    with pytest.raises(TypeError, match="O2_gas_free_energy"):
        OER_catalyst_study(
            _make_atoms(),
            H2O_gas_free_energy=-4.0,
            H2_gas_free_energy=-2.0,
            O2_gas_free_energy=-1.0,
        )


def test_set_energies_rejects_removed_o2_gas_free_energy_keyword():
    study = OER_catalyst_study(
        _make_atoms(),
        H2O_gas_free_energy=-4.0,
        H2_gas_free_energy=-2.0,
    )

    with pytest.raises(TypeError, match="O2_gas_free_energy"):
        study.set_energies(O2_gas_free_energy=-1.0)
