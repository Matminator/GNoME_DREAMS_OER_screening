import copy
import pathlib

import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from gnome_dreams_oer_screening import OER_catalyst_study
from gnome_dreams_oer_screening.oer.oer_study_utils import (
    eval_stoichiometry_diff,
)


@pytest.fixture(scope="module")
def prepared_surface_and_study():
    cif_path = (
        pathlib.Path(__file__).parent / "../notebooks/TiO2_rutile.cif"
    ).resolve()
    atoms = read(cif_path, format="cif")

    study = OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )

    study.set_relaxed_structure(study.get_init_atoms())
    study.predict_most_likely_surfaces(method="O-coordination")

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    set_ok = surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.13500000000005,
    )
    assert set_ok is True

    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=False
    )

    return study, surface_study


def _get_on_top_site_index(df: pd.DataFrame) -> int:
    on_top_rows = df[df["site type"] == "on-top"]
    if on_top_rows.empty:
        pytest.skip("No on-top adsorption sites found in this fixture.")
    return int(on_top_rows["Site index"].iloc[0])


def test_adsorption_site_dataframe_uses_new_energy_columns(
    prepared_surface_and_study,
):
    _, surface_study = prepared_surface_and_study

    df = surface_study.get_adsorption_sites_df()
    assert isinstance(df, pd.DataFrame)

    expected_cols = [
        "G(O)",
        "G(O) deviation",
        "G(OH)",
        "G(OH) deviation",
        "G(OOH) from scaling relation",
        "G(OOH, scaling) deviation",
        "ideal overpotential",
        "overpotential from OH-OOH scaling relation",
    ]
    for col in expected_cols:
        assert col in df.columns

    removed_cols = [
        "G_approx(O)",
        "G_approx(OH)",
        "G_approx(OOH)",
    ]
    for col in removed_cols:
        assert col not in df.columns


def test_update_adsorption_energies_populates_g_and_scaling_columns(
    prepared_surface_and_study,
):
    study, surface_study = prepared_surface_and_study
    study = copy.deepcopy(study)
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    site_index = _get_on_top_site_index(df)

    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    constants = study.get_gas_and_energy_corrections()
    target_g_o = 2.46
    target_g_oh = 1.60

    e_o = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )
    e_oh = (
        clean_energy
        + target_g_oh
        - constants["OH_ad_correction"]
        - (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )

    ok_o, msg_o = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=e_o,
    )
    assert ok_o is True
    assert msg_o is None

    ok_oh, msg_oh = ad_study.set_relaxed_OH_calc(
        ad_study.get_OH_adsorbed_atoms()[0],
        energies=[e_oh],
    )
    assert ok_oh is True
    assert msg_oh is None

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert row["G(OH)"] == pytest.approx(target_g_oh, abs=1e-8)
    assert row["G(OH) deviation"] == pytest.approx(0.37, abs=1e-8)
    assert not pd.isna(row["ideal overpotential"])
    assert not pd.isna(row["G(OOH) from scaling relation"])
    assert row["G(OOH, scaling) deviation"] == pytest.approx(1.11, abs=1e-8)
    assert not pd.isna(row["overpotential from OH-OOH scaling relation"])


def test_update_adsorption_energies_with_only_o_populates_only_o_deviation(
    prepared_surface_and_study,
):
    study, surface_study = prepared_surface_and_study
    study = copy.deepcopy(study)
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    site_index = _get_on_top_site_index(df)

    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    constants = study.get_gas_and_energy_corrections()
    target_g_o = 2.61
    expected_deviation = 0.15

    e_o = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )

    ok_o, msg_o = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=e_o,
    )
    assert ok_o is True
    assert msg_o is None

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert row["G(O) deviation"] == pytest.approx(
        expected_deviation, abs=1e-8
    )
    assert pd.isna(row["G(OH)"])
    assert pd.isna(row["G(OH) deviation"])
    assert pd.isna(row["G(OOH) from scaling relation"])
    assert pd.isna(row["G(OOH, scaling) deviation"])
    assert pd.isna(row["ideal overpotential"])
    assert pd.isna(row["overpotential from OH-OOH scaling relation"])


def test_update_adsorption_energies_hides_oh_metrics_until_batch_finalized(
    prepared_surface_and_study,
):
    study, surface_study = prepared_surface_and_study
    study = copy.deepcopy(study)
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    site_index = _get_on_top_site_index(df)

    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    constants = study.get_gas_and_energy_corrections()
    target_g_o = 2.46
    target_g_oh = 1.60

    e_o = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )
    e_oh = (
        clean_energy
        + target_g_oh
        - constants["OH_ad_correction"]
        - (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )

    ok_o, msg_o = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=e_o,
    )
    assert ok_o is True
    assert msg_o is None

    ok_oh, msg_oh = ad_study.set_relaxed_OH_calc(
        ad_study.get_OH_adsorbed_atoms()[0],
        energies=[e_oh],
    )
    assert ok_oh is True
    assert msg_oh is None

    ad_study.set_oh_batch_finalized(False)
    surface_study.update_adsorption_energies()
    partial_row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert partial_row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert partial_row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert pd.isna(partial_row["G(OH)"])
    assert pd.isna(partial_row["G(OH) deviation"])
    assert pd.isna(partial_row["G(OOH) from scaling relation"])
    assert pd.isna(partial_row["G(OOH, scaling) deviation"])
    assert pd.isna(partial_row["ideal overpotential"])
    assert pd.isna(partial_row["overpotential from OH-OOH scaling relation"])

    ad_study.set_oh_batch_finalized(True)
    surface_study.update_adsorption_energies()
    final_row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert final_row["G(OH)"] == pytest.approx(target_g_oh, abs=1e-8)
    assert final_row["G(OH) deviation"] == pytest.approx(0.37, abs=1e-8)
    assert pd.notna(final_row["G(OOH) from scaling relation"])
    assert pd.notna(final_row["G(OOH, scaling) deviation"])
    assert pd.notna(final_row["ideal overpotential"])
    assert pd.notna(final_row["overpotential from OH-OOH scaling relation"])


def test_update_adsorption_energies_clears_stale_derived_columns_when_oh_missing(
    prepared_surface_and_study,
):
    study, surface_study = prepared_surface_and_study
    study = copy.deepcopy(study)
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    site_index = _get_on_top_site_index(df)

    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    constants = study.get_gas_and_energy_corrections()
    target_g_o = 2.46
    target_g_oh = 1.60

    e_o = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )
    e_oh = (
        clean_energy
        + target_g_oh
        - constants["OH_ad_correction"]
        - (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )

    ok_o, _ = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=e_o,
    )
    ok_oh, _ = ad_study.set_relaxed_OH_calc(
        ad_study.get_OH_adsorbed_atoms()[0],
        energies=[e_oh],
    )
    assert ok_o is True
    assert ok_oh is True

    surface_study.update_adsorption_energies()
    populated_row = surface_study.get_adsorption_sites_df().loc[site_index]
    assert pd.notna(populated_row["G(OOH, scaling) deviation"])
    assert pd.notna(populated_row["ideal overpotential"])

    # Simulate the OH result becoming unavailable and ensure a fresh
    # update clears all derived OH/OOH/overpotential fields.
    ad_study.relaxed_OH_atoms = None
    ad_study.accepted_relaxed_OH_pairs = []
    ad_study.energies_dict["OH_adsorbed_energy"] = None

    surface_study.update_adsorption_energies()
    cleared_row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert cleared_row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert cleared_row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert pd.isna(cleared_row["G(OH)"])
    assert pd.isna(cleared_row["G(OH) deviation"])
    assert pd.isna(cleared_row["G(OOH) from scaling relation"])
    assert pd.isna(cleared_row["G(OOH, scaling) deviation"])
    assert pd.isna(cleared_row["ideal overpotential"])
    assert pd.isna(cleared_row["overpotential from OH-OOH scaling relation"])


def test_set_relaxed_oh_calc_keeps_lowest_energy_across_calls(
    prepared_surface_and_study,
):
    _, surface_study = prepared_surface_and_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    site_index = _get_on_top_site_index(df)

    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    ok_o, _ = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=-280.0,
    )
    assert ok_o is True

    oh_candidates = ad_study.get_OH_adsorbed_atoms()
    ok_1, msg_1 = ad_study.set_relaxed_OH_calc(
        oh_candidates[0], energies=[-280.0]
    )
    ok_2, msg_2 = ad_study.set_relaxed_OH_calc(
        oh_candidates[1], energies=[-281.0]
    )

    assert ok_1 is True and msg_1 is None
    assert ok_2 is True and msg_2 is None
    assert ad_study.get_adsorption_site_energies()[
        "OH_adsorbed_energy"
    ] == pytest.approx(-281.0, abs=1e-12)
    assert len(ad_study.accepted_relaxed_OH_pairs) >= 2


def test_eval_stoichiometry_diff_handles_missing_species_flag():
    config_1 = Atoms("TiO2")
    config_2 = Atoms("Ti")

    assert not eval_stoichiometry_diff(
        config_1,
        config_2,
        all_species_present=True,
        max_diff=1.0,
    )

    assert eval_stoichiometry_diff(
        config_1,
        config_2,
        all_species_present=False,
        max_diff=1.0,
    )
