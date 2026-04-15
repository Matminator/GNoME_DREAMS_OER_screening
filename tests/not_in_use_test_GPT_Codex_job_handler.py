from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from ase.io import read

from gnome_dreams_oer_screening import Explog, OER_catalyst_study
import gnome_dreams_oer_screening.vasp.vasp_calculation as \
    vasp_calculation_module


class DummySurfaceStudy:
    def __init__(self, df: pd.DataFrame | None):
        self._df = df
        self.update_calls = 0

    def update_adsorption_energies(self):
        self.update_calls += 1

    def get_adsorption_sites_df(self):
        return self._df


class DummyStudy:
    def __init__(self, surfaces: dict[int, DummySurfaceStudy]):
        self._surfaces = surfaces

    def get_surface_studies(self):
        return self._surfaces


class UndeterminedSurfaceStudy:
    def __init__(self):
        self.update_calls = 0

    def update_adsorption_energies(self):
        self.update_calls += 1
        raise AssertionError(
            "update_adsorption_energies should not be called when "
            "adsorption sites have not been determined."
        )

    def get_adsorption_sites_df(self):
        return None


@pytest.fixture()
def local_explog_dir() -> Path:
    base = Path(".pytest_tmp_job_handler")
    base.mkdir(exist_ok=True)
    path = base / f"test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


def test_candidate_schema_tracks_summary_metric_fields(
    local_explog_dir: Path,
):
    explog = Explog()
    explog.init(local_explog_dir, mode="test")

    columns = set(explog.relational_frame.candidates.df.columns)
    assert "G(O) deviation" in columns
    assert "idealOverPotential" in columns
    assert "Overpotential_from_scaling" in columns
    assert "G(O) Deviation from optimal" not in columns
    assert "best_G(O)" not in columns
    assert "best_G(OH)" not in columns
    assert "best_G(OOH)_from_scaling" not in columns
    assert "OOH_from_scaling" not in columns


def test_update_candidate_oer_metrics_populates_summary_fields(
    local_explog_dir: Path,
):
    # Best ideal overpotential should be 0.31 (from surface 1).
    surface_1 = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.22, 0.06, 0.14],
                "ideal overpotential": [np.nan, 0.49, 0.31],
                "overpotential from OH-OOH scaling relation": [0.40, 0.38, 0.36],
            }
        )
    )
    # Best scaling overpotential should be 0.28 (from surface 2).
    surface_2 = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.19, 0.08],
                "ideal overpotential": [0.72, 0.64],
                "overpotential from OH-OOH scaling relation": [0.41, 0.28],
            }
        )
    )
    dummy_study = DummyStudy({0: surface_1, 1: surface_2})

    explog = Explog()
    explog.init(local_explog_dir, mode="test")
    candidate_id = "cand_overpot_fields"
    explog.add_candidate(candidate_id=candidate_id, study_obj=dummy_study)

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert row.df["G(O) deviation"].iloc[0] == pytest.approx(0.06, abs=1e-12)
    assert row.idealOverPotential == pytest.approx(0.31, abs=1e-12)
    assert row.Overpotential_from_scaling == pytest.approx(0.28, abs=1e-12)
    assert surface_1.update_calls == 1
    assert surface_2.update_calls == 1


def test_update_candidate_oer_metrics_resets_fields_to_nan_without_valid_data(
    local_explog_dir: Path,
):
    surface_none = DummySurfaceStudy(None)
    surface_no_metrics = DummySurfaceStudy(pd.DataFrame({"x": [1, 2]}))
    dummy_study = DummyStudy({0: surface_none, 1: surface_no_metrics})

    explog = Explog()
    explog.init(local_explog_dir, mode="test")
    candidate_id = "cand_nan_reset"
    explog.add_candidate(candidate_id=candidate_id, study_obj=dummy_study)

    # Seed values to verify reset behavior.
    explog.relational_frame.candidates.set_value(
        candidate_id, "G(O) deviation", 0.12
    )
    explog.relational_frame.candidates[candidate_id].idealOverPotential = 0.55
    explog.relational_frame.candidates[candidate_id].Overpotential_from_scaling = 0.66

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(row.df["G(O) deviation"].iloc[0])
    assert pd.isna(row.idealOverPotential)
    assert pd.isna(row.Overpotential_from_scaling)
    assert surface_none.update_calls == 0
    assert surface_no_metrics.update_calls == 1


def test_update_candidate_oer_metrics_populates_o_deviation_without_overpotentials(
    local_explog_dir: Path,
):
    surface_1 = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.18, 0.04],
                "ideal overpotential": [np.nan, np.nan],
                "overpotential from OH-OOH scaling relation": [np.nan, np.nan],
            }
        )
    )
    surface_2 = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.11, 0.07],
                "ideal overpotential": [np.nan, np.nan],
                "overpotential from OH-OOH scaling relation": [np.nan, np.nan],
            }
        )
    )
    dummy_study = DummyStudy({0: surface_1, 1: surface_2})

    explog = Explog()
    explog.init(local_explog_dir, mode="test")
    candidate_id = "cand_o_only_metric"
    explog.add_candidate(candidate_id=candidate_id, study_obj=dummy_study)

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert row.df["G(O) deviation"].iloc[0] == pytest.approx(0.04, abs=1e-12)
    assert pd.isna(row.idealOverPotential)
    assert pd.isna(row.Overpotential_from_scaling)


def test_update_candidate_oer_metrics_aggregates_o_deviation_independently(
    local_explog_dir: Path,
):
    surface_best_o = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.21, 0.03],
                "ideal overpotential": [np.nan, np.nan],
                "overpotential from OH-OOH scaling relation": [np.nan, np.nan],
            }
        )
    )
    surface_best_overpot = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.12, 0.09],
                "ideal overpotential": [0.44, 0.31],
                "overpotential from OH-OOH scaling relation": [0.37, 0.28],
            }
        )
    )
    dummy_study = DummyStudy({0: surface_best_o, 1: surface_best_overpot})

    explog = Explog()
    explog.init(local_explog_dir, mode="test")
    candidate_id = "cand_independent_aggregation"
    explog.add_candidate(candidate_id=candidate_id, study_obj=dummy_study)

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert row.df["G(O) deviation"].iloc[0] == pytest.approx(0.03, abs=1e-12)
    assert row.idealOverPotential == pytest.approx(0.31, abs=1e-12)
    assert row.Overpotential_from_scaling == pytest.approx(0.28, abs=1e-12)


def test_update_candidate_oer_metrics_skips_surfaces_without_determined_adsorption_sites(
    local_explog_dir: Path,
):
    skipped_surface = UndeterminedSurfaceStudy()
    valid_surface = DummySurfaceStudy(
        pd.DataFrame(
            {
                "G(O) deviation": [0.11, 0.04],
                "ideal overpotential": [np.nan, 0.29],
                "overpotential from OH-OOH scaling relation": [0.41, 0.33],
            }
        )
    )
    dummy_study = DummyStudy({0: skipped_surface, 1: valid_surface})

    explog = Explog()
    explog.init(local_explog_dir, mode="test")
    candidate_id = "cand_skip_undetermined_surface"
    explog.add_candidate(candidate_id=candidate_id, study_obj=dummy_study)

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert row.df["G(O) deviation"].iloc[0] == pytest.approx(0.04, abs=1e-12)
    assert row.idealOverPotential == pytest.approx(0.29, abs=1e-12)
    assert row.Overpotential_from_scaling == pytest.approx(0.33, abs=1e-12)
    assert skipped_surface.update_calls == 0
    assert valid_surface.update_calls == 1


def test_update_log_does_not_leave_rows_updating_when_other_surface_lacks_adsorption_sites(
    local_explog_dir: Path,
):
    cif_path = (
        Path(__file__).resolve().parent / "../notebooks/TiO2_rutile.cif"
    ).resolve()
    atoms = read(cif_path, format="cif")

    explog = Explog()
    explog.init(local_explog_dir, mode="test")

    candidate_id = "cand_multi_surface_guard"
    study = OER_catalyst_study(
        atoms.copy(),
        H2O_gas_free_energy=-14.217,
        H2_gas_free_energy=-6.77,
        use_init_as_relaxed_bulk=True,
    )
    study.predict_most_likely_surfaces(max_miller=1)
    termination_indices = list(study.get_terminations_dict().keys())
    if len(termination_indices) < 2:
        pytest.skip("Need at least two terminations for this regression test.")

    valid_term = int(termination_indices[0])
    undetermined_term = int(termination_indices[1])

    for offset, term_idx in enumerate((valid_term, undetermined_term)):
        study.initialize_oer_surface_study(term_idx)
        surface_study = study.get_surface_studies()[term_idx]
        set_ok = surface_study.set_relaxed_base_surface(
            surface_study.get_non_relaxed_surface(),
            energy=-278.135 - offset,
        )
        assert set_ok is True

    valid_surface = study.get_surface_studies()[valid_term]
    valid_surface.determine_adsorption_sites(
        only_reduced_coord_O_sites=True
    )
    site_index = int(
        valid_surface.get_adsorption_sites_df()["Site index"].iloc[0]
    )
    valid_surface.initialize_adsorption_site_study(site_index)

    explog.add_candidate(candidate_id=candidate_id, study_obj=study)
    o_process_id = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="O_adsorption",
        termination_index=valid_term,
        site_index=site_index,
    )
    explog.submit_process(o_process_id)

    trailing_candidate_id = "cand_trailing_bulk"
    trailing_study = OER_catalyst_study(
        atoms.copy(),
        H2O_gas_free_energy=-14.217,
        H2_gas_free_energy=-6.77,
    )
    explog.add_candidate(
        candidate_id=trailing_candidate_id,
        study_obj=trailing_study,
    )
    bulk_process_id = explog.add_process(
        candidate_id=trailing_candidate_id,
        calculation_type="bulk_relaxation",
    )
    explog.submit_process(bulk_process_id)

    updates = explog.update_log()

    assert updates[o_process_id] == "completed"
    assert updates[bulk_process_id] == "completed"

    pdf = explog.relational_frame.processes.df.copy()
    assert set(pdf["status"].tolist()) == {"completed"}


def test_refresh_oh_batch_finalization_state_waits_for_all_terminal_statuses(
    local_explog_dir: Path,
):
    cif_path = (
        Path(__file__).resolve().parent / "../notebooks/TiO2_rutile.cif"
    ).resolve()
    atoms = read(cif_path, format="cif")

    explog = Explog()
    explog.init(local_explog_dir, mode="test")

    candidate_id = "cand_oh_batch_status_gate"
    study = OER_catalyst_study(
        atoms.copy(),
        H2O_gas_free_energy=-14.217,
        H2_gas_free_energy=-6.77,
        use_init_as_relaxed_bulk=True,
    )
    study.set_relaxed_structure(study.get_init_atoms())
    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)

    surface_study = study.get_surface_studies()[0]
    set_ok = surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.135,
    )
    assert set_ok is True

    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=True
    )
    site_index = int(
        surface_study.get_adsorption_sites_df()["Site index"].iloc[0]
    )
    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    ok_o, msg_o = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=-280.0,
    )
    assert ok_o is True
    assert msg_o is None

    explog.add_candidate(candidate_id=candidate_id, study_obj=study)
    process_ids = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="OH_adsorption",
        termination_index=0,
        site_index=site_index,
    )
    assert isinstance(process_ids, list)
    assert len(process_ids) == 3
    assert ad_study.is_oh_batch_finalized() is False

    pdf = explog.relational_frame.processes.df
    pdf.loc[pdf["process_id"] == process_ids[0], "status"] = "completed"
    pdf.loc[pdf["process_id"] == process_ids[1], "status"] = "failed"
    pdf.loc[pdf["process_id"] == process_ids[2], "status"] = "pending"

    all_terminal = explog.job_handler._refresh_oh_batch_finalization_state(
        candidate_id,
        0,
        site_index,
    )
    assert all_terminal is False
    assert ad_study.is_oh_batch_finalized() is False

    pdf.loc[
        pdf["process_id"] == process_ids[2],
        "status",
    ] = "unrecoverable: mock rejection"

    all_terminal = explog.job_handler._refresh_oh_batch_finalization_state(
        candidate_id,
        0,
        site_index,
    )
    assert all_terminal is True
    assert ad_study.is_oh_batch_finalized() is True


def test_enter_process_rejects_second_oh_batch_after_first_was_submitted(
    local_explog_dir: Path,
):
    cif_path = (
        Path(__file__).resolve().parent / "../notebooks/TiO2_rutile.cif"
    ).resolve()
    atoms = read(cif_path, format="cif")

    explog = Explog()
    explog.init(
        local_explog_dir,
        mode="test",
        reject_if_failed_exists=True,
    )

    candidate_id = "cand_no_second_oh_batch"
    study = OER_catalyst_study(
        atoms.copy(),
        H2O_gas_free_energy=-14.217,
        H2_gas_free_energy=-6.77,
        use_init_as_relaxed_bulk=True,
    )
    study.set_relaxed_structure(study.get_init_atoms())
    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)

    surface_study = study.get_surface_studies()[0]
    set_ok = surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.135,
    )
    assert set_ok is True

    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=True
    )
    site_index = int(
        surface_study.get_adsorption_sites_df()["Site index"].iloc[0]
    )
    surface_study.initialize_adsorption_site_study(site_index)
    ad_study = surface_study.get_adsorption_site_studies_dict()[site_index]

    ok_o, msg_o = ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(),
        energy=-280.0,
    )
    assert ok_o is True
    assert msg_o is None

    explog.add_candidate(candidate_id=candidate_id, study_obj=study)
    process_ids = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="OH_adsorption",
        termination_index=0,
        site_index=site_index,
    )
    assert isinstance(process_ids, list)
    assert len(process_ids) == 3

    for process_id in process_ids:
        explog.submit_process(process_id)

    pdf = explog.relational_frame.processes.df
    pdf.loc[pdf["process_id"].isin(process_ids), "status"] = "failed"

    with pytest.raises(RuntimeError, match="type OH_adsorption"):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="OH_adsorption",
            termination_index=0,
            site_index=site_index,
        )


def test_test_mode_workflow_yields_near_ideal_oh_energy_when_noise_is_zero(
    local_explog_dir: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        vasp_calculation_module.np.random,
        "normal",
        lambda loc=0, scale=1: 0.0,
    )

    cif_path = (
        Path(__file__).resolve().parent / "../notebooks/TiO2_rutile.cif"
    ).resolve()
    atoms = read(cif_path, format="cif")

    explog = Explog()
    explog.init(local_explog_dir, mode="test")

    candidate_id = "cand_test_mode_oh"
    study = OER_catalyst_study(
        atoms.copy(),
        H2O_gas_free_energy=-14.217,
        H2_gas_free_energy=-6.77,
    )
    explog.add_candidate(candidate_id=candidate_id, study_obj=study)

    p_bulk = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="bulk_relaxation",
    )
    explog.submit_process(p_bulk)
    updates = explog.update_log()
    assert updates[p_bulk] == "completed"

    candidate_study = explog.relational_frame.candidates[candidate_id].study_obj
    candidate_study.predict_most_likely_surfaces(max_miller=1)
    candidate_study.initialize_oer_surface_study(0)

    p_surface = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="surface_relaxation",
        termination_index=0,
    )
    explog.submit_process(p_surface)
    updates = explog.update_log()
    assert updates[p_surface] == "completed"

    surface_study = candidate_study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=True
    )
    site_index = int(
        surface_study.get_adsorption_sites_df()["Site index"].iloc[0]
    )
    surface_study.initialize_adsorption_site_study(site_index)

    p_o = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="O_adsorption",
        termination_index=0,
        site_index=site_index,
    )
    explog.submit_process(p_o)
    updates = explog.update_log()
    assert updates[p_o] == "completed"
    p_o_row = explog.relational_frame.processes[p_o].df.iloc[0]
    assert p_o_row["G(O)"] == pytest.approx(2.46, abs=1e-8)
    assert p_o_row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert pd.isna(p_o_row["G(OH)"])
    assert pd.isna(p_o_row["G(OH) deviation"])
    assert pd.isna(p_o_row["ideal overpotential"])
    assert pd.isna(p_o_row["overpotential from OH-OOH scaling relation"])

    p_oh = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="OH_adsorption",
        termination_index=0,
        site_index=site_index,
    )
    assert isinstance(p_oh, list)
    assert len(p_oh) == 3
    for process_id in p_oh:
        explog.submit_process(process_id)

    updates = explog.update_log()
    assert set(updates.keys()) == set(p_oh)
    assert set(updates.values()) == {"completed"}
    for p_id in p_oh:
        p_row = explog.relational_frame.processes[p_id].df.iloc[0]
        assert p_row["G(O)"] == pytest.approx(2.46, abs=1e-8)
        assert p_row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
        assert p_row["G(OH)"] == pytest.approx(1.23, abs=1e-8)
        assert p_row["G(OH) deviation"] == pytest.approx(0.0, abs=1e-8)
        assert p_row["ideal overpotential"] == pytest.approx(0.0,
                                                             abs=1e-8)
        assert p_row["overpotential from OH-OOH scaling relation"] == \
            pytest.approx(0.74, abs=1e-8)

    row = surface_study.get_adsorption_sites_df().loc[site_index]
    candidate_row = explog.relational_frame.candidates[candidate_id]

    assert row["G(O)"] == pytest.approx(2.46, abs=1e-8)
    assert row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert row["G(OH)"] == pytest.approx(1.23, abs=1e-8)
    assert row["G(OH) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert row["ideal overpotential"] == pytest.approx(0.0, abs=1e-8)
    assert row["overpotential from OH-OOH scaling relation"] == \
        pytest.approx(0.74, abs=1e-8)

    assert candidate_row.df["G(O) deviation"].iloc[0] == \
        pytest.approx(0.0, abs=1e-8)
    assert candidate_row.idealOverPotential == pytest.approx(0.0, abs=1e-8)
    assert candidate_row.Overpotential_from_scaling == pytest.approx(
        0.74, abs=1e-8
    )
