from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from gnome_dreams_oer_screening import Explog


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


@pytest.fixture()
def local_explog_dir() -> Path:
    base = Path(".pytest_tmp_job_handler")
    base.mkdir(exist_ok=True)
    path = base / f"test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


def test_candidate_schema_tracks_only_overpotential_summary_fields(
    local_explog_dir: Path,
):
    explog = Explog()
    explog.init(local_explog_dir, mode="test")

    columns = set(explog.relational_frame.candidates.df.columns)
    assert "idealOverPotential" in columns
    assert "Overpotential_from_scaling" in columns
    assert "best_G(O)" not in columns
    assert "best_G(OH)" not in columns
    assert "best_G(OOH)_from_scaling" not in columns
    assert "OOH_from_scaling" not in columns


def test_update_ideal_over_pot_populates_both_overpotential_fields(
    local_explog_dir: Path,
):
    # Best ideal overpotential should be 0.31 (from surface 1).
    surface_1 = DummySurfaceStudy(
        pd.DataFrame(
            {
                "ideal overpotential": [np.nan, 0.49, 0.31],
                "overpotential from OH-OOH scaling relation": [0.40, 0.38, 0.36],
            }
        )
    )
    # Best scaling overpotential should be 0.28 (from surface 2).
    surface_2 = DummySurfaceStudy(
        pd.DataFrame(
            {
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

    explog.job_handler._update_ideal_over_pot(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert row.idealOverPotential == pytest.approx(0.31, abs=1e-12)
    assert row.Overpotential_from_scaling == pytest.approx(0.28, abs=1e-12)
    assert surface_1.update_calls == 1
    assert surface_2.update_calls == 1


def test_update_ideal_over_pot_resets_fields_to_nan_without_valid_data(
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
    explog.relational_frame.candidates[candidate_id].idealOverPotential = 0.55
    explog.relational_frame.candidates[candidate_id].Overpotential_from_scaling = 0.66

    explog.job_handler._update_ideal_over_pot(candidate_id)

    row = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(row.idealOverPotential)
    assert pd.isna(row.Overpotential_from_scaling)
    assert surface_none.update_calls == 1
    assert surface_no_metrics.update_calls == 1
