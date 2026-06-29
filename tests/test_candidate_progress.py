# Tests for the per-candidate progress projection added to the candidates table
# (Job_handler._recompute_candidate_progress, run on every update_log()).
#
# What is covered:
#   - the per-stage started/finalized counters track a real bulk->surface->O->OH
#     pipeline (in "test" mode, where jobs complete instantly);
#   - the stage-gated <NA> semantics: a counter is <NA> until the candidate is
#     eligible for that stage, so "n_X_started == 0" means "ready for stage X";
#   - counts are per LOGICAL UNIT: bulk collapses to 0/1 (despite N magnetic
#     sub-jobs) and OH is counted per adsorption site (despite 3 sub-jobs/site);
#   - the active/failed `state`: all-bulk-failed and "no valid surface" both fail
#     the candidate, surface/adsorbate failures do not, and `failed` latches.
#
# Failure / in-flight scenarios are built by INJECTING process rows with chosen
# terminal/running statuses directly into the processes table. This is safe in
# "test" mode because update_log() only advances rows whose status is 'pending';
# any other status is left untouched and flows straight into the recompute.

from pathlib import Path

import pandas as pd
import pytest
from ase.io import read

from gnome_dreams_oer_screening import Explog, OER_catalyst_study


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tio2_rutile_atoms():
    dir_path = Path(__file__).resolve().parent
    path = dir_path / "../notebooks/TiO2_rutile.cif"
    return read(path.resolve(), format="cif")


def _new_study(atoms):
    return OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )


def _val(explog, candidate_id, col):
    """Return one candidate's value for a column (nullable-aware scalar)."""
    cdf = explog.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == candidate_id, col].iloc[0]


def _inject_process(explog, candidate_id, job_type, status,
                    termination_index=None, site_index=None):
    """Append a process row directly (bypassing submit) with a chosen status,
    so terminal / running scenarios can be built deterministically."""
    table = explog.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row(
        {
            "process_id": pid,
            "candidate_id": candidate_id,
            "job_type": job_type,
            "status": status,
            "termination_index": termination_index,
            "site_index": site_index,
        },
        allow_update=False,
    )
    return pid


def _set_status(explog, process_id, status):
    explog.relational_frame.processes.set_value(process_id, "status", status)


# ---------------------------------------------------------------------------
# Integration: counters track the real pipeline, stage by stage
# ---------------------------------------------------------------------------

def test_progress_columns_track_pipeline_stage(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "tio2"
    explog.add_candidate(
        candidate_id=cid,
        study_obj=_new_study(tio2_rutile_atoms),
        reason_or_hypothesis="pipeline progress",
    )

    # Fresh candidate: active, ready for bulk, downstream stages <NA>.
    assert _val(explog, cid, "state") == "active"
    assert _val(explog, cid, "n_bulk_started") == 0
    assert _val(explog, cid, "n_bulk_finalized") == 0
    for col in ("n_surface_started", "n_O_started", "n_OH_started"):
        assert pd.isna(_val(explog, cid, col))

    # --- bulk ---
    bulk_ids = explog.add_process(candidate_id=cid,
                                  calculation_type="bulk_relaxation")
    for pid in bulk_ids:
        explog.submit_process(pid)
    explog.update_log()
    assert _val(explog, cid, "state") == "active"
    assert _val(explog, cid, "n_bulk_started") == 1     # one logical unit
    assert _val(explog, cid, "n_bulk_finalized") == 1
    assert _val(explog, cid, "n_surface_started") == 0  # eligible -> 0, not <NA>
    assert _val(explog, cid, "n_surface_finalized") == 0
    assert pd.isna(_val(explog, cid, "n_O_started"))
    assert pd.isna(_val(explog, cid, "n_OH_started"))

    # --- surface ---
    study = explog.relational_frame.candidates[cid].study_obj
    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)
    p_surf = explog.add_process(candidate_id=cid,
                                calculation_type="surface_relaxation",
                                termination_index=0)
    explog.submit_process(p_surf)
    explog.update_log()
    assert _val(explog, cid, "n_surface_started") == 1
    assert _val(explog, cid, "n_surface_finalized") == 1
    assert _val(explog, cid, "n_O_started") == 0
    assert _val(explog, cid, "n_O_finalized") == 0
    assert pd.isna(_val(explog, cid, "n_OH_started"))

    # --- O ---
    surface_study = study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=True)
    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)
    p_o = explog.add_process(candidate_id=cid, calculation_type="O_adsorption",
                             termination_index=0, site_index=site_index)
    explog.submit_process(p_o)
    explog.update_log()
    assert _val(explog, cid, "n_O_started") == 1
    assert _val(explog, cid, "n_O_finalized") == 1
    assert _val(explog, cid, "n_OH_started") == 0
    assert _val(explog, cid, "n_OH_finalized") == 0

    # --- OH (a batch of 3 sub-jobs for the single site) ---
    oh_ids = explog.add_process(candidate_id=cid,
                                calculation_type="OH_adsorption",
                                termination_index=0, site_index=site_index)
    for pid in oh_ids:
        explog.submit_process(pid)
    explog.update_log()
    assert _val(explog, cid, "n_OH_started") == 1     # one SITE, not 3 sub-jobs
    assert _val(explog, cid, "n_OH_finalized") == 1
    assert _val(explog, cid, "state") == "active"

    # Regression guard: the same update_log path still fills the OER metrics.
    assert pd.notna(_val(explog, cid, "G(O) deviation"))


# ---------------------------------------------------------------------------
# The "ready for stage X" filter (the headline use case)
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_ready_for_stage_filter(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    for name in ("new", "bulk_done", "surf_done"):
        explog.add_candidate(candidate_id=name,
                             study_obj=_new_study(tio2_rutile_atoms),
                             reason_or_hypothesis="filter test")

    # bulk_done: bulk relaxed, nothing further -> ready for surface.
    _inject_process(explog, "bulk_done", "bulk_relaxation", "completed")
    # surf_done: bulk + surface relaxed -> ready for O.
    _inject_process(explog, "surf_done", "bulk_relaxation", "completed")
    _inject_process(explog, "surf_done", "surface_relaxation", "completed",
                    termination_index=0)

    explog.update_log()
    cdf = explog.relational_frame.candidates.df

    def ready_for(col):
        # <NA> == 0 is <NA>; .fillna(False) drops the not-yet-eligible rows.
        return cdf[(cdf[col] == 0).fillna(False)]["candidate_id"].tolist()

    assert ready_for("n_bulk_started") == ["new"]
    assert ready_for("n_surface_started") == ["bulk_done"]
    assert ready_for("n_O_started") == ["surf_done"]


# ---------------------------------------------------------------------------
# Per-logical-unit counting: OH per site, bulk capped at 0/1
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_oh_counted_per_site_not_per_subjob(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="oh per site")

    # Minimal eligible chain so the OH counters are not gated to <NA>.
    _inject_process(explog, cid, "bulk_relaxation", "completed")
    _inject_process(explog, cid, "surface_relaxation", "completed",
                    termination_index=0)
    _inject_process(explog, cid, "O_adsorption", "completed",
                    termination_index=0, site_index=0)
    # Site 0: all 3 OH sub-jobs terminal.
    for _ in range(3):
        _inject_process(explog, cid, "OH_adsorption", "completed",
                        termination_index=0, site_index=0)
    # Site 1: 2 done + 1 still running -> site not fully terminal.
    _inject_process(explog, cid, "OH_adsorption", "completed",
                    termination_index=0, site_index=1)
    _inject_process(explog, cid, "OH_adsorption", "completed",
                    termination_index=0, site_index=1)
    running = _inject_process(explog, cid, "OH_adsorption", "running",
                              termination_index=0, site_index=1)

    explog.update_log()
    assert _val(explog, cid, "n_OH_started") == 2      # two sites, not six rows
    assert _val(explog, cid, "n_OH_finalized") == 1    # only site 0 all-terminal

    _set_status(explog, running, "completed")
    explog.update_log()
    assert _val(explog, cid, "n_OH_started") == 2
    assert _val(explog, cid, "n_OH_finalized") == 2


@pytest.mark.mini
def test_bulk_capped_and_surface_gated_until_winner(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="bulk cap")

    # Two magnetic sub-jobs: one done, one still running.
    _inject_process(explog, cid, "bulk_relaxation", "completed")
    running = _inject_process(explog, cid, "bulk_relaxation", "running")
    explog.update_log()
    assert _val(explog, cid, "n_bulk_started") == 1      # capped, not 2
    assert _val(explog, cid, "n_bulk_finalized") == 0    # batch not settled
    assert pd.isna(_val(explog, cid, "n_surface_started"))  # no winner -> gated
    assert _val(explog, cid, "state") == "active"

    _set_status(explog, running, "completed")
    explog.update_log()
    assert _val(explog, cid, "n_bulk_started") == 1
    assert _val(explog, cid, "n_bulk_finalized") == 1
    assert _val(explog, cid, "n_surface_started") == 0   # winner -> surface opens


# ---------------------------------------------------------------------------
# Failed-state rule (bulk only) and latching
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_all_bulk_failed_marks_failed(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="all bulk failed")

    _inject_process(explog, cid, "bulk_relaxation", "failed")
    _inject_process(explog, cid, "bulk_relaxation", "failed")
    explog.update_log()
    assert _val(explog, cid, "state") == "failed"
    assert _val(explog, cid, "n_bulk_started") == 1      # one logical unit
    assert _val(explog, cid, "n_bulk_finalized") == 1    # terminal (failures count)
    assert pd.isna(_val(explog, cid, "n_surface_started"))


@pytest.mark.mini
def test_no_valid_surface_marks_failed(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="no valid surface")

    pid = _inject_process(explog, cid, "bulk_relaxation", "completed")
    explog.update_log()
    assert _val(explog, cid, "state") == "active"

    # Reproduce get_terminations_ranking's no-surface overwrite (material_agent
    # src/tools.py): every bulk row -> "unrecoverable: no valid surface ...".
    _set_status(explog, pid, "unrecoverable: no valid surface can be determined")
    explog.update_log()
    assert _val(explog, cid, "state") == "failed"
    assert _val(explog, cid, "n_bulk_finalized") == 1
    assert pd.isna(_val(explog, cid, "n_surface_started"))


@pytest.mark.mini
def test_failed_surface_keeps_candidate_active(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="surface failure")

    _inject_process(explog, cid, "bulk_relaxation", "completed")
    _inject_process(explog, cid, "surface_relaxation", "failed",
                    termination_index=0)
    explog.update_log()
    # A single failed surface must NOT fail the candidate (other surfaces remain).
    assert _val(explog, cid, "state") == "active"
    assert _val(explog, cid, "n_surface_started") == 1
    assert _val(explog, cid, "n_surface_finalized") == 1   # finalized includes failures
    # ...but a finalized-yet-failed surface is not usable, so O stays gated.
    assert pd.isna(_val(explog, cid, "n_O_started"))


@pytest.mark.mini
def test_failed_state_latches(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="latch")

    pid = _inject_process(explog, cid, "bulk_relaxation", "failed")
    explog.update_log()
    assert _val(explog, cid, "state") == "failed"

    # A later bulk completion must NOT revive a candidate already marked failed.
    _set_status(explog, pid, "completed")
    explog.update_log()
    assert _val(explog, cid, "state") == "failed"


# ---------------------------------------------------------------------------
# In-flight semantics, non-bulk unrecoverable, and edge cases
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_in_flight_counts_for_multi_job_stages(tmp_path, tio2_rutile_atoms):
    """For the per-row stages (surface, O), started > finalized while jobs are
    still running, i.e. in-flight = started - finalized."""
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="in flight")

    _inject_process(explog, cid, "bulk_relaxation", "completed")
    # Two surface relaxations: one done, one still running.
    _inject_process(explog, cid, "surface_relaxation", "completed",
                    termination_index=0)
    _inject_process(explog, cid, "surface_relaxation", "running",
                    termination_index=1)
    explog.update_log()
    assert _val(explog, cid, "n_surface_started") == 2
    assert _val(explog, cid, "n_surface_finalized") == 1     # one still in flight

    # One surface completed -> O is eligible. Two O jobs: one done, one running.
    _inject_process(explog, cid, "O_adsorption", "completed",
                    termination_index=0, site_index=0)
    _inject_process(explog, cid, "O_adsorption", "running",
                    termination_index=0, site_index=1)
    explog.update_log()
    assert _val(explog, cid, "n_O_started") == 2
    assert _val(explog, cid, "n_O_finalized") == 1
    assert _val(explog, cid, "state") == "active"


@pytest.mark.mini
def test_unrecoverable_surface_finalized_but_keeps_active(tmp_path, tio2_rutile_atoms):
    """A non-bulk 'unrecoverable: <reason>' row (e.g. a dissociated surface) is
    terminal -> counts as finalized, does NOT fail the candidate, and does not
    open the next stage (it produced no usable result)."""
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "c"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="dissociated surface")

    _inject_process(explog, cid, "bulk_relaxation", "completed")
    _inject_process(
        explog, cid, "surface_relaxation",
        "unrecoverable: Relaxed surface dissociated and this surface can "
        "therefore not be used for adsorption studies.",
        termination_index=0,
    )
    explog.update_log()
    assert _val(explog, cid, "state") == "active"          # surface-level, not candidate-level
    assert _val(explog, cid, "n_surface_started") == 1
    assert _val(explog, cid, "n_surface_finalized") == 1   # 'unrecoverable' is terminal
    assert pd.isna(_val(explog, cid, "n_O_started"))       # no usable surface -> O gated


@pytest.mark.mini
def test_recompute_on_empty_frame_does_not_raise(tmp_path):
    """update_log() (which runs the recompute) must be a no-op crash-free pass
    when there are no candidates and no processes."""
    explog = Explog()
    explog.init(tmp_path, mode="test")
    explog.update_log()  # must not raise on empty groupby / map
    assert len(explog.relational_frame.candidates.df) == 0
    # the new columns exist on the (empty) schema
    for col in ("state", "n_bulk_started", "n_OH_finalized"):
        assert col in explog.relational_frame.candidates.df.columns


# ---------------------------------------------------------------------------
# Bulk magnetic-enumeration failure: candidate is recorded as failed even though
# no bulk process row was ever entered (pymatgen raises before entry).
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_manual_failed_state_latches_without_a_failed_row(tmp_path, tio2_rutile_atoms):
    # A candidate whose bulk enumeration failed has NO process rows; mark_candidate_failed
    # latches state='failed'. The recompute on the next update_log must PRESERVE it (the
    # latch in _recompute_candidate_progress), not reset it to 'active'.
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "broken"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="enumeration failed -> no process rows")
    explog.relational_frame.candidates.set_value(
        cid, "state", "failed", allow_new_columns=True)
    explog.update_log()                                # recompute must LATCH, not reset
    assert _val(explog, cid, "state") == "failed"


@pytest.mark.mini
def test_mark_candidate_failed_sets_latches_and_ignores_unknown(tmp_path, tio2_rutile_atoms):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "broken"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="bulk enumeration failed")
    explog.mark_candidate_failed(cid)
    assert _val(explog, cid, "state") == "failed"
    explog.update_log()                                # survives a recompute
    assert _val(explog, cid, "state") == "failed"
    explog.mark_candidate_failed("does-not-exist")     # unknown id -> no-op, no crash


@pytest.mark.mini
def test_bulk_enumeration_failure_raises_and_enters_no_rows(
        tmp_path, tio2_rutile_atoms, monkeypatch):
    # When the magnetic enumeration raises (pymatgen ValueError "Too many magnetic
    # sites..."), enter_process must translate it to MagneticEnumerationError and leave
    # NO bulk process rows behind.
    from gnome_dreams_oer_screening.vasp import job_handler as jh_mod
    from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
        MagneticEnumerationError)
    explog = Explog()
    explog.init(tmp_path, mode="test")
    cid = "tio2"
    explog.add_candidate(candidate_id=cid, study_obj=_new_study(tio2_rutile_atoms),
                         reason_or_hypothesis="enum failure")

    def _boom(*args, **kwargs):
        raise ValueError("Too many magnetic sites to sensibly perform enumeration.")
    monkeypatch.setattr(jh_mod, "enumerate_magnetic_structures", _boom)

    with pytest.raises(MagneticEnumerationError):
        explog.add_process(candidate_id=cid, calculation_type="bulk_relaxation")
    pdf = explog.relational_frame.processes.df
    assert len(pdf[pdf["candidate_id"] == cid]) == 0   # no rows entered
