# Tests for the disposition coverage engine (Step 2):
#   Job_handler._recompute_disposition_state  -> writes needs_disposition_update
#   Job_handler.candidate_outstanding         -> pure helper (shared by the tools)
# both run/derived on every update_log().
#
# These are written test-FIRST (TDD): the functions under test do not exist yet,
# so the suite is RED until Step 2 is implemented.
#
# Contract pinned here (the spec):
#   candidate_outstanding(candidate_id) -> dict with keys
#     "must_cover":        list[unit]   # terminal units, not covered, not exempt
#     "legacy_optional":   list[unit]   # terminal units, not covered, but in L
#     "latest_disposition": dict | None # last entry of disposition_record
#     "has_finalized":     bool         # any terminal (batch-collapsed) unit exists
#   where unit = {"job_type": str, "termination_index": int|None,
#                 "site_index": int|None, "ids": list[int] (sorted, plain ints)}
#
#   A unit is COVERED iff some disposition's Summarized_process_id intersects its
#   ids (citing any one sub-job covers the whole batch). A unit is EXEMPT iff all
#   its ids are in job_handler.legacy_disposition_exempt_ids (L).
#
#   needs_disposition_update(c) = (must_cover non-empty)
#                                 OR (disposition_record empty AND has_finalized)
#   "finalized" = terminal, INCLUDING failures/unrecoverable.
#
# Scenarios are built by injecting process rows with chosen statuses directly
# (as test_candidate_progress.py does); in "test" mode update_log() only advances
# 'pending' rows, so injected terminal/running rows are left untouched.

from pathlib import Path

import pandas as pd
import pytest

from gnome_dreams_oer_screening import Explog

# All tests here are fast (dummy study, injected rows, "test" mode) -> mini.
pytestmark = pytest.mark.mini


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _explog(tmp_path):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    return explog


def _add_candidate(explog, cid):
    # The coverage engine never reads study_obj, so a dummy object is enough.
    explog.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="disposition coverage test")


def _val(explog, cid, col):
    cdf = explog.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def _inject_process(explog, cid, job_type, status,
                    termination_index=None, site_index=None):
    table = explog.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row(
        {
            "process_id": pid,
            "candidate_id": cid,
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


def _set_legacy(explog, ids):
    explog.job_handler.legacy_disposition_exempt_ids = set(int(i) for i in ids)


def _add_disposition(explog, cid, summarized_ids, decision="Investigating"):
    cdf = explog.relational_frame.candidates.df
    idx = cdf.index[cdf["candidate_id"] == cid][0]
    rec = cdf.at[idx, "disposition_record"]
    if not isinstance(rec, list):
        rec = []
    rec = rec + [{
        "Summary": "s",
        "Summarized_process_id": [int(i) for i in summarized_ids],
        "Future_plan": "f",
        "Decision": decision,
    }]
    cdf.at[idx, "disposition_record"] = rec


def _co(explog, cid):
    return explog.job_handler.candidate_outstanding(cid)


def _covers(units, pid):
    return any(int(pid) in set(int(i) for i in u["ids"]) for u in units)


# ---------------------------------------------------------------------------
# 1. Fresh candidate: no finalized work -> not needed
# ---------------------------------------------------------------------------

def test_fresh_candidate_not_needed(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712
    co = _co(explog, "c")
    assert co["must_cover"] == []
    assert co["has_finalized"] is False
    assert co["latest_disposition"] is None


# ---------------------------------------------------------------------------
# 2. A terminal, uncovered unit -> needs True; it shows in must_cover
# ---------------------------------------------------------------------------

def test_terminal_surface_uncovered_needs(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    # Non-zero termination so the unit must report the real index.
    pid_s = _inject_process(explog, "c", "surface_relaxation", "completed",
                            termination_index=1)
    co = _co(explog, "c")
    assert co["has_finalized"] is True
    assert len(co["must_cover"]) == 1
    unit = co["must_cover"][0]
    assert unit["job_type"] == "surface_relaxation"
    assert unit["termination_index"] == 1      # the surface/termination index
    assert unit["site_index"] is None          # surface jobs have no site
    assert set(unit["ids"]) == {pid_s}         # non-batch -> single id
    assert _covers(co["must_cover"], pid_s)
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712


# ---------------------------------------------------------------------------
# 3. Disposition covering the unit clears the need
# ---------------------------------------------------------------------------

def test_disposition_clears_need(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    pid_s = _inject_process(explog, "c", "surface_relaxation", "completed",
                            termination_index=0)
    _add_disposition(explog, "c", [pid_s])
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712
    co = _co(explog, "c")
    assert co["must_cover"] == []
    assert co["latest_disposition"] is not None


# ---------------------------------------------------------------------------
# 4. Citing ANY ONE sub-job of a batch covers the whole batch unit
# ---------------------------------------------------------------------------

def test_cite_one_bulk_subjob_covers_batch(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    pid_b1 = _inject_process(explog, "c", "bulk_relaxation", "completed")
    pid_b2 = _inject_process(explog, "c", "bulk_relaxation", "completed")
    co = _co(explog, "c")
    # one bulk unit whose ids contain both sub-jobs
    assert len(co["must_cover"]) == 1
    assert set(co["must_cover"][0]["ids"]) == {pid_b1, pid_b2}
    _add_disposition(explog, "c", [pid_b1])  # cite only one
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712


# ---------------------------------------------------------------------------
# 5. A partial OH batch is not yet a finalized unit; completing it makes it one
# ---------------------------------------------------------------------------

def test_partial_oh_batch_not_a_unit(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    # Distinct non-zero indices so the unit must report the real values.
    pid1 = _inject_process(explog, "c", "OH_adsorption", "completed",
                           termination_index=2, site_index=3)
    pid2 = _inject_process(explog, "c", "OH_adsorption", "completed",
                           termination_index=2, site_index=3)
    pid3 = _inject_process(explog, "c", "OH_adsorption", "running",
                           termination_index=2, site_index=3)
    co = _co(explog, "c")
    assert co["has_finalized"] is False        # site not all-terminal -> no unit
    assert co["must_cover"] == []
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712

    _set_status(explog, pid3, "completed")     # batch now complete
    co = _co(explog, "c")
    assert co["has_finalized"] is True
    assert len(co["must_cover"]) == 1
    unit = co["must_cover"][0]
    assert unit["job_type"] == "OH_adsorption"
    assert unit["termination_index"] == 2      # the surface/termination index
    assert unit["site_index"] == 3
    # the collapsed unit carries ALL 3 sub-job ids (cite any one to cover it)
    assert set(unit["ids"]) == {pid1, pid2, pid3}
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712


# ---------------------------------------------------------------------------
# 6. An all-failed bulk batch is terminal -> still requires a disposition
# ---------------------------------------------------------------------------

def test_failed_bulk_needs_disposition(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    _inject_process(explog, "c", "bulk_relaxation", "failed")
    _inject_process(explog, "c", "bulk_relaxation", "failed")
    co = _co(explog, "c")
    assert co["has_finalized"] is True
    assert len(co["must_cover"]) == 1
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712


# ---------------------------------------------------------------------------
# 7. Legacy-exempt unit: shows as legacy_optional, but the >=1-disposition rule
#    still forces exactly one disposition for the candidate.
# ---------------------------------------------------------------------------

def test_legacy_exempt_forces_one_disposition(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    pid_s = _inject_process(explog, "c", "surface_relaxation", "completed",
                            termination_index=0)
    _set_legacy(explog, [pid_s])
    co = _co(explog, "c")
    assert co["must_cover"] == []                 # exempt -> not required to cite
    assert _covers(co["legacy_optional"], pid_s)  # ...but visible as optional
    assert co["has_finalized"] is True
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712

    _add_disposition(explog, "c", [])             # one disposition, no ids cited
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712


# ---------------------------------------------------------------------------
# 8. Mixed exempt + non-exempt: only the non-exempt unit is required
# ---------------------------------------------------------------------------

def test_mixed_exempt_and_nonexempt(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    pid_old = _inject_process(explog, "c", "surface_relaxation", "completed",
                              termination_index=0)
    pid_new = _inject_process(explog, "c", "surface_relaxation", "completed",
                              termination_index=1)
    _set_legacy(explog, [pid_old])
    co = _co(explog, "c")
    assert _covers(co["must_cover"], pid_new)
    assert not _covers(co["must_cover"], pid_old)
    assert _covers(co["legacy_optional"], pid_old)
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712

    _add_disposition(explog, "c", [pid_new])
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712


# ---------------------------------------------------------------------------
# 9. Coverage is over the UNION of dispositions
# ---------------------------------------------------------------------------

def test_union_coverage_two_dispositions(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    pid1 = _inject_process(explog, "c", "surface_relaxation", "completed",
                           termination_index=0)
    pid2 = _inject_process(explog, "c", "O_adsorption", "completed",
                           termination_index=0, site_index=0)
    _add_disposition(explog, "c", [pid1])
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712 (pid2 unit uncovered)

    _add_disposition(explog, "c", [pid2])
    explog.update_log()
    assert _val(explog, "c", "needs_disposition_update") == False  # noqa: E712


# ---------------------------------------------------------------------------
# 10. Self-heal: a resumed pre-change checkpoint lacks the 4 columns
# ---------------------------------------------------------------------------

def test_self_heal_missing_columns(tmp_path):
    explog = _explog(tmp_path)
    _add_candidate(explog, "c")
    _inject_process(explog, "c", "surface_relaxation", "completed",
                    termination_index=0)
    # Simulate an old checkpoint: drop the disposition columns entirely.
    cdf = explog.relational_frame.candidates.df
    explog.relational_frame.candidates.df = cdf.drop(columns=[
        "disposition_record", "decision",
        "ready_for_disposition_update", "needs_disposition_update",
    ])
    explog.update_log()  # must self-heal, not raise
    cols = explog.relational_frame.candidates.df.columns
    for c in ("disposition_record", "decision",
              "ready_for_disposition_update", "needs_disposition_update"):
        assert c in cols
    assert _val(explog, "c", "needs_disposition_update") == True  # noqa: E712


# ---------------------------------------------------------------------------
# 11. Empty frame: the recompute is a crash-free no-op
# ---------------------------------------------------------------------------

def test_empty_frame_no_raise(tmp_path):
    explog = _explog(tmp_path)
    explog.update_log()
    assert len(explog.relational_frame.candidates.df) == 0
    assert "needs_disposition_update" in explog.relational_frame.candidates.df.columns
