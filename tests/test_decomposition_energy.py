# Tests for the persisted `max_dG_U[1.2,2.0]_pH0` (Pourbaix decomposition
# energy) column on the candidates table, written test-FIRST (TDD). Mirrors
# test_reduced_formula.py's pattern -- see there for the general rationale
# (this package has no aq_gnome dependency; values are computed by the caller
# and passed into add_candidate). One real difference: unlike "Reduced
# Formula" (a cheap AQ-GNoME CSV lookup), computing this value is an H5PY
# read per candidate (see src/aq_gnome_candidate_sync.py's
# compute_decomposition_energy), so reconciliation on resume must fill ONLY
# candidates missing a value, never unconditionally recompute -- that's
# sync_decomposition_energy's job in the material_agent repo, not this
# package's self-heal (which, like reduced_formula's, only guarantees the
# column EXISTS).
#
# Contract pinned here:
#   Explog.add_candidate(..., decomposition_energy: float | None = None)
#     stores the value (or <NA> if omitted) in a new "max_dG_U[1.2,2.0]_pH0"
#     column.
#   Job_handler.update_log() self-heals the column (creates it, NA-filled) if
#     a resumed checkpoint predates this feature.
#   Explog.add_candidate ALSO self-heals the column itself (independent of
#     update_log()), so a resumed pre-feature checkpoint never raises
#     "unknown columns" if add_candidate happens to be the first EXPLOG call
#     after resume.

import pandas as pd
import pytest

from gnome_dreams_oer_screening import Explog

pytestmark = pytest.mark.mini

DECOMP_COL = "max_dG_U[1.2,2.0]_pH0"


def _explog(tmp_path):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    return explog


def _val(explog, cid, col):
    cdf = explog.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def test_add_candidate_stores_decomposition_energy(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", decomposition_energy=0.42)
    assert _val(explog, "c", DECOMP_COL) == 0.42


def test_add_candidate_without_decomposition_energy_is_na(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r")
    assert pd.isna(_val(explog, "c", DECOMP_COL))


def test_self_heals_missing_decomposition_energy_column_on_update_log(tmp_path):
    # Simulate a pre-feature checkpoint: the column doesn't exist at all.
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", decomposition_energy=0.1)
    cdf = explog.relational_frame.candidates.df
    explog.relational_frame.candidates.df = cdf.drop(columns=[DECOMP_COL])
    explog.update_log()  # must self-heal, not raise
    assert DECOMP_COL in explog.relational_frame.candidates.df.columns
    # Self-heal only creates the column (NA-filled); it cannot recover the
    # original value since this package can't compute it.
    assert pd.isna(_val(explog, "c", DECOMP_COL))


def test_self_heal_does_not_clobber_existing_values(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", decomposition_energy=0.1)
    explog.update_log()
    assert _val(explog, "c", DECOMP_COL) == 0.1


def test_empty_frame_no_raise(tmp_path):
    explog = _explog(tmp_path)
    explog.update_log()
    assert len(explog.relational_frame.candidates.df) == 0
    assert DECOMP_COL in explog.relational_frame.candidates.df.columns


def test_add_candidate_self_heals_without_update_log_first(tmp_path):
    # Simulate a resumed pre-feature checkpoint whose candidates.df predates
    # the column entirely, where add_candidate is the FIRST EXPLOG call after
    # resume -- no update_log() has run yet, so its self-heal hasn't fired.
    # add_candidate must not raise "unknown columns ['max_dG_U[1.2,2.0]_pH0']".
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="old", study_obj=object(),
                         reason_or_hypothesis="r", decomposition_energy=0.2)
    cdf = explog.relational_frame.candidates.df
    explog.relational_frame.candidates.df = cdf.drop(columns=[DECOMP_COL])

    explog.add_candidate(candidate_id="new", study_obj=object(),
                         reason_or_hypothesis="r", decomposition_energy=0.3)

    assert DECOMP_COL in explog.relational_frame.candidates.df.columns
    assert _val(explog, "new", DECOMP_COL) == 0.3
    # The self-heal (schema only) also reaches the pre-existing row, but it
    # cannot recover a value this package never had -- stays <NA>.
    assert pd.isna(_val(explog, "old", DECOMP_COL))
