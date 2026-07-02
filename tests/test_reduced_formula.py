# Tests for the persisted `reduced_formula` column on the candidates table,
# written test-FIRST (TDD). This column holds the AQ-GNoME "Reduced Formula"
# string for each candidate (candidate_id == AQ-GNoME MaterialId). The value
# itself is looked up by the caller (src/tools.py, which has access to the
# AQ-GNoME database) and passed into add_candidate -- this package has no
# aq_gnome dependency and never does the lookup itself.
#
# Contract pinned here:
#   Explog.add_candidate(..., reduced_formula: str | None = None) stores the
#     value (or <NA> if omitted) in a new "reduced_formula" column.
#   The column is the 2nd column of the candidates table (right after
#     candidate_id) -- "one of the first columns".
#   Job_handler.update_log() self-heals the column (creates it, NA-filled) if
#     a resumed checkpoint predates this feature, mirroring how the four
#     disposition columns already self-heal in _recompute_disposition_state.
#     Self-healing only creates the column; it never fills in real values
#     (that requires the AQ-GNoME database, which this package cannot see).
#   Explog.add_candidate ALSO self-heals the column itself (independent of
#     update_log()), so a resumed pre-feature checkpoint never raises
#     "unknown columns" if add_candidate happens to be the first EXPLOG call
#     after resume (i.e. before any update_log()-triggering tool has run).

import pandas as pd
import pytest

from gnome_dreams_oer_screening import Explog

pytestmark = pytest.mark.mini


def _explog(tmp_path):
    explog = Explog()
    explog.init(tmp_path, mode="test")
    return explog


def _val(explog, cid, col):
    cdf = explog.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def test_add_candidate_stores_reduced_formula(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", reduced_formula="RuO2")
    assert _val(explog, "c", "reduced_formula") == "RuO2"


def test_add_candidate_without_reduced_formula_is_na(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r")
    assert pd.isna(_val(explog, "c", "reduced_formula"))


def test_reduced_formula_is_second_column(tmp_path):
    explog = _explog(tmp_path)
    cols = list(explog.relational_frame.candidates.df.columns)
    assert cols[0] == "candidate_id"
    assert cols[1] == "reduced_formula"


def test_self_heals_missing_reduced_formula_column_on_update_log(tmp_path):
    # Simulate a pre-feature checkpoint: the column doesn't exist at all.
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", reduced_formula="RuO2")
    cdf = explog.relational_frame.candidates.df
    explog.relational_frame.candidates.df = cdf.drop(columns=["reduced_formula"])
    explog.update_log()  # must self-heal, not raise
    assert "reduced_formula" in explog.relational_frame.candidates.df.columns
    # Self-heal only creates the column (NA-filled); it cannot recover the
    # original value since this package has no AQ-GNoME access.
    assert pd.isna(_val(explog, "c", "reduced_formula"))


def test_self_heal_does_not_clobber_existing_values(tmp_path):
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="c", study_obj=object(),
                         reason_or_hypothesis="r", reduced_formula="RuO2")
    explog.update_log()
    assert _val(explog, "c", "reduced_formula") == "RuO2"


def test_empty_frame_no_raise(tmp_path):
    explog = _explog(tmp_path)
    explog.update_log()
    assert len(explog.relational_frame.candidates.df) == 0
    assert "reduced_formula" in explog.relational_frame.candidates.df.columns


def test_add_candidate_self_heals_without_update_log_first(tmp_path):
    # Simulate a resumed pre-feature checkpoint whose candidates.df predates
    # the column entirely, where add_candidate is the FIRST EXPLOG call after
    # resume -- no update_log() has run yet, so its self-heal hasn't fired.
    # add_candidate must not raise "unknown columns ['reduced_formula']".
    explog = _explog(tmp_path)
    explog.add_candidate(candidate_id="old", study_obj=object(),
                         reason_or_hypothesis="r", reduced_formula="TiO2")
    cdf = explog.relational_frame.candidates.df
    explog.relational_frame.candidates.df = cdf.drop(columns=["reduced_formula"])

    explog.add_candidate(candidate_id="new", study_obj=object(),
                         reason_or_hypothesis="r", reduced_formula="RuO2")

    assert "reduced_formula" in explog.relational_frame.candidates.df.columns
    assert _val(explog, "new", "reduced_formula") == "RuO2"
    # The self-heal (schema only) also reaches the pre-existing row, but it
    # cannot recover a value this package never had -- stays <NA>.
    assert pd.isna(_val(explog, "old", "reduced_formula"))