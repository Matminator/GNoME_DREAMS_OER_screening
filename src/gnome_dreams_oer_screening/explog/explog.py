import os
import pandas as pd
from typing import Literal
from pathlib import Path
from pysqa import QueueAdapter
from sympy import rf
import pickle

from .relational_frame import RelationalFrame
from gnome_dreams_oer_screening.oer.oer_study import OER_catalyst_study
from gnome_dreams_oer_screening.vasp.job_handler import Job_handler
from gnome_dreams_oer_screening.vasp.job_validation import (
    ValidationResult, validate_enter_process)


class Explog:
    # we'll manually init later
    def init(self, 
        vasp_calculation_dir: Path | str = Path("vasp_calculations"),
        mode: Literal["test", "slurm_test", "MLIP_test", "production_test", 
                      "production"] = "production",
        prefix_vasp_dirs: bool = True,
        sleep_min = 2,
        reject_if_failed_exists: bool = False,
        require_relaxed_o_for_oh: bool = False,
        magnetic_bulk_n_max: int = 4,
        legacy_disposition_exempt_ids: set | None = None,
        disposition_decisions=None,
        ):
        """
        Initialize the experiment log and attached job handler.

        Args:
            reject_if_failed_exists:
                Default duplicate policy used for process-entry
                validation. If True, matching failed processes also
                block re-entry unless overridden per call.
            require_relaxed_o_for_oh:
                Default OH policy used for process-entry validation. If
                True, OH jobs require an accepted relaxed O structure
                unless overridden per call.
        """
        # Setup of relational frame, with tables for candidates and 
        # processes (both of which are needed for the Job_handler etc.)
        rf = RelationalFrame(root="candidates")

        study_df = pd.DataFrame({
            "candidate_id": pd.Series(dtype=pd.StringDtype()),
            # AQ-GNoME "Reduced Formula" for this candidate (candidate_id ==
            # AQ-GNoME MaterialId), stored under that EXACT name for
            # consistency with OER_data_analasis_v2/browse_df, which use the
            # same AQ-GNoME field name. This package has no aq_gnome
            # dependency, so the value is looked up by the caller and passed
            # into add_candidate; self-healed (column created, NA-filled) on
            # update_log() for checkpoints that predate this column.
            "Reduced Formula": pd.Series(dtype=pd.StringDtype()),
            # AQ-GNoME's maximum (worst-case) Pourbaix decomposition energy
            # for this candidate, stored under AQ-GNoME's own field name
            # (matches OER_data_analasis_v2). Like Reduced Formula, looked up
            # by the caller (computing it requires the AQ-GNoME H5PY
            # decomposition-energy data, unavailable to this package) and
            # passed into add_candidate; self-healed on update_log().
            "max_dG_U[1.2,2.0]_pH0": pd.Series(dtype=pd.Float64Dtype()),
            "reason_or_hypothesis": pd.Series(dtype=pd.StringDtype()),
            "notes": pd.Series(dtype=pd.StringDtype()),
            "study_obj": pd.Series(dtype="object"), # can store any Python object
            "G(O) deviation": pd.Series(dtype=pd.Float64Dtype()),
            "Overpotential_from_scaling": pd.Series(dtype=pd.Float64Dtype()),
            "idealOverPotential": pd.Series(dtype=pd.Float64Dtype()), # nullable float (supports <NA>)
            # --- per-candidate progress projection ---
            # Derived in Job_handler._recompute_candidate_progress and
            # recomputed on every update_log(). Counts are per LOGICAL UNIT,
            # not per raw DFT sub-job:
            #   bulk -> 0/1 (one bulk relaxation per candidate, even though it
            #           runs up to N magnetic sub-jobs internally)
            #   OH   -> per adsorption (termination, site), not per 3 sub-jobs
            # "started" = work initiated; "finalized" = terminal (incl failures).
            # surface/O/OH counters stay <NA> until the candidate is eligible
            # for that stage, so "n_X_started == 0" means "ready for stage X".
            "n_bulk_started": pd.Series(dtype=pd.Int64Dtype()),
            "n_bulk_finalized": pd.Series(dtype=pd.Int64Dtype()),
            "n_surface_started": pd.Series(dtype=pd.Int64Dtype()),
            "n_surface_finalized": pd.Series(dtype=pd.Int64Dtype()),
            "n_O_started": pd.Series(dtype=pd.Int64Dtype()),
            "n_O_finalized": pd.Series(dtype=pd.Int64Dtype()),
            "n_OH_started": pd.Series(dtype=pd.Int64Dtype()),
            "n_OH_finalized": pd.Series(dtype=pd.Int64Dtype()),
            "state": pd.Series(dtype=pd.StringDtype()), # 'active' | 'failed'
            # --- disposition gate ---
            # Append-only list of disposition dicts
            #   {Summary, Summarized_process_id, Future_plan, Decision}.
            # Not exposed to the agent (dropped like study_obj); read via the
            # get_disposition_info tool.
            "disposition_record": pd.Series(dtype="object"),
            # Decision of the latest disposition (one of
            # var.DISPOSITION_DECISIONS). Agent-visible for filtering/sorting.
            "decision": pd.Series(dtype=pd.StringDtype()),
            # Write-lock forcing get-before-update; not exposed to the agent.
            "ready_for_disposition_update": pd.Series(dtype=pd.BooleanDtype()),
            # Derived in _recompute_disposition_state on every update_log():
            # True when the candidate has finalized-but-undispositioned work.
            # Agent-visible for filtering.
            "needs_disposition_update": pd.Series(dtype=pd.BooleanDtype()),
        })
        rf.add_table("candidates", study_df, pk="candidate_id")

        process_df = pd.DataFrame({
            "process_id": pd.Series(dtype=pd.Int64Dtype()),
            "candidate_id": pd.Series(dtype=pd.StringDtype()),
            "job_type": pd.Series(dtype=pd.StringDtype()),
            "slurmID": pd.Series(dtype=pd.StringDtype()),
            "status": pd.Series(dtype=pd.StringDtype()),
            'termination_index': pd.Series(dtype=pd.Int64Dtype()), # nullable integer (supports <NA>)
            'site_index': pd.Series(dtype=pd.Int64Dtype()),       # nullable integer (supports <NA>)
            "VASP_dir": pd.Series(dtype=pd.StringDtype()),
            "processNote": pd.Series(dtype=pd.StringDtype()),
            "G(O)": pd.Series(dtype=pd.Float64Dtype()),
            "G(O) deviation": pd.Series(dtype=pd.Float64Dtype()),
            "G(OH)": pd.Series(dtype=pd.Float64Dtype()),
            "G(OH) deviation": pd.Series(dtype=pd.Float64Dtype()),
            "G(OOH) from scaling relation": pd.Series(
                dtype=pd.Float64Dtype()
            ),
            "G(OOH, scaling) deviation": pd.Series(
                dtype=pd.Float64Dtype()
            ),
            "ideal overpotential": pd.Series(dtype=pd.Float64Dtype()),
            "overpotential from OH-OOH scaling relation": pd.Series(
                dtype=pd.Float64Dtype()
            ),
        })
        rf.add_table("processes", process_df, pk="process_id")

        # Add relation between candidates and processes:
        rf.candidates.add_relation(name="processes", child_table="processes",
            fk_in_child="candidate_id")
        
        # Ensure vasp_calculation_dir is a Path object
        if isinstance(vasp_calculation_dir, str):
            vasp_calculation_dir = Path(vasp_calculation_dir)
        self.relational_frame = rf
        self.vasp_calculation_dir = vasp_calculation_dir
        self.job_handler = Job_handler(
            rf,
            vasp_calculation_dir,
            mode=mode,
            prefix_vasp_dirs_with_prefix=prefix_vasp_dirs,
            sleep_min=sleep_min,
            reject_if_failed_exists=reject_if_failed_exists,
            require_relaxed_o_for_oh=require_relaxed_o_for_oh,
            magnetic_bulk_n_max=magnetic_bulk_n_max,
            legacy_disposition_exempt_ids=legacy_disposition_exempt_ids,
            disposition_decisions=disposition_decisions,
        )
        
    def _save_pickle(self) -> None:
        payload = {
            "candidates_df": self.relational_frame.candidates.df,
            "processes_df": self.relational_frame.processes.df,
        }
        # Write to a temporary file first and then atomically replace
        # the main file - This guards against data corruption if the 
        # program crashes during writing.
        tmp = self.vasp_calculation_dir / "explog.pkl.tmp"
        dest = self.vasp_calculation_dir / "explog.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        try:
            os.replace(tmp, dest)
        except PermissionError:
            # os.replace is atomic on Linux HPC (where it matters).
            # On Windows it can fail if the destination is locked by
            # another handle; fall back to a non-atomic rename.
            if dest.exists():
                dest.unlink()
            tmp.rename(dest)

    def add_candidate(self,
                      candidate_id: str,
                      study_obj: OER_catalyst_study,
                      reason_or_hypothesis: str | None = None,
                      notes: str | None = None,
                      reduced_formula: str | None = None,
                      decomposition_energy: float | None = None) -> None:

        # Self-heal the Reduced Formula / decomposition-energy columns
        # before writing: new_candidate_row always includes both keys, so on
        # a resumed pre-feature checkpoint (columns missing entirely) this
        # would otherwise raise "unknown columns" if add_candidate is the
        # first EXPLOG call after resume -- i.e. before any
        # update_log()-triggering tool call has had a chance to self-heal
        # them first. Safe/idempotent to call unconditionally.
        self.job_handler._ensure_reduced_formula_column()
        self.job_handler._ensure_decomposition_energy_column()

        if candidate_id in \
            self.relational_frame.candidates.df['candidate_id'].values:
            raise ValueError(f"Candidate with id {candidate_id} " \
                              "already exists in the log.")

        new_candidate_row = {"candidate_id": candidate_id,
                             "Reduced Formula": reduced_formula,
                             "max_dG_U[1.2,2.0]_pH0": decomposition_energy,
                             "reason_or_hypothesis": reason_or_hypothesis,
                             "notes": notes,
                             "study_obj": study_obj,
                             # A fresh candidate is active and ready for bulk;
                             # surface/O/OH counters stay <NA> (not eligible).
                             # Recomputed (idempotently) on the next update_log.
                             "state": "active",
                             "n_bulk_started": 0,
                             "n_bulk_finalized": 0,
                             # A fresh candidate has no finalized work yet, so it
                             # is vacuously fully-dispositioned: no record, no
                             # pending need, write-lock closed (a get must
                             # precede any update). Recomputed on update_log.
                             "disposition_record": [],
                             "decision": None,
                             "ready_for_disposition_update": False,
                             "needs_disposition_update": False}
        self.relational_frame.candidates.add_row(new_candidate_row,
                                                 allow_update=False)

    def can_add_process(
        self,
        candidate_id: str,
        calculation_type: Literal["bulk_relaxation",
            "surface_relaxation", "O_adsorption", "OH_adsorption"],
        termination_index: int | None = None,
        site_index: int | None = None,
        reject_if_failed_exists: bool | None = None,
        require_relaxed_o_for_oh: bool | None = None,
    ) -> ValidationResult:
        """
        Non-mutating pre-check for adding a process to the log.

        This wraps `validate_enter_process` and returns a structured
        `ValidationResult` without creating or updating any process
        rows. If optional policy arguments are `None`, handler defaults
        from `Explog.init(...)` are used.
        """
        if reject_if_failed_exists is None:
            reject_if_failed_exists = \
                self.job_handler.default_reject_if_failed_exists
        if require_relaxed_o_for_oh is None:
            require_relaxed_o_for_oh = \
                self.job_handler.default_require_relaxed_o_for_oh

        return validate_enter_process(
            self.relational_frame,
            candidate_id=candidate_id,
            job_type=calculation_type,
            termination_index=termination_index,
            site_index=site_index,
            reject_if_failed_exists=reject_if_failed_exists,
            require_relaxed_o_for_oh=require_relaxed_o_for_oh,
        )

    def add_process(self, 
        candidate_id: str, 
        calculation_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        termination_index: int | None = None,
        site_index: int | None = None,
        note: str | None = None,
        reject_if_failed_exists: bool | None = None,
        require_relaxed_o_for_oh: bool | None = None,
        ) -> int | list[int]:
        """
        Add one process (or an OH batch) to the processes table.

        If `reject_if_failed_exists` or `require_relaxed_o_for_oh` is
        `None`, the handler defaults from `Explog.init(...)` are used.
        """
        result =  self.job_handler.enter_process(
            candidate_id, 
            calculation_type, 
            termination_index, 
            site_index, 
            note,
            reject_if_failed_exists = reject_if_failed_exists,
            require_relaxed_o_for_oh = require_relaxed_o_for_oh
            )
        
        self._save_pickle()
        return result

    def submit_process(self, process_id: int | str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto']
        = 'auto') -> None:

        if isinstance(process_id, str):
            try:
                process_id = int(process_id)
            except ValueError:
                raise ValueError(f"Process ID must be an integer or a "+
                    "string that can be converted to an integer. Got " +
                    f"{process_id}.")

        self.job_handler.submit_vasp_calculation(process_id,
                                                 partition=partition)
        self._save_pickle()
        
    def update_log(self):
        result = self.job_handler.update_log()
        self._save_pickle()
        return result

    def candidates_needing_disposition(self) -> list:
        """Candidate ids that block waiting (Gate 1): those with finalized work
        not yet fully dispositioned.

        Recomputes the derived ``needs_disposition_update`` flag from the
        currently loaded frame (cheap; no HPC poll) so the answer reflects the
        latest known log state, then returns the flagged candidates. The wait
        tool refuses to wait while this is non-empty.
        """
        self.job_handler._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        if "needs_disposition_update" not in cdf.columns or len(cdf) == 0:
            return []
        mask = cdf["needs_disposition_update"].astype("boolean").fillna(False)
        return [str(c) for c in cdf.loc[mask, "candidate_id"].tolist()]

    def get_disposition_info(self, candidate_id: str) -> dict:
        """Mark a candidate ready for a disposition write and return its
        outstanding-coverage info (the structured data the agent summarizes).

        Refreshes the disposition columns / needs flag, sets
        ``ready_for_disposition_update = True``, and returns
        ``candidate_outstanding(candidate_id)`` -- or ``{"unknown_candidate":
        True}`` when the candidate is not in the log (symmetric with
        update_disposition_info, so a bad id never reads like a vacuous success).
        The agent-facing sentence is built from the result by
        ``src/disposition_messages.py``.
        """
        self.job_handler._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        mask = cdf["candidate_id"] == candidate_id
        if not mask.any():
            return {"unknown_candidate": True, "candidate_id": candidate_id}
        cdf.loc[mask, "ready_for_disposition_update"] = True
        self._save_pickle()
        return self.job_handler.candidate_outstanding(candidate_id)

    def update_disposition_info(
        self,
        candidate_id: str,
        summary: str,
        summarized_process_id,
        future_plan: str,
        decision: str,
    ) -> dict:
        """Record (or reject) a disposition for a candidate.

        Returns a structured result
        ``{ok, status, candidate_id, decision, missing, ids}``; the
        agent-facing sentence is built from it by ``src/disposition_messages.py``.

        Write-lock: ONLY a successful write clears
        ``ready_for_disposition_update`` -- incomplete coverage and validation
        failures leave it set so the agent can retry without re-reading.
        """
        jh = self.job_handler
        # Self-heal the four disposition columns before any column read, so a
        # resumed pre-rollout checkpoint that predates them cannot KeyError here
        # (it returns "locked" via the write-lock check instead).
        jh._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        mask = cdf["candidate_id"] == candidate_id
        result = {
            "ok": False, "status": None, "candidate_id": candidate_id,
            "decision": decision, "missing": [], "ids": [],
        }

        # (0) the candidate must exist (otherwise the .iloc[0] reads below fail).
        if not mask.any():
            result["status"] = "unknown_candidate"
            return result

        # (1) write-lock: get_disposition_info must have run first.
        ready_val = cdf.loc[mask, "ready_for_disposition_update"].iloc[0]
        if not (pd.notna(ready_val) and bool(ready_val)):
            result["status"] = "locked"
            return result

        # (2) Decision must be in the allowed vocabulary (sourced from var.py).
        allowed = getattr(jh, "disposition_decisions", ()) or ()
        if allowed and decision not in allowed:
            result["status"] = "invalid_decision"
            return result

        pdf = self.relational_frame.processes.df
        cited = [int(i) for i in (summarized_process_id or [])]

        # (3) every cited id must belong to THIS candidate (catches ids copied
        # from another candidate, or ids that do not exist at all).
        own_ids = set(
            int(x) for x in
            pdf.loc[pdf["candidate_id"] == candidate_id, "process_id"].tolist()
        )
        foreign = [pid for pid in cited if pid not in own_ids]
        if foreign:
            result["status"] = "foreign_ids"
            result["ids"] = foreign
            return result

        # (4) every cited process id must be terminal (finalized).
        non_terminal = []
        for pid in cited:
            rows = pdf[pdf["process_id"] == pid]
            status = rows["status"].iloc[0] if not rows.empty else None
            if not jh._is_terminal_process_status(status):
                non_terminal.append(pid)
        if non_terminal:
            result["status"] = "non_terminal_ids"
            result["ids"] = non_terminal
            return result

        # (5) coverage: the cited ids must cover every required (non-exempt) unit.
        out = jh.candidate_outstanding(candidate_id)
        cited_set = set(cited)
        uncovered = [u for u in out["must_cover"]
                     if not (set(u["ids"]) & cited_set)]
        if uncovered:
            result["status"] = "incomplete"
            result["missing"] = sorted(
                {int(i) for u in uncovered for i in u["ids"]}
            )
            return result  # nothing written -> ready stays True

        # (6) success: append the disposition and update the flags.
        idx = cdf.index[mask][0]
        existing = cdf.at[idx, "disposition_record"]
        if not isinstance(existing, list):
            existing = []
        cdf.at[idx, "disposition_record"] = existing + [{
            "Summary": summary,
            "Summarized_process_id": cited,
            "Future_plan": future_plan,
            "Decision": decision,
        }]
        cdf.loc[mask, "decision"] = decision
        cdf.loc[mask, "ready_for_disposition_update"] = False
        jh._recompute_disposition_state()   # refresh needs from the new record
        self._save_pickle()
        result["ok"] = True
        result["status"] = "ok"
        return result

    def mark_candidate_failed(self, candidate_id: str) -> None:
        """Latch a candidate's state to 'failed' and persist it.

        Used when bulk setup fails before any process row is entered (e.g. the
        magnetic enumeration raises). The latch in _recompute_candidate_progress
        then keeps state == 'failed' on every later recompute, and
        find_forgotten_jobs / the wait-tool hints skip 'failed' candidates. The
        explicit _save_pickle is needed because the failing add_process aborts
        before its own save. No-op for an unknown candidate id.
        """
        cdf = self.relational_frame.candidates.df
        if not (cdf["candidate_id"] == candidate_id).any():
            return
        self.relational_frame.candidates.set_value(
            candidate_id, "state", "failed", allow_new_columns=True
        )
        self._save_pickle()

EXPLOG = Explog()
    
