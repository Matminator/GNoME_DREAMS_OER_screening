from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Unique job types possible in via the jobhandler:
JobType = Literal[
    "bulk_relaxation",
    "surface_relaxation",
    "O_adsorption",
    "OH_adsorption",
]


@dataclass(frozen=True)
class ValidationResult:
    """
    Structured result for non-mutating process validation checks.
    """

    ok: bool
    code: str | None = None
    message: str | None = None
    error_type: type[Exception] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def raise_for_error(self) -> None:
        if self.ok:
            return
        exc_type = self.error_type or RuntimeError
        raise exc_type(self.message or self.code or 
                       "Validation failed.")
    

def _pass(**context: Any) -> ValidationResult:
    return ValidationResult(ok=True, context=context)


def _fail(
    code: str,
    message: str,
    error_type: type[Exception],
    **context: Any,
) -> ValidationResult:
    return ValidationResult(
        ok=False,
        code=code,
        message=message,
        error_type=error_type,
        context=context,
    )


def _extract_oh_init_index(process_note: object) -> int | None:
    if not isinstance(process_note, str):
        return None
    token = "OH_init_index="
    if token not in process_note:
        return None
    tail = process_note.split(token, 1)[1]
    tail = tail.split("|", 1)[0].strip()
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None


def validate_enter_process(
    rframe,
    candidate_id: str,
    job_type: JobType,
    termination_index: int | None = None,
    site_index: int | None = None,
    reject_if_failed_exists: bool = False,
    require_relaxed_o_for_oh: bool = False,
) -> ValidationResult:
    """
    Validate whether a new process may be created for a candidate.

    This check is non-mutating and reusable by external callers. It
    mirrors the policy checks used when entering a process in the job
    handler, including:
    - duplicate process checks for the same
      `(job_type, termination_index, site_index)` scope,
    - required/forbidden indices for each job type,
    - required initialized surface/adsorption-site studies, and
    - guards against re-running accepted O-stage jobs and OH
      re-batching.
    - optional guard enforcing accepted O-stage before OH-stage.

    Args:
        rframe:
            The relational frame containing `candidates` and
            `processes` tables.
        candidate_id:
            Candidate identifier in the `candidates` table.
        job_type:
            Job type to validate
            (`bulk_relaxation`, `surface_relaxation`, `O_adsorption`,
            or `OH_adsorption`).
        termination_index:
            Surface termination index for surface/adsorption jobs.
        site_index:
            Adsorption-site index for adsorption jobs.
        reject_if_failed_exists:
            If False (default), previously failed matching jobs do not
            block creating a new process. If True, any matching job
            status (including `failed`) blocks creation.
        require_relaxed_o_for_oh:
            If True, `OH_adsorption` is rejected unless a relaxed O
            structure has already been accepted in the site study
            (`ad_study.relaxed_O_atoms is not None`).
            Defaults to False.

    Returns:
        ValidationResult:
            `ok=True` when creation is allowed. In that case,
            `context` includes resolved objects (for example candidate,
            study, and possibly surface/ad-site studies) for downstream
            use.

            `ok=False` when creation is not allowed. In that case,
            `code`, `message`, and `error_type` describe the reason.

    Notes:
        This function does not raise on validation failures. Use
        `result.raise_for_error()` if raise-based flow is preferred.
    """

    # test if candidate exists:
    try:
        cand_row = rframe.candidates[candidate_id]
    except Exception:
        return _fail(
            "candidate_not_found",
            f"Candidate {candidate_id} does not exist.",
            KeyError,
            candidate_id=candidate_id,
        )

    # Make copy of candidate level pdf:
    cand_processes = cand_row.processes
    pdf = cand_processes.df.copy()

    # Get processes matching the termination_index, if provided:
    if termination_index is None:
        termination_mask = pdf["termination_index"].isna()
    else:
        termination_mask = pdf["termination_index"] == termination_index

    # Get processes matching the site_index, if provided:
    if site_index is None:
        site_mask = pdf["site_index"].isna()
    else:
        site_mask = pdf["site_index"] == site_index

    # Get duplicate processes matching the job_type and indices:
    matching_pdf = pdf[termination_mask & site_mask]
    duplicate_pdf = matching_pdf[matching_pdf["job_type"] == job_type]

    # If not reject_if_failed_exists, ignore failed jobs when checking:
    if not reject_if_failed_exists:
        duplicate_pdf = duplicate_pdf[
            duplicate_pdf["status"] != "failed"
        ]
    
    # If duplicate process - put together error message and fail:
    if not duplicate_pdf.empty:
        states = duplicate_pdf["status"].values
        if reject_if_failed_exists:
            code = "duplicate_process_including_failed"
            msg_prefix = "a calculation"
        else:
            code = "duplicate_non_failed_process"
            msg_prefix = "a non-failed calculation"
        return _fail(
            code,
            f"Candidate {candidate_id} has {msg_prefix} of "
            f"type {job_type} with status in {states}.",
            RuntimeError,
            candidate_id=candidate_id,
            job_type=job_type,
            termination_index=termination_index,
            site_index=site_index,
            reject_if_failed_exists=reject_if_failed_exists,
        )

    # Retrieve study object, to test validity of requsted job:
    study_obj = cand_row.df["study_obj"].values[0]

    if job_type == "bulk_relaxation":
        # No indices should be provided for bulk relaxations:
        if termination_index is not None or site_index is not None:
            return _fail(
                "bulk_indices_not_allowed",
                "bulk_relaxation does not use termination_index or "
                "site_index.",
                ValueError,
            )
        
        # Has relaxed bulk already been set for this candidate:
        if study_obj.get_bulk_atoms() is not None:
            return _fail(
                "bulk_already_set",
                f"Candidate {candidate_id} already has a relaxed bulk "
                "structure set.",
                RuntimeError,
            )

        return _pass(cand_row=cand_row, study_obj=study_obj)

    if job_type == "surface_relaxation":
        # Surface relaxation requires termination_index:
        if termination_index is None:
            return _fail(
                "surface_missing_termination_index",
                "surface_relaxation requires termination_index.",
                ValueError,
            )
        
        # site_index should not be provided for surface relaxation:
        if site_index is not None:
            return _fail(
                "surface_site_index_not_allowed",
                "surface_relaxation does not use site_index.",
                ValueError,
            )

        # test that surface study has been initialized:
        surface_studies = study_obj.get_surface_studies()
        if surface_studies is None:
            return _fail(
                "surface_studies_not_initialized",
                "No initialized surface studies exist for this "
                "candidate.",
                ValueError,
            )
        if termination_index not in surface_studies:
            return _fail(
                "surface_study_not_initialized",
                "No initialized surface study exists for "
                f"termination_index={termination_index}.",
                ValueError,
            )

        # test that relaxed surface has not already been set:
        surface_study = surface_studies[termination_index]
        if surface_study.base_surface_relaxed is not None:
            return _fail(
                "surface_already_relaxed",
                "A relaxed surface is already set for "
                f"termination_index={termination_index}.",
                RuntimeError,
            )

        return _pass(
            cand_row=cand_row,
            study_obj=study_obj,
            surface_study=surface_study,
            termination_index=int(termination_index),
        )

    if job_type in ["O_adsorption", "OH_adsorption"]:

        # termnation_index required for both O and OH adsorption cals:
        if termination_index is None:
            return _fail(
                "ads_missing_termination_index",
                f"{job_type} requires termination_index.",
                ValueError,
            )
        
        # site_index required for both O and OH adsorption cals:
        if site_index is None:
            return _fail(
                "ads_missing_site_index",
                f"{job_type} requires site_index.",
                ValueError,
            )

        # Make sure surface study exists for the requested calculation:
        surface_studies = study_obj.get_surface_studies()
        if surface_studies is None:
            return _fail(
                "surface_studies_not_initialized",
                "No initialized surface studies exist for this "
                "candidate.",
                ValueError,
            )
        if termination_index not in surface_studies:
            return _fail(
                "surface_study_not_initialized",
                "No initialized surface study exists for "
                f"termination_index={termination_index}.",
                ValueError,
            )
        surface_study = surface_studies[termination_index]

        # Make sure adsorption-site study exists for the requested calc:
        adsite_studies = surface_study.get_adsorption_site_studies_dict()
        if adsite_studies is None:
            return _fail(
                "adsite_studies_not_initialized",
                "No initialized adsorption-site studies exist for "
                f"termination_index={termination_index}.",
                ValueError,
            )
        if site_index not in adsite_studies:
            return _fail(
                "adsite_study_not_initialized",
                "No initialized adsorption-site study exists for "
                f"site_index={site_index} on "
                f"termination_index={termination_index}.",
                ValueError,
            )
        ad_study = adsite_studies[site_index]

        # For O calc - make sure the relaxed O structure is not set:
        if job_type == "O_adsorption" and \
                ad_study.relaxed_O_atoms is not None:
            return _fail(
                "o_already_relaxed",
                "A relaxed O structure has already been accepted for "
                f"site_index={site_index} on "
                f"termination_index={termination_index}.",
                RuntimeError,
            )

        
        if job_type == "OH_adsorption":
            # If OH calc blocked before O calc, test O structure is set:
            if require_relaxed_o_for_oh and \
                    ad_study.relaxed_O_atoms is None:
                return _fail(
                    "oh_requires_relaxed_o",
                    "OH adsorption requires an accepted relaxed O "
                    "structure for this site before OH jobs can be "
                    "entered.",
                    RuntimeError,
                    candidate_id=candidate_id,
                    termination_index=termination_index,
                    site_index=site_index,
                    require_relaxed_o_for_oh=require_relaxed_o_for_oh,
                )

        return _pass(
            cand_row=cand_row,
            study_obj=study_obj,
            surface_study=surface_study,
            ad_study=ad_study,
            termination_index=int(termination_index),
            site_index=int(site_index),
        )

    return _fail(
        "unsupported_job_type",
        f"Unsupported job_type: {job_type}",
        ValueError,
    )


def validate_submit_process(
    rframe,
    process_id: int,
) -> ValidationResult:
    """
    Validate whether a process row can be submitted right now.
    """

    if isinstance(process_id, bool) or not isinstance(process_id, Integral):
        return _fail(
            "invalid_process_id_type",
            "process_id must be an integer.",
            ValueError,
        )
    pid = int(process_id)

    process_row_df = rframe.processes.df.loc[
        rframe.processes.df["process_id"] == pid
    ]
    if process_row_df.empty:
        return _fail(
            "process_not_found",
            f"No process found for process_id {pid}.",
            ValueError,
            process_id=pid,
        )

    process_row = process_row_df.iloc[0]
    if process_row["status"] != "un-submitted":
        return _fail(
            "process_not_unsubmitted",
            f"Process {pid} is not an un-submitted process and cannot "
            "be submitted.",
            RuntimeError,
            process_id=pid,
        )

    candidate_id = process_row["candidate_id"]
    try:
        study = rframe.candidates[candidate_id].df["study_obj"].values[0]
    except Exception:
        return _fail(
            "candidate_not_found",
            f"Candidate {candidate_id} does not exist.",
            KeyError,
            process_id=pid,
            candidate_id=candidate_id,
        )
    job_type = process_row["job_type"]
    calc_dir = Path(process_row["VASP_dir"])

    ctx = dict(
        process_id=pid,
        process_row=process_row,
        candidate_id=candidate_id,
        study=study,
        job_type=job_type,
        calc_dir=calc_dir,
    )

    if job_type == "bulk_relaxation":
        if study.get_bulk_atoms() is not None:
            return _fail(
                "bulk_already_set",
                f"Candidate {candidate_id} already has a relaxed "
                "structure.",
                RuntimeError,
                **ctx,
            )
        return _pass(**ctx)

    if job_type in ["surface_relaxation", "O_adsorption", "OH_adsorption"]:
        termination_index = process_row["termination_index"]
        surface_studies = study.get_surface_studies()
        if surface_studies is None:
            return _fail(
                "surface_studies_not_initialized",
                "Surface studies have not been initialized for "
                f"candidate {candidate_id}.",
                RuntimeError,
                **ctx,
            )
        if termination_index not in surface_studies:
            return _fail(
                "termination_not_initialized",
                f"Termination index {termination_index} has not been "
                f"initialized for candidate {candidate_id}.",
                ValueError,
                **ctx,
            )

        surface_study = surface_studies[termination_index]
        ctx["termination_index"] = int(termination_index)
        ctx["surface_study"] = surface_study

        if job_type == "surface_relaxation":
            if surface_study.get_relaxed_surface() is not None:
                return _fail(
                    "surface_already_relaxed",
                    f"Candidate {candidate_id} already has a relaxed "
                    "surface.",
                    RuntimeError,
                    **ctx,
                )
            return _pass(**ctx)

        site_index = process_row["site_index"]
        if pd.isna(site_index):
            return _fail(
                "site_index_missing",
                f"{job_type} requires a valid site_index in the process "
                "row.",
                ValueError,
                **ctx,
            )
        site_index = int(site_index)

        adsite_studies = surface_study.get_adsorption_site_studies_dict()
        if adsite_studies is None:
            return _fail(
                "adsite_studies_not_initialized",
                "No initialized adsorption-site studies exist for "
                f"termination_index={termination_index}.",
                ValueError,
                **ctx,
            )
        if site_index not in adsite_studies:
            return _fail(
                "adsite_study_not_initialized",
                "No initialized adsorption-site study exists for "
                f"site_index={site_index} on "
                f"termination_index={termination_index}.",
                ValueError,
                **ctx,
            )
        ad_study = adsite_studies[site_index]

        ctx["site_index"] = site_index
        ctx["ad_study"] = ad_study

        if job_type == "O_adsorption" and ad_study.relaxed_O_atoms is not None:
            return _fail(
                "o_already_relaxed",
                "A relaxed O structure has already been accepted for "
                "this adsorption site.",
                RuntimeError,
                **ctx,
            )

        if job_type == "OH_adsorption":
            oh_candidates = ad_study.get_OH_adsorbed_atoms()
            if oh_candidates is None:
                return _fail(
                    "oh_no_candidates",
                    "No OH initialization structures are available for "
                    "submission.",
                    RuntimeError,
                    **ctx,
                )
            try:
                n_oh_candidates = len(oh_candidates)
            except TypeError:
                return _fail(
                    "oh_candidates_invalid",
                    "OH initialization structures are in an invalid "
                    "format and cannot be indexed.",
                    RuntimeError,
                    **ctx,
                )
            if n_oh_candidates == 0:
                return _fail(
                    "oh_no_candidates",
                    "No OH initialization structures are available for "
                    "submission.",
                    RuntimeError,
                    **ctx,
                )

            process_note = process_row["processNote"]
            oh_init_index = _extract_oh_init_index(process_note)
            if oh_init_index is None:
                return _fail(
                    "oh_init_index_missing",
                    "OH_adsorption process row is missing OH_init_index "
                    "in processNote.",
                    ValueError,
                    **ctx,
                )
            if oh_init_index < 0 or oh_init_index >= n_oh_candidates:
                return _fail(
                    "oh_init_index_out_of_range",
                    f"OH_init_index={oh_init_index} is out of range "
                    f"(n={n_oh_candidates}).",
                    ValueError,
                    **ctx,
                )

            ctx["oh_init_index"] = int(oh_init_index)
            ctx["oh_candidates"] = oh_candidates

        return _pass(**ctx)

    return _fail(
        "unsupported_job_type",
        f"Unsupported job_type: {job_type}",
        ValueError,
        **ctx,
    )
