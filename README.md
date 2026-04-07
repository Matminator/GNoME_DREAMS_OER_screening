# GNoME_DREAMS_OER_screening

## v0.5.0 Update Notes

This release adds structured, reusable process validation for OER
workflow orchestration and improves process-state robustness.

- Added `vasp.job_validation` with `ValidationResult`,
  `validate_enter_process`, and `validate_submit_process`.
- Added non-mutating pre-check API on `Explog`:
  `Explog.can_add_process(...)`.
- Standardized process ids to integer semantics in the process table
  and process-related APIs.
- Improved validation stability by returning structured failures for
  edge cases that previously could raise unexpectedly (for example
  missing surface/ad-site study state or malformed `OH_init_index`).
- Added dedicated mini test coverage for validator behavior in
  `tests/test_validate_submit_process.py`.

### Behavioral Notes

- Duplicate-process checks treat only `"failed"` as retryable by
  default.
- `"unrecoverable"` statuses are treated as non-failed terminal
  outcomes and therefore block duplicate re-entry under default rules.
