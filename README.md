# GNoME_DREAMS_OER_screening

## v0.5.2 Update Notes

This release streamlines process-entry validation configuration and
clarifies policy ownership between `Explog` and `Job_handler`.

- `Job_handler` now owns default process-entry policy settings:
  `reject_if_failed_exists` and `require_relaxed_o_for_oh`.
- `Explog.init(...)` exposes the same defaults and forwards them to
  `Job_handler`.
- `Explog.add_process(...)` and `Explog.can_add_process(...)` now
  support per-call overrides using `None` as "use handler default".
- `Job_handler.enter_process(...)` now resolves effective policy from
  defaults + overrides and applies `validate_enter_process(...)`
  directly as the shared pre-check.
- Legacy inline `enter_process` sanity checks are deprecated and kept
  as commented reference; validator-based checks are the active path.
- Removed the older OH-specific duplicate policy flag and consolidated
  duplicate handling through the shared validator path.
- Updated and expanded tests for validator behavior and job-handler
  integration.

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
