# Testing Gaps — Production Readiness

Gaps identified through a structured review of what is needed before running the
screening pipeline unattended on HPC with hundreds of GNoME candidates.
Primary concern: **silent wrong results** (not hard crashes).

---

## 1. Persistence / State Recovery

### 1a — Interrupted write corrupts the pickle
The full Explog state is saved as a pickle. If the SLURM node is killed mid-write,
the file is left in a corrupt or incomplete state. On the next run, loading the pickle
fails (or silently loads partial data), and **all progress up to that point is lost**.

- **Risk level**: High — unattended HPC jobs are routinely killed by walltime limits.
- **Missing test**: No test verifies recovery from a corrupt or missing pickle.
- **Possible mitigation to test**: Write to a temp file and rename atomically; test that
  a run can continue from the last good checkpoint after a simulated corrupt write.

### 1b — Code changes between runs break unpickling silently
If a class attribute is added, removed, or renamed on `OER_catalyst_study`,
`OER_surface_study`, `OER_adsorption_site_study`, etc. between the run that wrote the
pickle and the run that reads it, the unpickled objects carry the old attribute layout.
This either raises `AttributeError` (detectable) or silently uses a stale value
(invisible and dangerous).

- **Risk level**: Medium-High — likely during active development.
- **Missing test**: No round-trip test that verifies state survives a pickle/unpickle cycle
  and produces identical downstream results.

### 1c — Resume scenario not tested end-to-end
When a run is interrupted mid-batch (e.g. 50 of 200 bulk relaxations done) and reloaded:

1. Are the 50 completed results preserved and not re-computed?
2. Do the 150 pending processes resume correctly?
3. Is the candidate→process→result mapping intact after the round-trip?

- **Missing test**: No test simulates a partial run → pickle → reload → continue and
  verifies that the resumed run produces the same final results as an uninterrupted run.

---

## 2. (To be filled in — interview in progress)

