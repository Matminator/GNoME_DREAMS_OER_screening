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

---

## 2. Potential Bugs Found in Code Review

### 2a — `all()` on empty pandas Series returns True (job_handler.py)
In `_refresh_oh_batch_finalization_state` the terminal-status check is
`oh_rows["status"].apply(...).all()`. If `oh_rows` is empty (no OH jobs submitted
yet), `.all()` returns `True` on an empty Series, which would mark the batch finalized
before any jobs exist. The code adds a `not oh_rows.empty` guard before the call, but
the guard and the call should be reviewed together to confirm the guard is on the correct
branch and covers all code paths.

- **File**: `src/gnome_dreams_oer_screening/vasp/job_handler.py` (~line 933)
- **Risk level**: Medium — could mark OH batch complete prematurely.
- **Missing test**: No test verifies that zero OH rows → `oh_batch_finalized` stays False.

### 2b — NaN magmoms propagate silently through the surface pipeline
`finalize_mag_bulk_batch` stores `list(best_magmom)` without checking for NaN values.
If VASP returns NaN magnetic moments (failed SCF, NELM exceeded), those NaNs flow into
slab site properties and eventually into INCAR MAGMOM tags.

- **File**: `src/gnome_dreams_oer_screening/oer/oer_study.py` (~line 257)
- **Risk level**: Medium — NaN in MAGMOM causes VASP to use defaults, producing
  silently wrong magnetic initial conditions.
- **Missing test**: No test checks that `finalize_mag_bulk_batch` raises or warns when
  magmoms contain NaN.

### 2c — Process ID generated from table length, not a sequence
`new_p_id = len(self.rframe.processes.df)` will collide if a row is ever deleted
from the processes table (length shrinks, but prior IDs are still in use). Even if
deletion is not currently exposed, this is a fragile ID scheme.

- **File**: `src/gnome_dreams_oer_screening/vasp/job_handler.py` (~line 320)
- **Risk level**: Low currently (no delete path exposed), but worth documenting.

---

## 3. Missing Test Coverage

### 3a — Overpotential formula never verified against a reference value
`compute_overpotential()` (or equivalent in `oer_surface_study.py`) is the primary
output of the entire pipeline. There is no test that supplies known G(OH), G(O),
G(OOH) values and checks the resulting overpotential against a hand-computed reference.

- **File**: `src/gnome_dreams_oer_screening/oer/oer_surface_study.py`
- **Risk level**: High — a sign error or wrong coefficient would silently produce
  plausible-looking but wrong numbers for every candidate.
- **Suggested test**: Unit test with G(OH)=1.23, G(O)=2.46, G(OOH)=3.69 (ideal
  scaling) → expected overpotential ≈ 0 V.

### 3b — Non-magnetic material end-to-end path not tested
`enumerate_magnetic_structures` catches `"Not detected as magnetic"` and returns a
single NM fallback entry. No test exercises this fallback through the full
`add_process("bulk_relaxation")` → `update_log()` → `finalize_mag_bulk_batch()` chain.

- **Files**: `src/gnome_dreams_oer_screening/vasp/magnetic_enumeration.py` (~line 94),
  `src/gnome_dreams_oer_screening/vasp/job_handler.py`
- **Risk level**: Medium — non-magnetic oxides (e.g. IrO₂, RuO₂) are prime OER
  candidates; a silent failure in this path would exclude them from screening.

### 3c — `set_relaxed_base_surface` lock-out path not tested
When `are_all_atoms_connected()` returns False, `set_relaxed_base_surface` sets
`study_locked = True`. No test verifies that subsequent calls to
`determine_adsorption_sites()` then raise `SurfaceStudyLockedError`, nor that an
unlocked surface after a fragmented one is correctly rejected (line 167 raises
`ValueError`).

- **File**: `src/gnome_dreams_oer_screening/oer/oer_surface_study.py` (~lines 151–185)
- **Risk level**: Medium — if the lock silently fails, adsorption sites are determined
  on a broken slab and produce invalid energies.

### 3d — `reject_if_failed_exists` policy not tested for adsorption processes
`enter_process` accepts a `reject_if_failed_exists` flag (job_validation.py ~line 195)
that controls whether a previously-failed adsorption job blocks re-submission.
Neither branch (allow retry / reject retry) has a dedicated test.

- **File**: `src/gnome_dreams_oer_screening/vasp/job_validation.py` (~line 195)
- **Risk level**: Medium — incorrect policy logic could prevent legitimate retries or
  allow infinite retry loops on a persistently failing site.

### 3e — Magmom propagation to adsorption site initial conditions not tested
Bulk magmoms propagate: `bulk_structure` → `slab.site_properties["magmom"]` →
`get_slab_magmom()` → `get_O_init_magmom()` / `get_OH_init_magmom()`. No test
verifies that the magmoms reaching the adsorption-site INCAR are correct after this
chain. The existing `test_oer_surface_study.py` tests check propagation only up to the
slab level.

- **Files**: `src/gnome_dreams_oer_screening/oer/oer_surface_study.py`,
  `src/gnome_dreams_oer_screening/oer/oer_adsorption_site_study.py`
- **Risk level**: Medium — wrong initial magmoms on adsorbates affect SCF convergence
  and local minima without causing a hard failure.

### 3f — Termination ranking for a stoichiometrically-failed slab (NaN score)
`_build_surfaces` returns `[]` when slab generation fails (timeout, layer limit, or
stoichiometry check). The NaN Score row is written to `termination_ranking_df` but no
test verifies: (a) NaN rows are excluded from surface initialization, (b) a mix of
valid and NaN-scored terminations is handled correctly by `initialize_oer_surface_study`.

- **File**: `src/gnome_dreams_oer_screening/oer/oer_study.py` (~lines 542–550)
- **Risk level**: Low-Medium — passing a NaN-scored termination index to
  `initialize_oer_surface_study` may produce a study with a None slab.

### 3g — Pickle round-trip produces identical downstream results (see also §1b)
No test creates an Explog, pickles it, reloads it, continues the workflow, and asserts
that final OER metrics match those from an uninterrupted run. This is distinct from §1b
(which focuses on attribute layout) — this gap concerns numerical reproducibility.

- **Files**: `src/gnome_dreams_oer_screening/explog/explog.py`
- **Risk level**: Medium — subtle float representation differences or pandas dtype
  changes after round-trip could alter winner selection in `finalize_mag_bulk_batch`.

