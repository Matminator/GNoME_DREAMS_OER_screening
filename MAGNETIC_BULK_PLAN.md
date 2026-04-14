# Plan: Magnetic Bulk Relaxation Pipeline

## Context

Every bulk relaxation goes through a mandatory magnetic enumeration:
1. `add_process(cid, "bulk_relaxation")` → enumerates N distinct magnetic orderings from the **initial (unrelaxed)** structure using pymatgen's `MagneticStructureEnumerator`, filtered to **same unit cell size only** → registers N processes
2. User submits those N processes to HPC
3. When ALL N are terminal: pick the lowest-energy successful one → **set relaxed bulk for the first time** + store winning MAGMOM
4. If all failed → no valid bulk; surface jobs are naturally blocked (`validate_enter_process` requires `get_bulk_atoms() is not None`)
5. MAGMOM propagates to surface and adsorption VASP inputs

No new DataFrame column. `processNote = "mag_config_index=N | ordering=FM"` (same pattern as `OH_init_index=N`).  
No new job type. Stays `"bulk_relaxation"` throughout.  
No initial non-magnetic bulk job. The enumerated batch IS the bulk relaxation.

---

## Verified Design Decisions

**`are_systems_equal` check in `set_relaxed_structure`** (atoms_sanity_checks.py:45-88):  
Only checks atom count, element symbols per index, and tags. Does NOT check positions. Since we filter magnetic configs to the same unit cell (same elements, same count, tags are all 0), the relaxed magnetic structure passes this check. We use `set_relaxed_structure` normally — no bypass needed.

**`AseAtomsAdaptor` does NOT transfer magmom** through conversion chains. Therefore magmom for surfaces must be read from `OER_surface_study.original_slab.site_properties["magmom"]` (pymatgen Slab preserved by SlabGenerator), not from the ASE Atoms chain.

**Existing validation works correctly for the batch pattern**: the N rows are created in one `enter_process` call (loop inside), bypassing per-row re-validation (same as OH batch). A second `enter_process("bulk_relaxation")` call is blocked by the duplicate check.

**`calculational_setup` mode behaviour** (`vasp_calculation.py:141-167`):
- `production_test` → uses `TestRelaxSet` (real VASP, minimal settings); `update_log` reads `vasprun.xml`
- `test` / `slurm_test` / `MLIP_test` → all contain `'test'` → create `fake_result.traj`
- `production` → uses RPBE sets; `update_log` reads `vasprun.xml`
- `potcar_spec = False if 'production' in self.mode else True` (existing pattern)

**MAGMOM via site properties, not INCAR injection**: `enumerate_magnetic_structures` writes magmoms into `site.properties["magmom"]` on every returned `Structure`. `VaspInputSet` (with `ISPIN=2` already in the RPBE sets) detects this and writes the `MAGMOM` INCAR tag automatically. **No separate magnetic VASP set classes are needed.** The existing `RPBE_relax_bulk_set` and `RPBE_relax_surface_set` are used directly — the only requirement is that the structure passed to them carries the magmoms in site properties.

**MAGMOM list to store**: use **post-relaxation VASP magmoms** (`atoms.get_magnetic_moments().tolist()` from the ASE Atoms returned by `read_vasp_results`). In test/slurm_test modes where no real VASP run occurs, fall back to the initial enumerator `config["magmom"]`. This distinction is handled in `update_log`; `oer_study.py` is agnostic and stores whatever list it receives.

**`sort_structure=False`**: already passed at every VASP write site. No change needed.

**Surface and adsorption MAGMOM**: before calling the existing `surface_relaxation()` (or adsorption relaxation), set `site.properties["magmom"]` on the pymatgen structure from `get_slab_magmom()`. For adsorption, append `0.0` for each adsorbate atom beyond the slab length. Then call the existing relaxation method — no new `magnetic_surface_relaxation()` method needed.

---

## Files to Modify / Create

| File | Change |
|---|---|
| ~~**New**: `vasp/magnetic_enumeration.py`~~ | **Done** — `enumerate_magnetic_structures` + `_extract_mag_config_index` |
| ~~`vasp/rpbe_relax_sets.py`~~ | **Not needed** — MAGMOM written via site properties, not INCAR injection |
| ~~`vasp/pre_defined_vasp_sets.py`~~ | **Not needed** — existing sets used directly |
| ~~`vasp/vasp_calculation.py`~~ | **Not needed** — no new methods; existing `bulk_relaxation` / `surface_relaxation` used |
| `vasp/job_handler.py` | Batch creation in `enter_process`; submit routing; `update_log` finalization; two helper methods |
| `oer/oer_study.py` | State fields + `store_mag_bulk_candidate`, `finalize_mag_bulk_batch`, `get_bulk_magmom` |
| `oer/oer_surface_study.py` | `get_slab_magmom()`, `get_relaxed_surface_magmom()`, `_relaxed_surface_magmom` field, magmom extraction in `set_relaxed_base_surface`, `require_magmom` param |
| `oer/oer_adsorption_site_study.py` | `get_O_calc_magmom()`, `get_OH_init_magmom()`, `_relaxed_O_magmom` field, magmom extraction in `set_relaxed_O_calc` |
| `explog/explog.py` | `magnetic_bulk_n_max: int = 4` init param |

---

## Implementation

### 1. `vasp/magnetic_enumeration.py` — DONE

File exists. Key guarantee: every returned `Structure` has `site.properties["magmom"]` set, and the `"magmom"` key in each config dict mirrors those values.

---

### 2. `oer/oer_study.py`

`AseAtomsAdaptor` is already imported at line 8.

Add to `OER_catalyst_study.__init__` (after line 118, alongside existing state variables):
```python
self._mag_bulk_candidates: dict[int, tuple] = {}  # {idx: (atoms, energy, magmom)}
self._mag_bulk_batch_finalized: bool = False
self._bulk_magmom: list[float] | None = None
```

Add methods:

```python
def store_mag_bulk_candidate(self, idx: int, atoms, energy: float,
                              magmom: list[float]) -> None:
    """Store a completed magnetic bulk result by config index."""
    self._mag_bulk_candidates[idx] = (atoms, energy, magmom)

def set_mag_bulk_batch_finalized(self, value: bool) -> None:
    self._mag_bulk_batch_finalized = value

def is_mag_bulk_batch_finalized(self) -> bool:
    return self._mag_bulk_batch_finalized

def finalize_mag_bulk_batch(self) -> bool:
    """
    Pick the lowest-energy magnetic candidate and set it as the
    relaxed bulk for the first time. Sets magmom site properties on
    bulk_structure so that SlabGenerator propagates them to slabs.
    Returns True on success, False if no completed candidates exist.
    """
    if not self._mag_bulk_candidates:
        return False

    best_idx = min(
        self._mag_bulk_candidates,
        key=lambda i: self._mag_bulk_candidates[i][1],
    )
    best_atoms, _, best_magmom = self._mag_bulk_candidates[best_idx]

    # Use the existing validated setter (handles consistency check):
    self.set_relaxed_structure(best_atoms)

    # Attach magmom site properties to bulk_structure so SlabGenerator
    # propagates them (AseAtomsAdaptor does not transfer magmom):
    for site, m in zip(self.bulk_structure.sites, best_magmom):
        site.properties["magmom"] = m
    self._bulk_magmom = list(best_magmom)
    self._mag_bulk_batch_finalized = True
    return True

def get_bulk_magmom(self) -> list[float] | None:
    """Per-atom initial magnetic moments for the winning bulk configuration."""
    return list(self._bulk_magmom) if self._bulk_magmom is not None else None
```

---

### 3. `oer/oer_surface_study.py` — DONE

**New state field** (in `__init__`, after `self.relaxed_surface_energy`):
```python
self._relaxed_surface_magmom: list[float] | None = None
```

**`set_relaxed_base_surface` signature** — new `require_magmom: bool = False` parameter.
Inside the success branch, before `self.base_surface_relaxed = surface.copy()`:
```python
# Extract relaxed magmoms BEFORE copy — atoms.copy() does not preserve the calculator.
if surface.calc is not None:
    moms = surface.get_magnetic_moments()
    if np.any(np.abs(moms) > 0.01):
        self._relaxed_surface_magmom = moms.tolist()

if require_magmom and self._relaxed_surface_magmom is None:
    raise ValueError(
        "require_magmom=True but no non-zero magnetic moments "
        "could be extracted from the relaxed surface."
    )
```
If no calculator is attached (test / fake-result modes), `_relaxed_surface_magmom` stays `None`.
The caller decides the fallback via `require_magmom` or by calling `get_slab_magmom()` explicitly.

**New methods:**
```python
def get_slab_magmom(self) -> list[float] | None:
    """Initial magmoms from original_slab.site_properties["magmom"] (from SlabGenerator).
    Returns None if the key is absent."""
    magmom = self.original_slab.site_properties.get("magmom")
    if magmom is None:
        return None
    return list(magmom)

def get_relaxed_surface_magmom(self) -> list[float] | None:
    """Post-VASP magmoms extracted in set_relaxed_base_surface.
    Returns None if not yet set or no calculator was attached.
    Use get_slab_magmom() for the initial (non-relaxed) magmoms."""
    if self._relaxed_surface_magmom is not None:
        return list(self._relaxed_surface_magmom)
    return None
```

**In `job_handler.py` (4e)** — use both getters:
- For **surface_relaxation** VASP setup: inject `get_slab_magmom()` (pre-relaxation)
- For **O/OH adsorption** VASP setup: prefer `get_relaxed_surface_magmom()`; fall back to `get_slab_magmom()`

---

### 3b. `oer/oer_adsorption_site_study.py`

**New state field** (in `__init__`):
```python
self._relaxed_O_magmom: list[float] | None = None
```

**`set_relaxed_O_calc`** — extract magmoms before `.copy()`:
```python
if atoms.calc is not None:
    moms = atoms.get_magnetic_moments()
    if np.any(np.abs(moms) > 0.01):
        self._relaxed_O_magmom = moms.tolist()
self.relaxed_O_atoms = atoms.copy()
```

**New methods:**
```python
def get_O_calc_magmom(self) -> list[float] | None:
    """Post-VASP magmoms from the O adsorption calculation.
    Returns None if set_relaxed_O_calc has not been called or no calculator was attached."""
    if self._relaxed_O_magmom is not None:
        return list(self._relaxed_O_magmom)
    return None

def get_OH_init_magmom(self, surface_magmom: list[float] | None) -> list[float] | None:
    """
    Initial magmoms for the OH adsorption calculation.

    For on-top sites: uses relaxed O magmoms (self._relaxed_O_magmom) as the base,
    with [0.0] appended for the H atom. Falls back to surface_magmom + [0.0, 0.0]
    if O magmoms are unavailable.

    For lattice-O sites: uses surface_magmom (the relaxed surface without the O atom)
    with [0.0] appended for the H atom. The O atom is already removed from the base.

    Args:
        surface_magmom: per-atom magmoms for the surface used as the OH base
                        (get_relaxed_surface_magmom() or get_slab_magmom() from surface_study).
                        May be None for non-magnetic systems.

    Returns None if no magnetic data is available for either site type.
    """
    if self.site_type == 'on-top':
        base = self._relaxed_O_magmom
        if base is None:
            if surface_magmom is None:
                return None
            # Fallback: surface magmoms + 0 for O + 0 for H
            return list(surface_magmom) + [0.0, 0.0]
        # O calc base already includes the O adsorbate; append 0 for H
        return list(base) + [0.0]
    elif self.site_type == 'lattice O':
        if surface_magmom is None:
            return None
        # Lattice-O OH base is the surface with O removed; append 0 for H
        return list(surface_magmom) + [0.0]
    return None
```

---

### 4. `vasp/job_handler.py`

#### 4a. Imports (top of file)

```python
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures, _extract_mag_config_index)
```

#### 4b. `__init__` — add parameter

```python
def __init__(self, ..., magnetic_bulk_n_max: int = 4, ...) -> None:
    ...
    self.magnetic_bulk_n_max = magnetic_bulk_n_max
```

#### 4c. `enter_process` — add bulk batch branch

After the OH batch block and before the final `return _add_single_process(note)`, add:

```python
if job_type == "bulk_relaxation":
    study_obj = validation_result.context.get("study_obj")
    configs = enumerate_magnetic_structures(
        study_obj.get_init_structure(), self.magnetic_bulk_n_max
    )
    # Cache on study object so submit step can retrieve without re-enumerating:
    study_obj._mag_bulk_configs = configs
    study_obj.set_mag_bulk_batch_finalized(False)
    process_ids = []
    for i, config in enumerate(configs):
        process_note = f"mag_config_index={i} | ordering={config['ordering']}"
        if note is not None:
            process_note = f"{note} | {process_note}"
        process_ids.append(_add_single_process(process_note))
    return process_ids   # list[int], consistent with OH return type
```

#### 4d. `submit_vasp_calculation` — replace `bulk_relaxation` branch

> **Note**: `job_handler.py` currently calls `study.set_relaxed_structure(atoms)` at lines 944
> and 1108 (test and production `update_log` blocks). These are the old non-magnetic calls.
> Both must be **removed** (not just renamed) when implementing 4f — finalization is now
> handled exclusively by `study.finalize_mag_bulk_batch()` via the `_finalize_mag_bulk_batch`
> helper. `_set_relaxed_structure` is private and must not be called from outside the class.

```python
if job_type == 'bulk_relaxation':
    if study.get_bulk_atoms() is not None:
        raise RuntimeError(
            f"Candidate {candidate_id} already has a relaxed bulk "
            "structure set. Cannot submit bulk_relaxation again."
        )
    process_note = process_row['processNote'].values[0]
    mag_idx = _extract_mag_config_index(process_note)
    if mag_idx is None:
        raise RuntimeError(
            f"Process {process_id}: bulk_relaxation is missing "
            "mag_config_index in processNote."
        )
    # Retrieve cached configs; re-enumerate if lost (e.g. after unpickling):
    configs = getattr(study, '_mag_bulk_configs', None)
    if configs is None or mag_idx >= len(configs):
        configs = enumerate_magnetic_structures(
            study.get_init_structure(), self.magnetic_bulk_n_max
        )
        study._mag_bulk_configs = configs
    config = configs[mag_idx]
    # config["structure"] already has site_properties["magmom"] set;
    # VaspInputSet writes MAGMOM automatically — no separate magnetic set needed.
    self.calc_setup.bulk_relaxation(
        structure=config["structure"], directory=calc_dir
    )
```

#### 4e. `submit_vasp_calculation` — surface/adsorption MAGMOM injection

For `surface_relaxation`, after `init_atoms = surface_study.get_non_relaxed_surface()`:

```python
init_atoms = surface_study.get_non_relaxed_surface()
init_structure = AseAtomsAdaptor.get_structure(init_atoms)
slab_magmom = surface_study.get_slab_magmom()
if slab_magmom is not None:
    for site, m in zip(init_structure.sites, slab_magmom):
        site.properties["magmom"] = m
self.calc_setup.surface_relaxation(init_structure, calc_dir)
```

For `O_adsorption` and `OH_adsorption`, after `init_structure = AseAtomsAdaptor.get_structure(init_atoms)`:

```python
slab_magmom = surface_study.get_slab_magmom()
if slab_magmom is not None:
    n_slab = len(slab_magmom)
    n_total = len(init_structure)
    full_magmom = slab_magmom + [0.0] * (n_total - n_slab)
    for site, m in zip(init_structure.sites, full_magmom):
        site.properties["magmom"] = m
self.calc_setup.surface_relaxation(init_structure, calc_dir)
```

#### 4f. `update_log` — bulk_relaxation result handling

**Helper to retrieve magmom for a completed process:**

```python
def _get_config_magmom(self, study, mag_idx: int) -> list[float]:
    """Retrieve the enumerator magmom for config index mag_idx."""
    configs = getattr(study, '_mag_bulk_configs', None)
    if configs is None or mag_idx >= len(configs):
        configs = enumerate_magnetic_structures(
            study.get_init_structure(), self.magnetic_bulk_n_max
        )
        study._mag_bulk_configs = configs
    if mag_idx < len(configs):
        return configs[mag_idx]["magmom"]
    return [0.0] * len(study.get_init_structure())
```

**In the test/slurm_test block**, replace the `bulk_relaxation` handler:

```python
if job_type == 'bulk_relaxation':
    mag_idx = _extract_mag_config_index(pdf_row['processNote'])
    if mag_idx is not None:
        magmom = self._get_config_magmom(study, mag_idx)
        study.store_mag_bulk_candidate(mag_idx, atoms, E, magmom)
    # status set to 'completed' below (existing code path)
```

After `self.rframe.processes.df.at[idx, 'status'] = 'completed'` AND after failed status setting, add:

```python
if job_type == 'bulk_relaxation':
    if self._all_magnetic_bulk_terminal(candt_id):
        self._finalize_mag_bulk_batch(candt_id)
```

**In the production block**, for completed `bulk_relaxation`:

```python
if job_type == 'bulk_relaxation':
    mag_idx = _extract_mag_config_index(pdf_row['processNote'])
    if mag_idx is not None:
        energy = atoms.get_potential_energy()
        magmom = self._get_config_magmom(study, mag_idx)
        study.store_mag_bulk_candidate(mag_idx, atoms, energy, magmom)
    # (do NOT call set_relaxed_structure here — handled by finalize)
```

For FAILED bulk_relaxation in production block: just mark failed (no store). Then trigger finalization as in test block.

#### 4g. New helper methods

```python
def _all_magnetic_bulk_terminal(self, candidate_id: str) -> bool:
    """True when every bulk_relaxation process for this candidate is terminal."""
    pdf = self.rframe.processes.df
    bulk_rows = pdf[
        (pdf["candidate_id"] == candidate_id) &
        (pdf["job_type"] == "bulk_relaxation")
    ]
    if bulk_rows.empty:
        return False
    return bulk_rows["status"].apply(self._is_terminal_process_status).all()


def _finalize_mag_bulk_batch(self, candidate_id: str) -> bool:
    """Pick lowest-energy completed bulk candidate and set on study."""
    study = self.rframe.candidates[candidate_id].df["study_obj"].values[0]
    return study.finalize_mag_bulk_batch()
```

---

### 5. `explog/explog.py`

Add `magnetic_bulk_n_max: int = 4` to `Explog.init()` and pass to `Job_handler.__init__`.

---

## Safety Checklist

| Concern | Status |
|---|---|
| `are_systems_equal` check passes for magnetic relaxed atoms | ✓ Only checks symbols + tags, not positions |
| No direct attribute bypass — uses `set_relaxed_structure` | ✓ |
| `sort_structure=False` preserved | ✓ Already passed at every VASP write site; no change needed |
| MAGMOM written via site properties, not INCAR injection | ✓ VaspInputSet handles automatically with existing sets |
| No new VASP set classes or `vasp_calculation` methods | ✓ Existing `bulk_relaxation` / `surface_relaxation` used directly |
| Energy extraction correct per mode | ✓ test: `atoms[0].charge`; others: `get_potential_energy()` |
| `production_test` uses `TestRelaxSet` | ✓ Handled by existing `bulk_relaxation` method |
| Failed bulk jobs handled (no `store_mag_bulk_candidate` call) | ✓ |
| Finalization triggered for BOTH completed AND failed terminal jobs | ✓ |
| MAGMOM from enumerator config (not ASE atoms) | ✓ Reliable across all modes |
| Slab MAGMOM from `original_slab` (not ASE conversion chain) | ✓ `get_slab_magmom()` reads `site_properties["magmom"]` |
| Relaxed surface MAGMOM extracted before `atoms.copy()` | ✓ `surface.calc is not None` guard; `require_magmom` param for strict mode |
| Relaxed MAGMOM fallback chain is explicit (caller-controlled, not hidden) | ✓ `get_relaxed_surface_magmom()` returns `None`; caller falls back to `get_slab_magmom()` |
| On-top OH: O magmoms carry over to OH (OH ≈ O + H) | ✓ `get_OH_init_magmom` uses `_relaxed_O_magmom` + `[0.0]` for H |
| Lattice-O OH: surface magmoms used (no separate O adsorbate added) | ✓ `get_OH_init_magmom` uses `surface_magmom` + `[0.0]` for H |
| Adsorption MAGMOM: slab atoms + `[0.0]` for adsorbate | ✓ Site properties set inline before calling existing relaxation |
| `_mag_bulk_configs` re-enumerated after unpickling | ✓ Fallback in `submit_vasp_calculation` and `_get_config_magmom` |
| N=1 and N=0 (NM fallback) handled | ✓ |
| Second `enter_process("bulk_relaxation")` correctly blocked | ✓ Duplicate check finds existing rows |

---

## Verification

1. **Unit tests** — `tests/`:
   - `_extract_mag_config_index`: parses correctly, returns None for OH notes — **Done** (`test_magnetic_enumeration.py`)
   - `enumerate_magnetic_structures`: supercell filter; NM fallback; n_max cap — **Done**
   - `finalize_mag_bulk_batch`: picks minimum energy; False when empty; magmom set on `bulk_structure` site properties

2. **Integration test** (`tests/production_test_mini_workflow.py`):
   - `add_process(cid, "bulk_relaxation")` → `list[int]` returned; each has `mag_config_index=N` in processNote
   - Submit all → `update_log` loop → all terminal
   - `study.get_bulk_atoms()` is set; `study.get_bulk_magmom()` is set
   - `study.bulk_structure.site_properties["magmom"]` is set
   - `initialize_oer_surface_study` → `surface_study.get_slab_magmom()` is not None
   - Surface submission: `init_structure` has `site_properties["magmom"]` set before calling `surface_relaxation`
   - INCAR in surface VASP dir contains MAGMOM

3. Run full existing test suite; confirm no regressions.
