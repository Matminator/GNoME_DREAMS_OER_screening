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

**MAGMOM list to store**: use the enumerator-assigned `config["magmom"]` (initial moments), NOT the converged VASP moments. This is reliable across all modes, is always available, and gives the right magnetic ordering for downstream initialisation.

**`sort_structure=False`**: already passed at every VASP write site. New magnetic sets follow the same pattern — pass through `**kwargs`, not hardcoded in `super()`.

---

## Files to Modify / Create

| File | Change |
|---|---|
| **New**: `vasp/magnetic_enumeration.py` | `enumerate_magnetic_structures` + `_extract_mag_config_index` |
| `vasp/rpbe_relax_sets.py` | `get_RPBE_magnetic_bulk_settings(magmom)` and `get_RPBE_magnetic_surface_settings(magmom)` |
| `vasp/pre_defined_vasp_sets.py` | `RPBE_relax_magnetic_bulk_set` and `RPBE_relax_magnetic_surface_set` |
| `vasp/vasp_calculation.py` | `calculational_setup.magnetic_bulk_relaxation()` and `magnetic_surface_relaxation()` |
| `vasp/job_handler.py` | Batch creation in `enter_process`; submit routing; `update_log` finalization; two helper methods |
| `oer/oer_study.py` | State fields + `store_mag_bulk_candidate`, `finalize_mag_bulk_batch`, `get_bulk_magmom` |
| `oer/oer_surface_study.py` | `get_slab_magmom()` |
| `explog/explog.py` | `magnetic_bulk_n_max: int = 4` init param |

---

## Implementation

### 1. `vasp/magnetic_enumeration.py` (new file)

```python
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator


def _ordering_priority(magmoms: list[float]) -> int:
    """
    Return a sort key so results are ordered FM → AFM/FiM → NM.
    FM first ensures the simplest, most common ordering is always included
    when n_max is small. AFM orderings are frequently the ground state in
    TM oxides and come second.
    """
    nonzero = [m for m in magmoms if abs(m) > 0.1]
    if not nonzero:
        return 2  # NM
    if all(m > 0 for m in nonzero) or all(m < 0 for m in nonzero):
        return 0  # FM
    return 1  # AFM or FiM (mixed signs)


def enumerate_magnetic_structures(structure, n_max: int = 4) -> list[dict]:
    """
    Returns up to n_max dicts: {"structure": Structure, "magmom": list[float], "ordering": str}

    Filters out orderings that require a supercell (different atom count from input).
    Sorts results FM → AFM/FiM → NM before applying the n_max cap, so the most
    physically relevant orderings are always selected regardless of enumlib's
    internal enumeration order.
    Falls back to a single non-magnetic entry if enumeration yields nothing.
    """
    n_input = len(structure)
    try:
        enumerator = MagneticStructureEnumerator(structure, automatic=True)
        candidates = []
        for mag_struct, origin in zip(
            enumerator.ordered_structures, enumerator.ordered_structure_origins
        ):
            if len(mag_struct) != n_input:
                continue  # discard supercell orderings
            magmoms = [site.properties.get("magmom", 0) for site in mag_struct.sites]
            candidates.append({
                "structure": mag_struct,
                "magmom": magmoms,
                "ordering": str(origin),
            })
    except Exception:
        candidates = []

    if not candidates:
        # Non-magnetic material or enumlib unavailable — one NM fallback job
        return [{
            "structure": structure,
            "magmom": [0] * n_input,
            "ordering": "NM",
        }]

    # Sort FM first, then AFM/FiM, preserving original order within each group:
    candidates.sort(key=lambda c: _ordering_priority(c["magmom"]))
    return candidates[:n_max]


def _extract_mag_config_index(process_note: object) -> int | None:
    """Parse 'mag_config_index=N' from a processNote string. Returns None if absent."""
    if not isinstance(process_note, str):
        return None
    token = "mag_config_index="
    if token not in process_note:
        return None
    tail = process_note.split(token, 1)[1].split("|", 1)[0].strip()
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None
```

### 2. `vasp/rpbe_relax_sets.py`

Append two functions (deep copy avoids mutating the shared class-level dict):

```python
def get_RPBE_magnetic_bulk_settings(magmom: list[float]) -> dict:
    import copy
    settings = copy.deepcopy(get_RPBE_bulk_settings())
    settings["INCAR"]["MAGMOM"] = magmom
    return settings


def get_RPBE_magnetic_surface_settings(magmom: list[float]) -> dict:
    import copy
    settings = copy.deepcopy(get_RPBE_surface_settings())
    settings["INCAR"]["MAGMOM"] = magmom
    return settings
```

### 3. `vasp/pre_defined_vasp_sets.py`

Add imports at top:
```python
from gnome_dreams_oer_screening.vasp.rpbe_relax_sets import (
    get_RPBE_bulk_settings, get_RPBE_surface_settings,
    get_RPBE_magnetic_bulk_settings, get_RPBE_magnetic_surface_settings,
)
```

Add two new classes. Each instance gets its own fresh CONFIG dict (avoiding the shared-mutation issue in the existing sets):

```python
@dataclass
class RPBE_relax_magnetic_bulk_set(VaspInputSet):
    CONFIG = get_RPBE_bulk_settings()   # class-level placeholder; shadowed at instance level

    def __init__(self, structure, magmom: list[float], **kwargs):
        # Instance-level dict — does not mutate the class-level CONFIG
        self.CONFIG = get_RPBE_magnetic_bulk_settings(magmom)
        self.CONFIG["KPOINTS"] = heuristic_compute_kpoints(structure, structure_type='bulk')
        super().__init__(structure=structure, **kwargs)
        # sort_structure=False is passed via **kwargs from the call site


@dataclass
class RPBE_relax_magnetic_surface_set(VaspInputSet):
    CONFIG = get_RPBE_surface_settings()   # class-level placeholder

    def __init__(self, structure, magmom: list[float], **kwargs):
        self.CONFIG = get_RPBE_magnetic_surface_settings(magmom)
        self.CONFIG["KPOINTS"] = heuristic_compute_kpoints(structure, structure_type='surface')
        super().__init__(structure=structure, **kwargs)
```

### 4. `vasp/vasp_calculation.py`

Update import at line 32:
```python
from gnome_dreams_oer_screening.vasp.pre_defined_vasp_sets import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set, TestRelaxSet,
    RPBE_relax_magnetic_bulk_set, RPBE_relax_magnetic_surface_set)
```

Add two methods to `calculational_setup`. Mirror the existing `bulk_relaxation` / `surface_relaxation` pattern exactly — including the `production_test` → `TestRelaxSet` fallback:

```python
def magnetic_bulk_relaxation(self, structure: Structure,
                              directory: Path, magmom: list[float]) -> None:
    if self.mode == 'production_test':
        vasp_set = TestRelaxSet(structure=structure, sort_structure=False)
    else:
        vasp_set = RPBE_relax_magnetic_bulk_set(
            structure=structure, magmom=magmom, sort_structure=False)
    vasp_set.write_input(directory, potcar_spec=False if 'production' in self.mode else True)

    if 'test' in self.mode:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        fake_energy = len(atoms)
        atoms[0].charge = fake_energy
        save_fake_results_to_traj(directory, atoms)


def magnetic_surface_relaxation(self, structure: Structure,
                                 directory: Path, magmom: list[float]) -> None:
    if self.mode == 'production_test':
        vasp_set = TestRelaxSet(structure=structure, sort_structure=False)
    else:
        vasp_set = RPBE_relax_magnetic_surface_set(
            structure=structure, magmom=magmom, sort_structure=False)
    vasp_set.write_input(directory, potcar_spec=False if 'production' in self.mode else True)

    if 'test' in self.mode:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Mirror the surface fake-energy formula from surface_relaxation:
        K_O = (-14.217) - 0.070 - (-6.77)
        K_OH = 0.361 + (-6.77)/2 - (-14.217)
        x = 2.46 + K_O
        y = 2*x + K_OH - 1.23
        import numpy as np
        random_factor = 0 if 'dummy' in self.mode else np.random.normal(0, 1)
        fake_energy = (x * len(atoms) - y + random_factor
                       if 'H' in atoms.symbols
                       else x * len(atoms) + random_factor)
        atoms[0].charge = fake_energy
        save_fake_results_to_traj(directory, atoms)
```

### 5. `oer/oer_study.py`

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
    relaxed bulk for the first time.  Sets magmom site properties on
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

### 6. `oer/oer_surface_study.py`

Add one method (reads from `self.original_slab` — pymatgen Slab with site properties intact):

```python
def get_slab_magmom(self) -> list[float] | None:
    """
    Returns per-atom magnetic moments from the original (non-relaxed) slab,
    or None if no magmom site properties are set or all are zero.
    """
    magmom = self.original_slab.site_properties.get("magmom")
    if magmom is None or all(m == 0 for m in magmom):
        return None
    return list(magmom)
```

### 7. `vasp/job_handler.py`

#### 7a. Imports (top of file)

```python
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures, _extract_mag_config_index)
```

#### 7b. `__init__` — add parameter

```python
def __init__(self, ..., magnetic_bulk_n_max: int = 4, ...) -> None:
    ...
    self.magnetic_bulk_n_max = magnetic_bulk_n_max
```

#### 7c. `enter_process` — add bulk batch branch

After the OH batch block (line 336-348) and before the final `return _add_single_process(note)`, add:

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

#### 7d. `submit_vasp_calculation` — replace `bulk_relaxation` branch (lines 403-413)

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
    init_structure = config["structure"]
    self.calc_setup.magnetic_bulk_relaxation(
        structure=init_structure, directory=calc_dir, magmom=config["magmom"]
    )
```

#### 7e. `submit_vasp_calculation` — surface/adsorption MAGMOM injection

For `surface_relaxation` (after existing `init_atoms = surface_study.get_non_relaxed_surface()` line):

```python
init_atoms = surface_study.get_non_relaxed_surface()
init_structure = AseAtomsAdaptor.get_structure(init_atoms)
slab_magmom = surface_study.get_slab_magmom()
if slab_magmom is not None:
    self.calc_setup.magnetic_surface_relaxation(
        init_structure, calc_dir, slab_magmom)
else:
    self.calc_setup.surface_relaxation(init_structure, calc_dir)
```

For `O_adsorption` and `OH_adsorption` (after existing `init_structure = AseAtomsAdaptor.get_structure(init_atoms)` lines):

```python
slab_magmom = surface_study.get_slab_magmom()
if slab_magmom is not None:
    n_slab = len(slab_magmom)
    n_total = len(init_structure)
    full_magmom = slab_magmom + [0.0] * (n_total - n_slab)
    self.calc_setup.magnetic_surface_relaxation(
        init_structure, calc_dir, full_magmom)
else:
    self.calc_setup.surface_relaxation(init_structure, calc_dir)
```

#### 7f. `update_log` — bulk_relaxation result handling

**Helper to retrieve magmom for a completed process** (used inside update_log):

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

**In the test/slurm_test block** (lines 926-995), replace the `bulk_relaxation` handler:

```python
# Energy E = atoms[0].charge is already extracted unconditionally at line 932

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

**In the production block** (lines 997+), for completed bulk_relaxation:

```python
if job_type == 'bulk_relaxation':
    mag_idx = _extract_mag_config_index(pdf_row['processNote'])
    if mag_idx is not None:
        energy = atoms.get_potential_energy()
        magmom = self._get_config_magmom(study, mag_idx)
        study.store_mag_bulk_candidate(mag_idx, atoms, energy, magmom)
    # (do NOT call set_relaxed_structure here — handled by finalize)
```

For FAILED bulk_relaxation in production block: just mark failed (no store). Then:

```python
if job_type == 'bulk_relaxation':
    if self._all_magnetic_bulk_terminal(candt_id):
        self._finalize_mag_bulk_batch(candt_id)
```

#### 7g. New helper methods

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

### 8. `explog/explog.py`

Add `magnetic_bulk_n_max: int = 4` to `Explog.init()` and pass to `Job_handler.__init__`.

---

## Safety Checklist

| Concern | Status |
|---|---|
| `are_systems_equal` check passes for magnetic relaxed atoms | ✓ Only checks symbols + tags, not positions |
| No direct attribute bypass — uses `set_relaxed_structure` | ✓ |
| `sort_structure=False` preserved in new VASP sets | ✓ Passed via `**kwargs` from call site |
| CONFIG mutation issue avoided | ✓ Instance-level `self.CONFIG` shadows class-level one |
| Energy extraction correct per mode | ✓ test: `atoms[0].charge`; others: `get_potential_energy()` |
| `production_test` uses `TestRelaxSet` | ✓ Mirrors existing `bulk_relaxation` |
| Failed bulk jobs handled (no `store_mag_bulk_candidate` call) | ✓ |
| Finalization triggered for BOTH completed AND failed terminal jobs | ✓ |
| MAGMOM from enumerator config (not ASE atoms) | ✓ Reliable across all modes |
| Slab MAGMOM from `original_slab` (not ASE conversion chain) | ✓ |
| Adsorption MAGMOM: slab atoms + `[0.0]` for adsorbate | ✓ |
| `_mag_bulk_configs` re-enumerated after unpickling | ✓ Fallback in `submit_vasp_calculation` and `_get_config_magmom` |
| N=1 and N=0 (NM fallback) handled | ✓ |
| Second `enter_process("bulk_relaxation")` correctly blocked | ✓ Duplicate check finds existing rows |

---

## Verification

1. **Unit tests** — `tests/`:
   - `_extract_mag_config_index`: parses correctly, returns None for OH notes
   - `enumerate_magnetic_structures`: supercell filter; NM fallback; n_max cap
   - `finalize_mag_bulk_batch`: picks minimum energy; False when empty; magmom set on bulk_structure site properties

2. **Integration test** (`tests/production_test_mini_workflow.py`):
   - `add_process(cid, "bulk_relaxation")` → `list[int]` returned; each has `mag_config_index=N` in processNote
   - Submit all → `update_log` loop → all terminal
   - `study.get_bulk_atoms()` is set; `study.get_bulk_magmom()` is set
   - `study.bulk_structure.site_properties["magmom"]` is set
   - `initialize_oer_surface_study` → `surface_study.get_slab_magmom()` is not None
   - Surface submission calls `magnetic_surface_relaxation`
   - INCAR in surface VASP dir contains MAGMOM

3. Run full existing test suite; confirm no regressions.
