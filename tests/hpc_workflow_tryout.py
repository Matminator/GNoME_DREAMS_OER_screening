import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from ase import Atoms
from ase.io import read

from pymatgen.io.vasp import Incar as VaspIncar, Poscar

from gnome_dreams_oer_screening import OER_catalyst_study, Explog

# ─── CLI argument ──────────────────────────────────────────────────────────────
VALID_MODES = ("slurm_test", "MLIP_test", "production_test", "production")
if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
    sys.exit(f"Usage: python3 {sys.argv[0]} <mode>  — mode in {VALID_MODES}")
MODE = sys.argv[1]

# ─── Material / candidate (change these to test a different material) ───────────
current_dir     = Path(__file__).resolve().parent
MATERIAL_PATH   = (current_dir / "../notebooks/TiO2_rutile.cif").resolve()
MATERIAL_FORMAT = "cif"
CAND_ID         = "TiO2_rutile_workflow_test"
H2O_GFE         = -14.217
H2_GFE          = -6.77

# ─── Per-mode configuration ─────────────────────────────────────────────────────
_CFG = {
    "slurm_test":      {"sleep_sec": 20,  "sleep_min": 0.25 },
    "MLIP_test":       {"sleep_sec": 20,  "sleep_min": 2  },
    "production_test": {"sleep_sec": 180, "sleep_min": 2  },
    "production":      {"sleep_sec": 180, "sleep_min": 2  },
}
SLEEP_SEC = _CFG[MODE]["sleep_sec"]
SLEEP_MIN = _CFG[MODE]["sleep_min"]

# ─── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = current_dir / f"TEMP_{MODE}_hpc_workflow"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
_start_time = time.monotonic()


def elapsed() -> str:
    s = int(time.monotonic() - _start_time)
    return f"{s//3600:02d}:{(s % 3600)//60:02d}:{s % 60:02d}"


def check(condition, message):
    tag = "[OK]  " if condition else "[FAIL]"
    print(f"{elapsed()}  {tag}  {message}", flush=True)
    if not condition:
        raise AssertionError(message)


def wait_for_terminal(process_ids, explog, sleep_sec, label=""):
    """Poll update_log() until every process_id is terminal. Returns status dict."""
    iteration = 0
    while True:
        explog.update_log()
        pdf = explog.relational_frame.processes.df
        statuses = {
            pid: str(pdf.loc[pdf["process_id"] == pid, "status"].values[0])
            for pid in process_ids
        }
        all_done = all(
            s in ("completed", "failed") or s.startswith("unrecoverable")
            for s in statuses.values()
        )
        print(f"{elapsed()}  [{label} iter={iteration}]  {statuses}", flush=True)
        if all_done:
            break
        time.sleep(sleep_sec)
        iteration += 1
    return statuses


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n  HPC workflow test — mode: {MODE}\n{'='*60}\n", flush=True)

atoms = read(MATERIAL_PATH, format=MATERIAL_FORMAT)
study = OER_catalyst_study(
    atoms, H2O_gas_free_energy=H2O_GFE, H2_gas_free_energy=H2_GFE
)

EXPLOG = Explog()
EXPLOG.init(OUTPUT_DIR, MODE, sleep_min=SLEEP_MIN)
EXPLOG.add_candidate(candidate_id=CAND_ID, study_obj=study)

# ── Stage 1: Bulk relaxation ──────────────────────────────────────────────────
print(f"\n{elapsed()}  ── Stage 1: Bulk relaxation ──", flush=True)

bulk_ids = EXPLOG.add_process(candidate_id=CAND_ID, calculation_type="bulk_relaxation")
for pid in bulk_ids:
    EXPLOG.submit_process(process_id=pid)

bulk_statuses = wait_for_terminal(bulk_ids, EXPLOG, SLEEP_SEC, "bulk")
for pid, s in bulk_statuses.items():
    check(s == "completed", f"Bulk process {pid} must complete (got: {s})")

study = EXPLOG.relational_frame.candidates[CAND_ID].study_obj
check(
    study.get_bulk_structure() is not None,
    "Relaxed bulk structure must be set after bulk batch finalizes",
)

# ── MAGMOM: VASP INCAR files ──────────────────────────────────────────────────
print(f"{elapsed()}  [magmom] Checking INCAR files", flush=True)
incar_files = list(OUTPUT_DIR.rglob("INCAR"))
if incar_files:
    for incar_f in incar_files:
        incar_parsed = VaspIncar.from_file(incar_f)
        check(
            "MAGMOM" in incar_parsed,
            f"INCAR must contain MAGMOM tag ({incar_f.parent.name}/INCAR)",
        )
        poscar_f = incar_f.parent / "POSCAR"
        if poscar_f.exists():
            n_atoms = len(Poscar.from_file(poscar_f).structure)
            n_magmoms = len(incar_parsed["MAGMOM"])
            check(
                n_magmoms == n_atoms,
                f"MAGMOM count ({n_magmoms}) must match atom count ({n_atoms})"
                f" in {incar_f.parent.name}",
            )
    print(f"{elapsed()}  [magmom] Verified MAGMOM in {len(incar_files)} INCAR file(s)", flush=True)
else:
    print(f"{elapsed()}  [magmom] No INCAR files found — MLIP mode, skipping file check", flush=True)

# ── MAGMOM: relaxed bulk structure site_properties ────────────────────────────
bulk_struct = study.get_bulk_structure()
check(
    "magmom" in bulk_struct.site_properties,
    "Relaxed bulk pymatgen Structure must have 'magmom' in site_properties",
)
bulk_magmoms = bulk_struct.site_properties["magmom"]
check(
    len(bulk_magmoms) == len(bulk_struct),
    f"Magmom list length ({len(bulk_magmoms)}) must equal site count ({len(bulk_struct)})",
)
print(f"{elapsed()}  [magmom] Bulk magmoms: {bulk_magmoms}", flush=True)

# ── Surface prediction + termination selection ────────────────────────────────
print(f"\n{elapsed()}  ── Surface prediction ──", flush=True)

study.predict_most_likely_surfaces()
ranking = study.get_termination_rankings()
check(
    ranking is not None and len(ranking) > 0,
    "Termination ranking must be non-empty after predict_most_likely_surfaces()",
)

valid = ranking.dropna(subset=["Normalized score"])
check(len(valid) > 0, "At least one termination with a valid score must exist")
TERM_IDX = int(
    valid.sort_values("Normalized score", ascending=False).iloc[0]["Termination index"]
)
print(f"{elapsed()}  [info] Selected termination index: {TERM_IDX}", flush=True)

pre_df = study.get_init_adsorption_sites_df(TERM_IDX)
check(
    pre_df is not None and len(pre_df) > 0,
    "Preliminary adsorption sites must be available for selected termination",
)
check(
    not pre_df["ad site element"].isna().all(),
    "Pre-relaxation adsorption site elements must not all be NaN",
)

study.initialize_oer_surface_study(TERM_IDX)

# ── Pickle checkpoint ─────────────────────────────────────────────────────────
print(f"\n{elapsed()}  ── Pickle checkpoint ──", flush=True)
with open(EXPLOG.vasp_calculation_dir / "explog.pkl", "rb") as f:
    payload = pickle.load(f)
check(
    payload["candidates_df"].shape == EXPLOG.relational_frame.candidates.df.shape,
    "Pickle candidates_df shape matches live EXPLOG",
)
check(
    payload["processes_df"].shape == EXPLOG.relational_frame.processes.df.shape,
    "Pickle processes_df shape matches live EXPLOG",
)
study = payload["candidates_df"].loc[
    payload["candidates_df"]["candidate_id"] == CAND_ID, "study_obj"
].values[0]
check(
    study.get_bulk_structure() is not None,
    "Relaxed bulk structure present in pickled study_obj",
)

# ── Stage 2: Surface relaxation ───────────────────────────────────────────────
print(f"\n{elapsed()}  ── Stage 2: Surface relaxation ──", flush=True)

surf_id = EXPLOG.add_process(
    candidate_id=CAND_ID,
    calculation_type="surface_relaxation",
    termination_index=TERM_IDX,
)

study = EXPLOG.relational_frame.candidates[CAND_ID].study_obj
surface_study = study.get_surface_studies()[TERM_IDX]
slab_magmom = surface_study.get_slab_magmom()
n_surface = len(surface_study.get_non_relaxed_surface())
check(slab_magmom is not None, "Slab magmom must be set before surface_relaxation")
check(len(slab_magmom) == n_surface,
      f"Slab magmom length must equal n_surface atoms ({n_surface})")
print(f"{elapsed()}  [magmom] Slab magmom (first 5): {slab_magmom[:5]}", flush=True)

EXPLOG.submit_process(process_id=surf_id)
surf_statuses = wait_for_terminal([surf_id], EXPLOG, SLEEP_SEC, "surface")
check(surf_statuses[surf_id] == "completed", "Surface relaxation must complete")

study = EXPLOG.relational_frame.candidates[CAND_ID].study_obj
surface_study = study.get_surface_studies()[TERM_IDX]
check(
    surface_study.get_relaxed_surface() is not None,
    "Relaxed surface must be set after surface_relaxation completes",
)

# ── MAGMOM: relaxed surface magmoms ──────────────────────────────────────────
relaxed_surf_magmom = surface_study.get_relaxed_surface_magmom()
n_relaxed = len(surface_study.get_relaxed_surface())
check(relaxed_surf_magmom is not None,
      "Relaxed surface magmoms must be available after surface_relaxation")
check(len(relaxed_surf_magmom) == n_relaxed,
      f"Relaxed surface magmom length must equal n_relaxed atoms ({n_relaxed})")
print(f"{elapsed()}  [magmom] Relaxed surface magmom (first 5): "
      f"{relaxed_surf_magmom[:5]}", flush=True)

surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=False)
df = surface_study.get_adsorption_sites_df()
check(
    isinstance(df, pd.DataFrame) and len(df) > 0,
    "Adsorption sites DataFrame must be non-empty after site determination",
)

on_top_rows    = df[df["site type"] == "on-top"]
lattice_o_rows = df[df["site type"] == "lattice O"]
check(len(on_top_rows) > 0,    "At least one on-top site expected")
check(len(lattice_o_rows) > 0, "At least one lattice-O site expected")

ON_TOP_IDX    = int(on_top_rows["Site index"].iloc[0])
LATTICE_O_IDX = int(lattice_o_rows["Site index"].iloc[0])

surface_study.initialize_adsorption_site_study(ON_TOP_IDX)
surface_study.initialize_adsorption_site_study(LATTICE_O_IDX)
ad_studies      = surface_study.get_adsorption_site_studies_dict()
on_top_study    = ad_studies[ON_TOP_IDX]
lattice_o_study = ad_studies[LATTICE_O_IDX]

# ── MAGMOM: O init magmoms before O adsorption ───────────────────────────────
for site_label, ad_st in [("on-top", on_top_study), ("lattice-O", lattice_o_study)]:
    o_init = ad_st.get_O_init_magmom()
    n_O = len(ad_st.get_atoms_for_O_calc())
    check(o_init is not None,
          f"{site_label}: O init magmom must be available before O_adsorption")
    check(len(o_init) == n_O,
          f"{site_label}: O init magmom length must equal n_O atoms ({n_O})")
    print(f"{elapsed()}  [magmom] {site_label} O init magmom (first 5): "
          f"{o_init[:5]}", flush=True)

# ── Stage 3: O adsorption — both sites in parallel ────────────────────────────
print(f"\n{elapsed()}  ── Stage 3: O adsorption (on-top + lattice-O in parallel) ──",
      flush=True)

o_id_on_top = EXPLOG.add_process(
    candidate_id=CAND_ID, calculation_type="O_adsorption",
    termination_index=TERM_IDX, site_index=ON_TOP_IDX,
)
o_id_lat = EXPLOG.add_process(
    candidate_id=CAND_ID, calculation_type="O_adsorption",
    termination_index=TERM_IDX, site_index=LATTICE_O_IDX,
)
EXPLOG.submit_process(process_id=o_id_on_top)
EXPLOG.submit_process(process_id=o_id_lat)

o_statuses = wait_for_terminal([o_id_on_top, o_id_lat], EXPLOG, SLEEP_SEC, "O_adsorption")
check(o_statuses[o_id_on_top] == "completed", "on-top O_adsorption must complete")
check(o_statuses[o_id_lat]    == "completed", "lattice-O O_adsorption must complete")

surface_study.update_adsorption_energies()
df = surface_study.get_adsorption_sites_df()
check(
    np.isfinite(float(df.loc[df["Site index"] == ON_TOP_IDX].iloc[0]["G(O)"])),
    "G(O) must be finite after on-top O_adsorption",
)
check(
    np.isfinite(float(df.loc[df["Site index"] == LATTICE_O_IDX].iloc[0]["G(O)"])),
    "G(O) must be finite after lattice-O O_adsorption",
)

# ── MAGMOM: relaxed O structures ─────────────────────────────────────────────
for site_label, ad_st in [("on-top", on_top_study), ("lattice-O", lattice_o_study)]:
    o_calc = ad_st.get_O_calc_magmom()
    n_O = len(ad_st.relaxed_O_atoms)
    check(o_calc is not None,
          f"{site_label}: O calc magmom must be set after O_adsorption")
    check(len(o_calc) == n_O,
          f"{site_label}: O calc magmom length must equal n_O atoms ({n_O})")
    print(f"{elapsed()}  [magmom] {site_label} O calc magmom (first 5): "
          f"{o_calc[:5]}", flush=True)

# ── MAGMOM: OH init magmoms before OH adsorption ─────────────────────────────
for site_label, ad_st in [("on-top", on_top_study), ("lattice-O", lattice_o_study)]:
    oh_init = ad_st.get_OH_init_magmom()
    n_OH = len(ad_st.OH_init_atoms[0])
    check(oh_init is not None,
          f"{site_label}: OH init magmom must be available before OH_adsorption")
    check(len(oh_init) == n_OH,
          f"{site_label}: OH init magmom length must equal n_OH atoms ({n_OH})")
    print(f"{elapsed()}  [magmom] {site_label} OH init magmom (first 5): "
          f"{oh_init[:5]}", flush=True)

# ── Stage 4: OH adsorption — both sites in parallel (6 jobs) ─────────────────
print(f"\n{elapsed()}  ── Stage 4: OH adsorption (on-top×3 + lattice-O×3 in parallel) ──",
      flush=True)

oh_ids_on_top = EXPLOG.add_process(
    candidate_id=CAND_ID, calculation_type="OH_adsorption",
    termination_index=TERM_IDX, site_index=ON_TOP_IDX,
)
oh_ids_lat = EXPLOG.add_process(
    candidate_id=CAND_ID, calculation_type="OH_adsorption",
    termination_index=TERM_IDX, site_index=LATTICE_O_IDX,
)
check(len(oh_ids_on_top) == 3, "OH_adsorption must create 3 processes for on-top site")
check(len(oh_ids_lat)    == 3, "OH_adsorption must create 3 processes for lattice-O site")
check(
    not on_top_study.is_oh_batch_finalized(),
    "oh_batch_finalized must be False after add_process('OH_adsorption') on-top",
)
check(
    not lattice_o_study.is_oh_batch_finalized(),
    "oh_batch_finalized must be False after add_process('OH_adsorption') lattice-O",
)

for pid in oh_ids_on_top + oh_ids_lat:
    EXPLOG.submit_process(process_id=pid)

oh_statuses = wait_for_terminal(
    oh_ids_on_top + oh_ids_lat, EXPLOG, SLEEP_SEC, "OH_adsorption"
)
for pid, s in oh_statuses.items():
    check(s == "completed", f"OH process {pid} must complete (got: {s})")

# ── MAGMOM: relaxed OH structures ────────────────────────────────────────────
for site_label, ad_st in [("on-top", on_top_study), ("lattice-O", lattice_o_study)]:
    oh_calc = ad_st.get_OH_calc_magmom()
    n_OH = len(ad_st.relaxed_OH_atoms)
    check(oh_calc is not None,
          f"{site_label}: OH calc magmom must be set after OH_adsorption")
    check(len(oh_calc) == n_OH,
          f"{site_label}: OH calc magmom length must equal n_OH atoms ({n_OH})")
    print(f"{elapsed()}  [magmom] {site_label} OH calc magmom (first 5): "
          f"{oh_calc[:5]}", flush=True)

# ── Stage 5: Final metric checks ──────────────────────────────────────────────
print(f"\n{elapsed()}  ── Stage 5: Final metric checks ──", flush=True)

check(
    on_top_study.is_oh_batch_finalized(),
    "oh_batch_finalized restored True once all 3 on-top OH processes are terminal",
)
check(
    lattice_o_study.is_oh_batch_finalized(),
    "oh_batch_finalized restored True once all 3 lattice-O OH processes are terminal",
)

surface_study.update_adsorption_energies()
df = surface_study.get_adsorption_sites_df()

on_top_row    = df.loc[df["Site index"] == ON_TOP_IDX].iloc[0]
lattice_o_row = df.loc[df["Site index"] == LATTICE_O_IDX].iloc[0]

for label, row in [("on-top", on_top_row), ("lattice-O", lattice_o_row)]:
    check(
        np.isfinite(float(row["G(OH)"])),
        f"G(OH) must be finite for {label} site",
    )
    check(
        np.isfinite(float(row["ideal overpotential"])),
        f"Ideal overpotential must be finite for {label} site",
    )
    check(
        np.isfinite(float(row["overpotential from OH-OOH scaling relation"])),
        f"Scaling overpotential must be finite for {label} site",
    )

cand_row = EXPLOG.relational_frame.candidates[CAND_ID]
check(
    np.isfinite(float(cand_row.idealOverPotential)),
    "Candidate-level idealOverPotential must be finite",
)
check(
    np.isfinite(float(cand_row.Overpotential_from_scaling)),
    "Candidate-level Overpotential_from_scaling must be finite",
)

# ── Summary ───────────────────────────────────────────────────────────────────
total_sec = int(time.monotonic() - _start_time)
print(f"\n{'='*60}")
print(f"  ALL ASSERTIONS PASSED — mode: {MODE}")
print(f"  Total run time: {total_sec//3600:02d}:{(total_sec % 3600)//60:02d}:{total_sec % 60:02d}")
print(f"{'='*60}\n", flush=True)

print("Process log:")
print(EXPLOG.relational_frame.processes.df.to_string())
print("\nCandidate log:")
print(EXPLOG.relational_frame.candidates.df.to_string())

# ── Deep pickle value check ───────────────────────────────────────────────────
print(f"\n{elapsed()}  ── Deep pickle value check ──", flush=True)
with open(EXPLOG.vasp_calculation_dir / "explog.pkl", "rb") as f:
    final_payload = pickle.load(f)

live_proc_df = EXPLOG.relational_frame.processes.df
live_cand_df = EXPLOG.relational_frame.candidates.df
pkl_proc_df  = final_payload["processes_df"]
pkl_cand_df  = final_payload["candidates_df"]

check(
    pkl_proc_df.shape == live_proc_df.shape,
    "Final pickle processes_df shape matches live",
)
check(
    pkl_cand_df.shape == live_cand_df.shape,
    "Final pickle candidates_df shape matches live",
)

for col in ["G(O)", "G(OH)", "ideal overpotential",
            "overpotential from OH-OOH scaling relation"]:
    live_vals = live_proc_df[col].dropna()
    pkl_vals  = pkl_proc_df[col].dropna()
    check(
        live_vals.equals(pkl_vals),
        f"processes_df['{col}'] values match between live and pickle",
    )

for col in ["idealOverPotential", "Overpotential_from_scaling"]:
    live_val = float(live_cand_df.loc[
        live_cand_df["candidate_id"] == CAND_ID, col].iloc[0])
    pkl_val  = float(pkl_cand_df.loc[
        pkl_cand_df["candidate_id"] == CAND_ID, col].iloc[0])
    check(
        live_val == pkl_val,
        f"candidates_df['{col}'] value matches: live={live_val:.4f} pkl={pkl_val:.4f}",
    )

pkl_study = pkl_cand_df.loc[
    pkl_cand_df["candidate_id"] == CAND_ID, "study_obj"].values[0]
check(
    pkl_study.get_bulk_structure() is not None,
    "Final pickle study_obj has relaxed bulk structure",
)
pkl_ss = pkl_study.get_surface_studies()
check(
    pkl_ss is not None and TERM_IDX in pkl_ss,
    "Final pickle study_obj has OER surface study for TERM_IDX",
)
check(
    pkl_ss[TERM_IDX].get_relaxed_surface() is not None,
    "Final pickle surface study has relaxed surface",
)
plk_ad_studies = pkl_ss[TERM_IDX].get_adsorption_site_studies_dict()
plk_on_top_study = plk_ad_studies[ON_TOP_IDX]
plk_lat_o_study = plk_ad_studies[LATTICE_O_IDX]

# Test that relaxed O magmoms are present in the pickled study objects:
check(
    len(plk_on_top_study._relaxed_O_magmom) == len(on_top_study.relaxed_O_atoms),
    "Final pickle on-top adsorption study has _relaxed_O_magmom",
)
check(
    len(plk_lat_o_study._relaxed_O_magmom) == len(lattice_o_study.relaxed_O_atoms),
    "Final pickle lattice-O adsorption study has _relaxed_O_magmom",
)

# Test that relaxed O atoms exist in the pickled study objects:
check(
    isinstance(plk_on_top_study.relaxed_O_atoms, Atoms) and len(plk_on_top_study.relaxed_O_atoms) > 0,
    "Final pickle on-top adsorption study has relaxed_O_atoms",
)
check(
    isinstance(plk_lat_o_study.relaxed_O_atoms, Atoms) and len(plk_lat_o_study.relaxed_O_atoms) > 0,
    "Final pickle lattice-O adsorption study has relaxed_O_atoms",
)

# Test that relaxed OH atoms exist in the pickled study objects:
check(
    isinstance(plk_on_top_study.relaxed_OH_atoms, Atoms) and len(plk_on_top_study.relaxed_OH_atoms) > 0,
    "Final pickle on-top adsorption study has relaxed_OH_atoms",
)
check(
    isinstance(plk_lat_o_study.relaxed_OH_atoms, Atoms) and len(plk_lat_o_study.relaxed_OH_atoms) > 0,
    "Final pickle lattice-O adsorption study has relaxed_OH_atoms",
)