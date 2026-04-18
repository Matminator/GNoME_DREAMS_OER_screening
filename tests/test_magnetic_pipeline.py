# End-to-end tests for the magnetic bulk → surface → adsorbate pipeline.
#
# These tests call MagneticStructureEnumerator WITHOUT mocking, so they
# require the enumlib binaries (enum.x, makestr.x / multienum.x) to be
# on PATH. They are skipped automatically when those binaries are absent
# (e.g. on Windows developer machines).
#
# Test-mode pipeline (mode='test', _run_magnetic_bulk_pipeline helper):
#   Phase 1  — pre-add_process state assertions
#   Phase 2  — add_process creates one bulk process per magnetic config
#   Phase 3  — submit + update_log → finalization, winning magmoms stored
#   Phase 4  — predict_most_likely_surfaces, initialize surface study,
#              slab magmom matches bulk
#   Phase 5  — surface_relaxation (test mode) → determine_adsorption_sites,
#              get_O_init_magmom length and value checks
#   Phase 6  — O_adsorption (test mode) → _relaxed_O_magmom injected,
#              get_OH_init_magmom length and value checks for all site types
#
# Production-mode pipeline (mode='production', monkeypatched VASP/SLURM):
#   test_production_mode_full_pipeline_fe    — magnetic material (Fe),
#     exercises atoms.get_magnetic_moments() from SinglePointCalculator,
#     set_relaxed_base_surface with require_magmom=True, _relaxed_surface_magmom
#     populated, set_relaxed_O_calc and set_relaxed_OH_calc real-calculator paths
#   test_production_mode_full_pipeline_al_nm — NM material (Al), validates
#     all-zero magmoms accepted by require_magmom=True
#   test_production_mode_unrecoverable_surface_fe — disconnected slab triggers
#     unrecoverable_reason path (job_handler.py:1247–1317)

import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import Explog, OER_catalyst_study
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
    extract_mag_config_index,
)


pytestmark = pytest.mark.mini

_TRAJ = Path(__file__).parent.parent / "notebooks" / "test_cands.traj"
_needs_traj = pytest.mark.skipif(not _TRAJ.exists(), reason="test_cands.traj not found")


def _enumlib_available() -> bool:
    """Return True if the required enumlib executables are on PATH."""
    has_enum = (
        shutil.which("enum.x") is not None
        or shutil.which("multienum.x") is not None
    )
    has_make = (
        shutil.which("makestr.x") is not None
        or shutil.which("makeStr.py") is not None
    )
    return has_enum and has_make


needs_enumlib = pytest.mark.skipif(
    not _enumlib_available(),
    reason="enumlib binaries (enum.x / makestr.x) not found on PATH",
)


def _potcars_available() -> bool:
    """Return True if pymatgen's PMG_VASP_PSP_DIR is configured."""
    import os
    try:
        from pymatgen.core import SETTINGS
        return bool(
            SETTINGS.get("PMG_VASP_PSP_DIR")
            or os.environ.get("PMG_VASP_PSP_DIR")
        )
    except Exception:
        return bool(os.environ.get("PMG_VASP_PSP_DIR"))


needs_potcars = pytest.mark.skipif(
    not _potcars_available(),
    reason="PMG_VASP_PSP_DIR not configured — POTCAR lookup unavailable",
)


# ---------------------------------------------------------------------------
# Production-mode infrastructure: fake VASP result reader + mock helpers
# ---------------------------------------------------------------------------

# Default per-element magmoms for production-mode fake results.
# 3d magnetic transition metals use high-spin values (~5.0) matching
# pymatgen's MAGNETIC_SPECIES_DEFAULT_MAGMOMS.  4d/5d metals and all
# main-group/non-metal elements default to 0.0 — VASP post-relaxation
# magmoms for these are typically near zero.
#
# NOTE: all-zero magmoms are valid here.  require_magmom=True (the
# production default) only checks that a calculator is *attached*
# (i.e. _relaxed_surface_magmom is not None).  A SinglePointCalculator
# with magmoms=[0.0, ...] satisfies that check, so non-magnetic materials
# (IrO2, RuO2, Al2O3, ...) complete the full pipeline without errors.
_FAKE_MAGMOMS_BY_ELEMENT = {
    'Fe': 5.0, 'Co': 5.0, 'Ni': 5.0, 'Mn': 5.0,
    'Cr': 5.0, 'V':  5.0, 'Ti': 0.0,
    'Ir': 0.0, 'Ru': 0.0, 'Rh': 0.0, 'Pd': 0.0,
}


def _default_fake_read_vasp_results(vasp_dir):
    """Fake read_vasp_results for production-mode tests.

    Reads the POSCAR that calculational_setup already wrote to vasp_dir,
    attaches a SinglePointCalculator with plausible energy and per-atom
    magmoms, and returns a one-element list matching the real function's
    format (the caller takes [-1]).

    Why POSCAR: calculational_setup always writes POSCAR before SLURM
    submission (vasp_calculation.py:173/186).  Reading it back mirrors
    exactly what VASP does: receive the input structure, return it with
    computed values attached.
    """
    atoms = read(str(Path(vasp_dir) / "POSCAR"))
    magmoms = [
        _FAKE_MAGMOMS_BY_ELEMENT.get(sym, 0.0)
        for sym in atoms.get_chemical_symbols()
    ]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-5.0 * len(atoms),  # plausible DFT-like total energy (eV)
        magmoms=magmoms,
    )
    return [atoms]


def _fake_disconnected_surface_reader(vasp_dir):
    """Fake reader that displaces one slab atom into the centre of the vacuum.

    Positions the last slab atom at the midpoint between the top of the slab
    and its periodic image at the bottom (i.e. the centre of the vacuum gap).
    This is always vacuum/2 = 6 Å from the nearest slab atom (vacuum=12 Å),
    well above the Fe-Fe bond cutoff (~3.2 Å), and never wraps via PBC
    regardless of slab thickness or Miller index.

    Earlier approaches used a fixed Cartesian or cell[2]-aligned displacement:
    - [0,0,10] is in-plane for BCC(110) (cell[2] ∥ [1,1,0]) and wraps through
      the ~2.87 Å in-plane PBC cell, leaving the atom bonded.
    - 10 × c_hat wraps in the surface-normal direction when the total cell
      height is ~18 Å (6 Å slab + 12 Å vacuum): the atom ends up only
      18 - (6 + 10) = 2 Å from the bottom layer through PBC.
    Using fractional coordinates avoids both failure modes.
    """
    atoms = read(str(Path(vasp_dir) / "POSCAR"))
    scaled = atoms.get_scaled_positions()
    top_frac = scaled[:, 2].max()    # top of slab in fractional c
    bot_frac = scaled[:, 2].min()    # bottom of slab in fractional c
    # Centre of the vacuum gap (between top of slab and the next periodic
    # image of the slab bottom), guaranteed to be vacuum/2 from any slab atom.
    vac_centre_frac = (top_frac + 1.0 + bot_frac) / 2.0
    if vac_centre_frac >= 1.0:
        vac_centre_frac -= 1.0
    scaled[-1, 2] = vac_centre_frac
    atoms.set_scaled_positions(scaled)
    magmoms = [
        _FAKE_MAGMOMS_BY_ELEMENT.get(sym, 0.0)
        for sym in atoms.get_chemical_symbols()
    ]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-5.0 * len(atoms),
        magmoms=magmoms,
    )
    return [atoms]


def _job_type_aware_disconnected_reader(vasp_dir):
    """Route bulk vs surface POSCAR to different readers.

    job_handler.py encodes the job_type in the vasp_dir name:
        {prefix}{candidate_id}_{job_type}_{process_id}
    Matching on "surface_relaxation" is exact and requires no geometry heuristics.
    """
    if "surface_relaxation" in Path(vasp_dir).name:
        return _fake_disconnected_surface_reader(vasp_dir)
    return _default_fake_read_vasp_results(vasp_dir)


def _apply_production_mocks(explog, monkeypatch, result_reader=None):
    """Apply the three mocks that let production-mode update_log run in pytest.

    1. Replace queueadapter with a no-op that returns a fake slurm_id.
       Direct attribute replacement is used because queueadapter is an
       instance attribute (job_handler.py:60); no module-namespace patch needed.

    2. Monkeypatch get_my_jobs_status to return {} (empty dict).
       An absent slurm_id is the normal post-completion state — the loop at
       job_handler.py:962-984 sets those processes to 'updating', exactly as
       it would when a real SLURM job finishes and leaves the queue.

    3. Monkeypatch read_vasp_results to return atoms built from the POSCAR
       that calculational_setup already wrote, with a SinglePointCalculator
       attached.  This exercises job_handler.py:1175 and all downstream
       consumers (set_relaxed_base_surface with require_magmom=True, etc.)
       for the first time in pytest.

    Parameters
    ----------
    result_reader : callable or None
        Override for read_vasp_results.  Defaults to
        _default_fake_read_vasp_results.  Pass a custom reader to inject
        broken structures (e.g. _job_type_aware_disconnected_reader for the
        unrecoverable-surface test).
    """
    if result_reader is None:
        result_reader = _default_fake_read_vasp_results

    class _MockQueueAdapter:
        def submit_job(self, **kwargs):
            return 9999  # fake slurm_id stored in the process row

    explog.job_handler.queueadapter = _MockQueueAdapter()

    monkeypatch.setattr(
        "gnome_dreams_oer_screening.vasp.job_handler.get_my_jobs_status",
        lambda: {},
    )
    monkeypatch.setattr(
        "gnome_dreams_oer_screening.vasp.job_handler.read_vasp_results",
        result_reader,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fe_bcc_atoms():
    """2-atom BCC Fe conventional cell."""
    return bulk('Fe', 'bcc', a=2.87, cubic=True)


def _load_gnome_atoms(traj_index: int):
    """Load an ASE Atoms object from the GNoME test trajectory."""
    return read(str(_TRAJ), index=traj_index)


@pytest.fixture(scope="module")
def gnome_Fe2Ir2O12Rh2_atoms():
    return _load_gnome_atoms(8)


@pytest.fixture(scope="module")
def gnome_IrO4Pd_atoms():
    return _load_gnome_atoms(30)


@pytest.fixture(scope="module")
def gnome_Ag3BiCoIr3O14_atoms():
    return _load_gnome_atoms(6)


@pytest.fixture(scope="module")
def al_fcc_atoms():
    """FCC Al — non-magnetic (Al is not in pymatgen's magnetic elements list,
    so MagneticStructureEnumerator raises 'Not detected as magnetic' and
    enumerate_magnetic_structures returns a single NM fallback entry)."""
    return bulk('Al', 'fcc', a=4.05, cubic=True)


# ---------------------------------------------------------------------------
# Reusable pipeline helper
# ---------------------------------------------------------------------------

def _run_magnetic_bulk_pipeline(
    atoms,
    n_expected: int,
    tmp_path,
    n_max: int = 4,
    is_magnetic: bool = True,
    slab_kwargs: dict | None = None,
) -> None:
    """Run the full magnetic bulk → surface → adsorbate pipeline and assert correctness.

    Args:
        atoms:       ASE Atoms for the bulk structure under test.
        n_expected:  Number of magnetic configs expected from
                     enumerate_magnetic_structures(structure, n_max=n_max).
                     The caller must supply this; the function verifies it.
        tmp_path:    Temporary directory (pytest tmp_path fixture or similar).
        n_max:       Cap on magnetic configs passed to Explog (default 4).
        is_magnetic: When False, replaces non-zero magmom assertions with
                     ≈0.6 checks (NM fallback path).
        slab_kwargs: Passed as **slab_kwargs to predict_most_likely_surfaces.
                     Defaults to {"min_atoms": 4, "min_slab_thickness": 4,
                     "max_layers": 12} when None.
    """
    _slab_kwargs = slab_kwargs or {
        "min_atoms": 4, "min_slab_thickness": 4, "max_layers": 12
    }

    # Verify the enumerator agrees with the caller's n_expected:
    structure = AseAtomsAdaptor.get_structure(atoms)
    actual_configs = enumerate_magnetic_structures(structure, n_max=n_max)
    assert len(actual_configs) == n_expected, (
        f"Caller expected {n_expected} configs, "
        f"enumerator returned {len(actual_configs)}"
    )

    n_atoms = len(atoms)

    EXPLOG = Explog()
    EXPLOG.init(
        vasp_calculation_dir=tmp_path,
        mode="test",
        prefix_vasp_dirs=False,
        magnetic_bulk_n_max=n_max,
    )

    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    cand_id = "mag_pipeline_test"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="magnetic bulk pipeline test",
        notes="tests FM/AFM enumeration and finalization",
        study_obj=study,
    )

    # -----------------------------------------------------------------------
    # Phase 1 — Before add_process: study has no configs, batch not finalized
    # -----------------------------------------------------------------------
    assert study.get_mag_bulk_configs() is None, (
        "Configs not stored until add_process is called"
    )
    assert study.is_mag_bulk_batch_finalized() == False, (
        "Batch must not be finalized before any bulk jobs run"
    )
    assert study.get_bulk_magmom() is None, (
        "get_bulk_magmom() must be None before finalization"
    )

    # -----------------------------------------------------------------------
    # Phase 2 — add_process("bulk_relaxation"): one process per config
    # -----------------------------------------------------------------------
    p_ids = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="bulk_relaxation",
    )

    assert len(p_ids) == n_expected, (
        f"Expected {n_expected} bulk processes (one per magnetic config), "
        f"got {len(p_ids)}"
    )

    stored_configs = study.get_mag_bulk_configs()
    assert stored_configs is not None
    assert len(stored_configs) == n_expected

    for i, cfg in enumerate(stored_configs):
        n_atoms_in_config = len(cfg['structure'].sites)
        assert len(cfg['magmom']) == n_atoms_in_config, (
            f"Stored config {i}: magmom length mismatch"
        )
        if is_magnetic:
            assert any(abs(m) > 1 for m in cfg['magmom']), (
                f"Stored config {i}: expected non-zero magmoms, got {cfg['magmom']}"
            )
        else:
            assert all(m == pytest.approx(0.6, abs=0.05) for m in cfg['magmom']), (
                f"NM config {i}: expected all magmoms ≈ 0.6, got {cfg['magmom']}"
            )

    assert study.is_mag_bulk_batch_finalized() == False

    pdf = EXPLOG.relational_frame.processes.df
    bulk_rows = pdf[pdf['job_type'] == 'bulk_relaxation']
    assert len(bulk_rows) == n_expected

    notes = bulk_rows['processNote'].tolist()
    config_indices = [extract_mag_config_index(n) for n in notes]
    assert None not in config_indices, (
        f"Every bulk processNote must contain mag_config_index=N, got: {notes}"
    )
    assert sorted(config_indices) == list(range(n_expected)), (
        f"mag_config_index values must be 0..{n_expected - 1}, "
        f"got {config_indices}"
    )

    # -----------------------------------------------------------------------
    # Phase 3 — submit all, update_log: finalization must occur
    # -----------------------------------------------------------------------
    for pid in p_ids:
        EXPLOG.submit_process(process_id=pid)

    updates = EXPLOG.update_log()
    assert all(v == 'completed' for v in updates.values()), (
        f"Expected all bulk processes to complete in test mode, got: {updates}"
    )
    assert len(updates) == n_expected, (
        f"Expected {n_expected} updates, got {len(updates)}"
    )

    assert study.is_mag_bulk_batch_finalized() == True, (
        "finalize_mag_bulk_batch must be called once all bulk processes complete"
    )

    bulk_magmom = study.get_bulk_magmom()
    assert bulk_magmom is not None
    assert len(bulk_magmom) <= n_atoms, (
        f"get_bulk_magmom() should have at most {n_atoms} entries, got {len(bulk_magmom)}"
    )

    # In test mode, fake energy = len(atoms), so E/atom = 1 for every config
    # regardless of cell size.  All candidates tie; Python's min() returns the
    # first (config 0):
    winning_magmom = stored_configs[0]['magmom']
    assert bulk_magmom == winning_magmom, (
        f"In test mode (all E/atom tie), config 0 wins. "
        f"Expected magmoms {winning_magmom}, got {bulk_magmom}"
    )

    if is_magnetic:
        assert any(abs(m) > 1 for m in bulk_magmom), (
            f"Winning bulk magmoms must be non-zero, got {bulk_magmom}"
        )
    else:
        assert all(m == pytest.approx(0.6, abs=0.05) for m in bulk_magmom), (
            f"NM winning magmoms must all be ≈ 0.6, got {bulk_magmom}"
        )

    bulk_struct = study.get_bulk_structure()
    assert bulk_struct is not None
    site_magmoms = [s.properties.get('magmom') for s in bulk_struct.sites]
    assert site_magmoms == winning_magmom, (
        "bulk_structure site_properties['magmom'] must equal the winning magmoms"
    )

    # -----------------------------------------------------------------------
    # Phase 4 — rank terminations, pick one with a finite score, init surface.
    # -----------------------------------------------------------------------
    study.predict_most_likely_surfaces(max_miller=1, **_slab_kwargs)

    ranking_df = study.get_termination_rankings()
    assert ranking_df is not None, (
        "get_termination_rankings() must return a DataFrame after "
        "predict_most_likely_surfaces has run"
    )
    assert len(ranking_df) >= 1, "At least one termination must be ranked"

    valid_rows = ranking_df[ranking_df['Score'].notna()]
    assert len(valid_rows) >= 1, (
        "At least one termination must have a non-NaN score — a NaN score "
        "indicates no valid slab could be generated for any termination"
    )
    termination_index = int(valid_rows.iloc[0]['Termination index'])
    assert np.isfinite(float(valid_rows.iloc[0]['Score'])), (
        f"Termination {termination_index}: Score must be finite, "
        f"got {valid_rows.iloc[0]['Score']}"
    )

    study.initialize_oer_surface_study(termination_index)
    surface_studies = study.get_surface_studies()
    assert len(surface_studies) >= 1

    surface_study = surface_studies[termination_index]
    slab_magmom = surface_study.get_slab_magmom()

    assert slab_magmom is not None, (
        "get_slab_magmom() must be non-None after SlabGenerator propagates "
        "bulk site_properties['magmom']"
    )

    tolerance = 0.05
    bulk_elem_magmoms: dict[str, set] = {}
    for site, m in zip(bulk_struct.sites, winning_magmom):
        elem = site.species_string
        bulk_elem_magmoms.setdefault(elem, set()).add(abs(m))

    slab_sites = surface_study.original_slab.sites
    for site, slab_m in zip(slab_sites, slab_magmom):
        elem = site.species_string
        assert elem in bulk_elem_magmoms, (
            f"Slab contains element '{elem}' not found in bulk structure"
        )
        expected_magnitudes = bulk_elem_magmoms[elem]
        assert any(
            abs(abs(slab_m) - ref_m) <= tolerance
            for ref_m in expected_magnitudes
        ), (
            f"Slab atom {elem}: |magmom|={abs(slab_m):.4f} does not match "
            f"any bulk reference magnitude {expected_magnitudes} "
            f"(tolerance {tolerance})"
        )

    # -----------------------------------------------------------------------
    # Phase 5 — Surface relaxation (test mode) + O-adsorbate magmom check.
    # job_handler passes require_magmom=False so test-mode atoms (no VASP
    # calculator) don't raise. Bulk-derived slab magmoms are unaffected.
    # determine_adsorption_sites is not called by job_handler — call it here.
    # -----------------------------------------------------------------------
    surf_pid = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="surface_relaxation",
        termination_index=termination_index,
    )
    EXPLOG.submit_process(process_id=surf_pid)
    EXPLOG.update_log()

    surface_study.determine_adsorption_sites()

    ad_sites_df = surface_study.add_sites_df_relaxed
    assert ad_sites_df is not None and len(ad_sites_df) >= 1, (
        "At least one adsorption site must be found after surface relaxation"
    )

    for _, row in ad_sites_df.iterrows():
        surface_study.initialize_adsorption_site_study(int(row['Site index']))

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()
    slab_n = len(slab_magmom)

    for site_idx, ad_study in ad_site_dict.items():
        o_magmom = ad_study.get_O_init_magmom(slab_magmom)
        site_type = ad_study.site_type

        if site_type == 'on-top':
            assert len(o_magmom) == slab_n + 1, (
                f"Site {site_idx} on-top: O magmom length must be {slab_n + 1}, "
                f"got {len(o_magmom)}"
            )
            assert o_magmom[:slab_n] == slab_magmom, (
                f"Site {site_idx} on-top: slab portion of O magmom must equal slab_magmom"
            )
            assert o_magmom[-1] == 0.0, (
                f"Site {site_idx} on-top: appended O atom magmom must be 0.0"
            )
            if is_magnetic:
                assert any(abs(m) > 0.1 for m in o_magmom[:slab_n]), (
                    f"Site {site_idx} on-top: slab portion must contain non-zero magmoms"
                )

        elif site_type == 'lattice O':
            assert len(o_magmom) == slab_n - 1, (
                f"Site {site_idx} lattice-O: O magmom length must be {slab_n - 1}, "
                f"got {len(o_magmom)}"
            )
            if is_magnetic:
                assert any(abs(m) > 0.1 for m in o_magmom), (
                    f"Site {site_idx} lattice-O: at least one non-zero magmom expected"
                )

    # -----------------------------------------------------------------------
    # Phase 6 — O_adsorption (test mode) + OH-adsorbate magmom check.
    # job_handler injects _relaxed_O_magmom from the initial O magmom when
    # no VASP calculator is present, enabling get_OH_init_magmom for on-top.
    # -----------------------------------------------------------------------
    for site_idx in ad_site_dict.keys():
        pid = EXPLOG.add_process(
            candidate_id=cand_id,
            calculation_type="O_adsorption",
            termination_index=termination_index,
            site_index=site_idx,
        )
        EXPLOG.submit_process(process_id=pid)

    EXPLOG.update_log()

    for site_idx, ad_study in ad_site_dict.items():
        site_type = ad_study.site_type

        if site_type == 'on-top':
            assert ad_study._relaxed_O_magmom is not None, (
                f"Site {site_idx} on-top: _relaxed_O_magmom must be set after "
                "O_adsorption completes in test mode"
            )
            oh_magmom = ad_study.get_OH_init_magmom(slab_magmom)
            # on-top: _relaxed_O_magmom (slab_n + 1 entries) + [0.0] for H
            assert len(oh_magmom) == slab_n + 2, (
                f"Site {site_idx} on-top: OH magmom length must be {slab_n + 2}, "
                f"got {len(oh_magmom)}"
            )
            assert oh_magmom[:slab_n] == slab_magmom, (
                f"Site {site_idx} on-top: slab portion of OH magmom must equal slab_magmom"
            )
            assert oh_magmom[-1] == 0.0, (
                f"Site {site_idx} on-top: appended H magmom must be 0.0"
            )
            if is_magnetic:
                assert any(abs(m) > 0.1 for m in oh_magmom[:slab_n]), (
                    f"Site {site_idx} on-top: slab portion must contain non-zero magmoms"
                )

        elif site_type == 'lattice O':
            oh_magmom = ad_study.get_OH_init_magmom(slab_magmom)
            # lattice-O: slab_magmom (slab_n entries) + [0.0] for H
            assert len(oh_magmom) == slab_n + 1, (
                f"Site {site_idx} lattice-O: OH magmom length must be {slab_n + 1}, "
                f"got {len(oh_magmom)}"
            )
            assert oh_magmom[:slab_n] == slab_magmom, (
                f"Site {site_idx} lattice-O: slab portion of OH magmom must equal slab_magmom"
            )
            assert oh_magmom[-1] == 0.0, (
                f"Site {site_idx} lattice-O: appended H magmom must be 0.0"
            )
            if is_magnetic:
                assert any(abs(m) > 0.1 for m in oh_magmom[:slab_n]), (
                    f"Site {site_idx} lattice-O: slab portion must contain non-zero magmoms"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@needs_enumlib
def test_enumerate_magnetic_structures_fe(fe_bcc_atoms):
    """enumerate_magnetic_structures returns ≥ 2 distinct configs for BCC Fe."""
    structure = AseAtomsAdaptor.get_structure(fe_bcc_atoms)
    configs = enumerate_magnetic_structures(structure)

    assert len(configs) >= 2, (
        f"Expected at least 2 configs for BCC Fe, got {len(configs)}"
    )

    for i, cfg in enumerate(configs):
        assert 'structure' in cfg
        assert 'magmom' in cfg
        assert 'ordering' in cfg

        n_cfg_atoms = len(cfg['structure'].sites)
        assert len(cfg['magmom']) == n_cfg_atoms, (
            f"Config {i}: magmom length {len(cfg['magmom'])} != "
            f"structure atom count {n_cfg_atoms}"
        )

        assert any(abs(m) > 0.1 for m in cfg['magmom']), (
            f"Config {i}: expected at least one non-zero magmom for Fe, "
            f"got {cfg['magmom']}"
        )

        stored = [s.properties.get('magmom') for s in cfg['structure'].sites]
        assert stored == cfg['magmom'], (
            f"Config {i}: site_properties['magmom'] does not match "
            f"config['magmom']"
        )

    magmom_tuples = [tuple(c['magmom']) for c in configs]
    assert len(magmom_tuples) == len(configs), (
        "All configs must have distinct magmom patterns"
    )

    has_afm = any(
        any(m > 0.1 for m in c['magmom']) and any(m < -0.1 for m in c['magmom'])
        for c in configs
    )
    assert has_afm, (
        "Expected at least one AFM-like config (mixed signs) for BCC Fe"
    )

    has_fm = any(all(m > 0.1 for m in c['magmom']) for c in configs)
    assert has_fm, (
        "Expected at least one FM config (all positive) for BCC Fe"
    )


@needs_enumlib
def test_magnetic_pipeline_fe(tmp_path, fe_bcc_atoms):
    """End-to-end magnetic bulk pipeline for 2-atom BCC Fe.

    BCC Fe in a 2-atom conventional cell yields 3 magnetic orderings
    (1-atom primitive FM + two 2-atom AFM), all within the default n_max=4 cap.
    """
    _run_magnetic_bulk_pipeline(
        fe_bcc_atoms,
        n_expected=3,
        tmp_path=tmp_path,
        slab_kwargs={"min_atoms": 4, "min_slab_thickness": 4, "max_layers": 12},
    )


@needs_enumlib
@_needs_traj
def test_magnetic_pipeline_Fe2Ir2O12Rh2(tmp_path, gnome_Fe2Ir2O12Rh2_atoms):
    """End-to-end pipeline for GNoME Fe2Ir2O12Rh2 (Fe and Rh magnetic, 2 orderings)."""
    _run_magnetic_bulk_pipeline(
        gnome_Fe2Ir2O12Rh2_atoms,
        n_expected=2,
        tmp_path=tmp_path,
    )


@needs_enumlib
@_needs_traj
def test_magnetic_pipeline_IrO4Pd(tmp_path, gnome_IrO4Pd_atoms):
    """End-to-end pipeline for GNoME IrO4Pd (NM fallback — not detected as magnetic)."""
    _run_magnetic_bulk_pipeline(
        gnome_IrO4Pd_atoms,
        n_expected=1,
        tmp_path=tmp_path,
        is_magnetic=False,
    )


@needs_enumlib
@_needs_traj
def test_magnetic_pipeline_Ag3BiCoIr3O14(tmp_path, gnome_Ag3BiCoIr3O14_atoms):
    """End-to-end pipeline for GNoME Ag3BiCoIr3O14 (Co magnetic, FM only — 1 ordering)."""
    _run_magnetic_bulk_pipeline(
        gnome_Ag3BiCoIr3O14_atoms,
        n_expected=1,
        tmp_path=tmp_path,
    )


@needs_enumlib
def test_bulk_pipeline_non_magnetic_al(tmp_path, al_fcc_atoms):
    """NM fallback path: Al FCC is non-magnetic.

    Al is not in pymatgen's magnetic elements list, so
    MagneticStructureEnumerator raises 'Not detected as magnetic' and
    enumerate_magnetic_structures returns exactly 1 NM fallback config
    with magmoms ≈ 0.6. The full pipeline must complete correctly.
    Note: Cu IS in the magnetic elements list and was deliberately not used.
    """
    _run_magnetic_bulk_pipeline(
        al_fcc_atoms,
        n_expected=1,
        tmp_path=tmp_path,
        is_magnetic=False,
    )


# ---------------------------------------------------------------------------
# Production-mode tests: monkeypatched read_vasp_results
# ---------------------------------------------------------------------------

@needs_enumlib
@needs_potcars
def test_production_mode_full_pipeline_fe(tmp_path, fe_bcc_atoms, monkeypatch):
    """Production-mode full pipeline: bulk → surface → O → OH (BCC Fe, magnetic).

    Uses mode='production' so update_log takes the real result-reading branch
    (job_handler.py:1095 else-block).  Three mocks replace the SLURM/VASP
    infrastructure; see _apply_production_mocks for details.

    New code paths exercised vs test mode:
      - atoms.get_magnetic_moments() from SinglePointCalculator (line 1236)
      - set_relaxed_base_surface(atoms) with require_magmom=True (line 1246)
      - _relaxed_surface_magmom populated for the first time (oer_surface_study.py:231)
      - set_relaxed_O_calc(atoms) with require_magmom=True (line 1258)
      - _relaxed_O_magmom from real calculator, not init-magmom proxy (line 338)
      - set_relaxed_OH_calc(atoms) real-calculator path (line 1271)
    """
    _slab_kwargs = {"min_atoms": 4, "min_slab_thickness": 4, "max_layers": 12}

    EXPLOG = Explog()
    EXPLOG.init(
        vasp_calculation_dir=tmp_path,
        mode="production",
        prefix_vasp_dirs=False,
        magnetic_bulk_n_max=4,
    )
    _apply_production_mocks(EXPLOG, monkeypatch)

    study = OER_catalyst_study(
        fe_bcc_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    cand_id = "prod_fe_test"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="production-mode pipeline test",
        notes="monkeypatched VASP, BCC Fe",
        study_obj=study,
    )

    # ------------------------------------------------------------------
    # Phase 1 — bulk relaxation in production mode
    # ------------------------------------------------------------------
    p_ids = EXPLOG.add_process(candidate_id=cand_id,
                               calculation_type="bulk_relaxation")
    for pid in p_ids:
        EXPLOG.submit_process(process_id=pid)

    updates = EXPLOG.update_log()
    assert all(v == 'completed' for v in updates.values()), (
        f"All bulk processes must complete: {updates}"
    )
    assert study.is_mag_bulk_batch_finalized()

    bulk_magmom = study.get_bulk_magmom()
    assert bulk_magmom is not None
    # Magmoms come from _FAKE_MAGMOMS_BY_ELEMENT via SinglePointCalculator,
    # not from the enumerator initial values used in test mode.
    assert all(
        abs(m) == pytest.approx(5.0, abs=0.1)
        for m in bulk_magmom
    ), (
        f"Production mode: bulk magmoms must come from fake calculator "
        f"(5.0 for Fe), got {bulk_magmom}"
    )

    # ------------------------------------------------------------------
    # Phase 2 — surface relaxation in production mode
    # ------------------------------------------------------------------
    study.predict_most_likely_surfaces(max_miller=1, **_slab_kwargs)
    ranking_df = study.get_termination_rankings()
    valid_rows = ranking_df[ranking_df['Score'].notna()]
    assert len(valid_rows) >= 1, "At least one valid termination required"
    termination_index = int(valid_rows.iloc[0]['Termination index'])
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    surf_pid = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="surface_relaxation",
        termination_index=termination_index,
    )
    EXPLOG.submit_process(process_id=surf_pid)
    updates = EXPLOG.update_log()
    assert updates[surf_pid] == 'completed', (
        f"Surface relaxation must complete in production mode, got: {updates}"
    )

    # _relaxed_surface_magmom is populated for the first time in any pytest:
    relaxed_surface_magmom = surface_study.get_relaxed_surface_magmom()
    assert relaxed_surface_magmom is not None, (
        "Production mode must populate _relaxed_surface_magmom via "
        "set_relaxed_base_surface(require_magmom=True); "
        "this is None only in test mode (require_magmom=False)"
    )
    assert len(relaxed_surface_magmom) == len(surface_study.base_surface_relaxed)
    assert any(abs(m) > 0.1 for m in relaxed_surface_magmom), (
        "Fe slab: relaxed_surface_magmom must contain non-zero Fe values "
        f"from the fake calculator, got {relaxed_surface_magmom}"
    )

    # ------------------------------------------------------------------
    # Phase 3 — O adsorption in production mode
    # ------------------------------------------------------------------
    surface_study.determine_adsorption_sites()
    ad_sites_df = surface_study.add_sites_df_relaxed
    assert ad_sites_df is not None and len(ad_sites_df) >= 1

    for _, row in ad_sites_df.iterrows():
        surface_study.initialize_adsorption_site_study(int(row['Site index']))

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()
    slab_n = len(surface_study.base_surface_relaxed)

    for site_idx in ad_site_dict.keys():
        pid = EXPLOG.add_process(
            candidate_id=cand_id,
            calculation_type="O_adsorption",
            termination_index=termination_index,
            site_index=site_idx,
        )
        EXPLOG.submit_process(process_id=pid)

    EXPLOG.update_log()

    for site_idx, ad_study in ad_site_dict.items():
        # _relaxed_O_magmom comes from real calculator, not the init proxy
        # injected by test mode.
        assert ad_study._relaxed_O_magmom is not None, (
            f"Site {site_idx}: _relaxed_O_magmom must be set from "
            "SinglePointCalculator in production mode"
        )
        if ad_study.site_type == 'on-top':
            # POSCAR for O_adsorption has slab + 1 O atom
            assert len(ad_study._relaxed_O_magmom) == slab_n + 1, (
                f"Site {site_idx} on-top: _relaxed_O_magmom length must be "
                f"{slab_n + 1}, got {len(ad_study._relaxed_O_magmom)}"
            )
            # O is not in _FAKE_MAGMOMS_BY_ELEMENT → magmom = 0.0
            assert ad_study._relaxed_O_magmom[-1] == pytest.approx(0.0, abs=0.1), (
                f"Site {site_idx} on-top: appended O magmom must be 0.0"
            )
            assert any(abs(m) > 0.1 for m in ad_study._relaxed_O_magmom[:-1]), (
                f"Site {site_idx} on-top: slab portion must have non-zero Fe magmoms"
            )

    # ------------------------------------------------------------------
    # Phase 4 — OH adsorption in production mode
    # ------------------------------------------------------------------
    for site_idx in ad_site_dict.keys():
        pids = EXPLOG.add_process(
            candidate_id=cand_id,
            calculation_type="OH_adsorption",
            termination_index=termination_index,
            site_index=site_idx,
        )
        for p in pids:
            EXPLOG.submit_process(process_id=p)

    EXPLOG.update_log()

    for site_idx, ad_study in ad_site_dict.items():
        assert ad_study.relaxed_OH_atoms is not None, (
            f"Site {site_idx}: relaxed_OH_atoms must be stored after "
            "OH_adsorption completes in production mode"
        )


@needs_enumlib
@needs_potcars
def test_production_mode_full_pipeline_al_nm(tmp_path, al_fcc_atoms, monkeypatch):
    """Production-mode pipeline for Al FCC: validates all-zero magmoms are accepted.

    require_magmom=True checks only that a calculator is attached (i.e.
    _relaxed_surface_magmom is not None).  A SinglePointCalculator with
    magmoms=[0.0, ...] satisfies this — non-magnetic materials must complete
    the production pipeline without errors.

    This test verifies the docstring on set_relaxed_base_surface is misleading:
    the code does NOT raise for all-zero magmoms, which is correct behaviour
    for non-magnetic OER candidates such as IrO2 or RuO2.
    """
    _slab_kwargs = {"min_atoms": 4, "min_slab_thickness": 4, "max_layers": 12}

    EXPLOG = Explog()
    EXPLOG.init(
        vasp_calculation_dir=tmp_path,
        mode="production",
        prefix_vasp_dirs=False,
        magnetic_bulk_n_max=4,
    )
    _apply_production_mocks(EXPLOG, monkeypatch)

    study = OER_catalyst_study(
        al_fcc_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    cand_id = "prod_al_nm_test"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="production-mode NM pipeline test",
        notes="monkeypatched VASP, Al FCC (all-zero magmoms)",
        study_obj=study,
    )

    # Phase 1 — bulk
    p_ids = EXPLOG.add_process(candidate_id=cand_id,
                               calculation_type="bulk_relaxation")
    for pid in p_ids:
        EXPLOG.submit_process(process_id=pid)
    EXPLOG.update_log()
    assert study.is_mag_bulk_batch_finalized()

    bulk_magmom = study.get_bulk_magmom()
    assert bulk_magmom is not None
    # Al is not in _FAKE_MAGMOMS_BY_ELEMENT → all magmoms are 0.0
    assert all(m == pytest.approx(0.0, abs=1e-6) for m in bulk_magmom), (
        f"Al production mode: all bulk magmoms must be 0.0, got {bulk_magmom}"
    )

    # Phase 2 — surface relaxation: require_magmom=True must NOT raise for
    # all-zero magmoms — a calculator IS attached, so _relaxed_surface_magmom
    # is set to [0.0, ...], which is not None.
    study.predict_most_likely_surfaces(max_miller=1, **_slab_kwargs)
    ranking_df = study.get_termination_rankings()
    valid_rows = ranking_df[ranking_df['Score'].notna()]
    assert len(valid_rows) >= 1
    termination_index = int(valid_rows.iloc[0]['Termination index'])
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    surf_pid = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="surface_relaxation",
        termination_index=termination_index,
    )
    EXPLOG.submit_process(process_id=surf_pid)
    EXPLOG.update_log()

    relaxed_surface_magmom = surface_study.get_relaxed_surface_magmom()
    assert relaxed_surface_magmom is not None, (
        "NM material: _relaxed_surface_magmom must be a list (even all "
        "zeros) — require_magmom=True must not raise for all-zero magmoms "
        "from a non-magnetic material"
    )
    assert all(m == pytest.approx(0.0, abs=1e-6) for m in relaxed_surface_magmom), (
        f"Al is non-magnetic: all relaxed surface magmoms must be 0.0, "
        f"got {relaxed_surface_magmom}"
    )


@needs_enumlib
@needs_potcars
def test_production_mode_unrecoverable_surface_fe(tmp_path, fe_bcc_atoms,
                                                   monkeypatch):
    """Disconnected slab is marked unrecoverable, not silently completed (BCC Fe).

    Exercises job_handler.py:1247–1252 and 1284–1317 — the unrecoverable_reason
    path — which has zero coverage in all other tests.  A silent bug here would
    allow a dissociated surface to pass into adsorption studies and produce
    invalid OER energies.

    The fake reader displaces one slab atom by 4 Å in z to break connectivity.
    4 Å > bond-length cutoff (~3.5 Å for Fe), so are_all_atoms_connected()
    returns False.  4 Å is also small enough not to wrap through the vacuum
    layer via PBC, which a 25 Å displacement would risk.
    """
    _slab_kwargs = {"min_atoms": 4, "min_slab_thickness": 4, "max_layers": 12}

    EXPLOG = Explog()
    EXPLOG.init(
        vasp_calculation_dir=tmp_path,
        mode="production",
        prefix_vasp_dirs=False,
        magnetic_bulk_n_max=4,
    )
    # Bulk phases use the normal reader; surface uses the disconnected reader.
    _apply_production_mocks(
        EXPLOG, monkeypatch,
        result_reader=_job_type_aware_disconnected_reader,
    )

    study = OER_catalyst_study(
        fe_bcc_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    cand_id = "prod_fe_unrecoverable"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="unrecoverable-surface path test",
        notes="surface atom displaced 6 Å to break connectivity",
        study_obj=study,
    )

    # Bulk phases must complete normally before surface is submitted.
    p_ids = EXPLOG.add_process(candidate_id=cand_id,
                               calculation_type="bulk_relaxation")
    for pid in p_ids:
        EXPLOG.submit_process(process_id=pid)
    EXPLOG.update_log()
    assert study.is_mag_bulk_batch_finalized()

    study.predict_most_likely_surfaces(max_miller=1, **_slab_kwargs)
    ranking_df = study.get_termination_rankings()
    valid_rows = ranking_df[ranking_df['Score'].notna()]
    assert len(valid_rows) >= 1
    termination_index = int(valid_rows.iloc[0]['Termination index'])
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    surf_pid = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="surface_relaxation",
        termination_index=termination_index,
    )
    EXPLOG.submit_process(process_id=surf_pid)
    updates = EXPLOG.update_log()

    assert updates[surf_pid] == 'unrecoverable', (
        "A disconnected relaxed surface must be marked 'unrecoverable', "
        f"not silently passed as 'completed'. Got: {updates}"
    )

    proc_df = EXPLOG.relational_frame.processes.df
    status = proc_df.loc[
        proc_df['process_id'] == surf_pid, 'status'
    ].values[0]
    assert status.startswith('unrecoverable:'), (
        f"Status column must contain the reason string, got: '{status}'"
    )

    assert surface_study.study_locked is True, (
        "study_locked must be True after a disconnected surface was submitted"
    )
    assert surface_study.add_sites_df_relaxed is None, (
        "Adsorption sites must not be determined on a disconnected surface"
    )
