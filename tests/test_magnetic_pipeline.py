# End-to-end tests for the magnetic bulk → surface → adsorbate pipeline.
#
# These tests call MagneticStructureEnumerator WITHOUT mocking, so they
# require the enumlib binaries (enum.x, makestr.x / multienum.x) to be
# on PATH. They are skipped automatically when those binaries are absent
# (e.g. on Windows developer machines).
#
# Coverage per run:
#   Phase 1  — pre-add_process state assertions
#   Phase 2  — add_process creates one bulk process per magnetic config
#   Phase 3  — submit + update_log → finalization, winning magmoms stored
#   Phase 4  — predict_most_likely_surfaces, initialize surface study,
#              slab magmom matches bulk
#   Phase 5  — surface_relaxation (test mode) → determine_adsorption_sites,
#              get_O_init_magmom length and value checks
#   Phase 6  — O_adsorption (test mode) → _relaxed_O_magmom injected,
#              get_OH_init_magmom length and value checks for all site types

import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
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
