# Tests for the magnetic bulk relaxation pipeline.
#
# These tests call MagneticStructureEnumerator WITHOUT mocking, so they
# require the enumlib binaries (enum.x, makestr.x / multienum.x) to be
# on PATH. They are skipped automatically when those binaries are absent
# (e.g. on Windows developer machines).
#
# Guards against regressions in the multi-config bulk path:
#   - enumerate_magnetic_structures produces distinct FM/AFM configs
#   - add_process("bulk_relaxation") creates one process per config
#   - each processNote carries the correct mag_config_index
#   - finalize_mag_bulk_batch runs when all bulk jobs are terminal
#   - the winning magmoms are propagated to bulk_structure.site_properties
#   - SlabGenerator propagates those magmoms to the slab atoms

import shutil

import numpy as np
import pytest
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import Explog, OER_catalyst_study
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
    extract_mag_config_index,
)


pytestmark = pytest.mark.mini


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
    return bulk('Fe', 'bcc', a=2.87, cubic=True)  # 2 atoms


# ---------------------------------------------------------------------------
# Reusable pipeline helper
# ---------------------------------------------------------------------------

def _run_magnetic_bulk_pipeline(
    atoms,
    n_expected: int,
    tmp_path,
    n_max: int = 4,
) -> None:
    """Run the full magnetic bulk → surface pipeline and assert correctness.

    Args:
        atoms:      ASE Atoms for the bulk structure under test.
        n_expected: Number of magnetic configs expected from
                    enumerate_magnetic_structures(structure, n_max=n_max).
                    The caller must supply this; the function verifies it.
        tmp_path:   Temporary directory (pytest tmp_path fixture or similar).
        n_max:      Cap on magnetic configs passed to Explog (default 4,
                    matches Explog.init default).
    """
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

    # add_process creates one process per enumerated config (capped at n_max).
    # n_expected was verified against enumerate_magnetic_structures above, so
    # len(p_ids) == n_expected holds whenever n_expected <= n_max.
    assert len(p_ids) == n_expected, (
        f"Expected {n_expected} bulk processes (one per magnetic config), "
        f"got {len(p_ids)}"
    )

    # get_mag_bulk_configs() returns the INITIAL (pre-relaxation) enumerated
    # configs stored by store_mag_bulk_configs immediately after add_process.
    # These are the input magnetic configurations before any DFT or test-mode
    # run has modified them.
    stored_configs = study.get_mag_bulk_configs()
    assert stored_configs is not None
    assert len(stored_configs) == n_expected

    # Each stored config must have the required keys and correct magmom length:
    for i, cfg in enumerate(stored_configs):
        n_atoms_in_config = len(cfg['structure'].sites)
        assert len(cfg['magmom']) == n_atoms_in_config, (
            f"Stored config {i}: magmom length mismatch"
        )
        assert any(abs(m) > 1 for m in cfg['magmom']), (
            f"Stored config {i}: expected non-zero magmoms, got {cfg['magmom']}"
        )

    # Batch still not finalized (no jobs run yet):
    assert study.is_mag_bulk_batch_finalized() == False

    # processNote of each process row must contain mag_config_index=N,
    # with all N distinct and covering 0..n_expected-1:
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
    # All bulk jobs complete immediately in test mode:
    assert all(v == 'completed' for v in updates.values()), (
        f"Expected all bulk processes to complete in test mode, got: {updates}"
    )
    assert len(updates) == n_expected, (
        f"Expected {n_expected} updates, got {len(updates)}"
    )

    # Batch finalized after all bulk processes reach terminal status:
    assert study.is_mag_bulk_batch_finalized() == True, (
        "finalize_mag_bulk_batch must be called once all bulk processes complete"
    )

    # get_bulk_magmom() is now set:
    bulk_magmom = study.get_bulk_magmom()
    assert bulk_magmom is not None
    assert len(bulk_magmom) <= n_atoms, (
        f"get_bulk_magmom() should have at most {n_atoms} entries, got {len(bulk_magmom)}"
    )

    # In test mode all fake energies are 0 → min is stable at config 0:
    winning_magmom = stored_configs[0]['magmom']
    assert bulk_magmom == winning_magmom, (
        f"In test mode (all E=0), config 0 wins. "
        f"Expected magmoms {winning_magmom}, got {bulk_magmom}"
    )

    # Some winning magmoms must be non-zero:
    assert any(abs(m) > 1 for m in bulk_magmom), (
        f"Winning bulk magmoms must all be non-zero, got {bulk_magmom}"
    )

    # bulk_structure.site_properties["magmom"] must be set to winning magmoms:
    bulk_struct = study.get_bulk_structure()
    assert bulk_struct is not None
    site_magmoms = [s.properties.get('magmom') for s in bulk_struct.sites]
    assert site_magmoms == winning_magmom, (
        "bulk_structure site_properties['magmom'] must equal the winning magmoms"
    )

    # -----------------------------------------------------------------------
    # Phase 4 — rank terminations, pick one with a finite score, init surface.
    # predict_most_likely_surfaces must be called before any
    # initialize_oer_surface_study call; it raises RuntimeError otherwise.
    # -----------------------------------------------------------------------
    study.predict_most_likely_surfaces(max_miller=1)

    ranking_df = study.get_termination_rankings()
    assert ranking_df is not None, (
        "get_termination_rankings() must return a DataFrame after "
        "predict_most_likely_surfaces has run"
    )
    assert len(ranking_df) >= 1, "At least one termination must be ranked"

    # Select the first termination that has a finite (non-NaN) score:
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

    # Build a reference dict: element → set of |magmom| values from the
    # winning bulk configuration. Every slab atom of that element must have
    # a |magmom| that matches one of the reference values to within tolerance.
    # Uses original_slab.sites (pymatgen PeriodicSite) to match the same
    # atom ordering that get_slab_magmom() reads from.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@needs_enumlib
def test_enumerate_magnetic_structures_fe(fe_bcc_atoms):
    """enumerate_magnetic_structures returns ≥ 2 distinct configs for BCC Fe.

    Verifies keys, magmom length, non-zero values, site_properties storage,
    distinct patterns, presence of FM (all-same-sign) and AFM (mixed-sign).
    """
    structure = AseAtomsAdaptor.get_structure(fe_bcc_atoms)
    configs = enumerate_magnetic_structures(structure)

    assert len(configs) >= 2, (
        f"Expected at least 2 configs for BCC Fe, got {len(configs)}"
    )

    n_atoms = len(fe_bcc_atoms)

    for i, cfg in enumerate(configs):
        assert 'structure' in cfg
        assert 'magmom' in cfg
        assert 'ordering' in cfg

        assert len(cfg['magmom']) == n_atoms, (
            f"Config {i}: magmom length {len(cfg['magmom'])} != n_atoms {n_atoms}"
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

    # Configs must be distinct (different magmom patterns):
    magmom_tuples = [tuple(c['magmom']) for c in configs]
    assert len(magmom_tuples) == len(configs), (
        "All configs must have distinct magmom patterns"
    )

    # At least one AFM-like config (mixed signs):
    has_afm = any(
        any(m > 0.1 for m in c['magmom']) and any(m < -0.1 for m in c['magmom'])
        for c in configs
    )
    assert has_afm, (
        "Expected at least one AFM-like config (mixed signs) for BCC Fe"
    )

    # At least one FM config (all positive):
    has_fm = any(all(m > 0.1 for m in c['magmom']) for c in configs)
    assert has_fm, (
        "Expected at least one FM config (all positive) for BCC Fe"
    )


@needs_enumlib
def test_magnetic_bulk_pipeline_fe(tmp_path, fe_bcc_atoms):
    """End-to-end magnetic bulk pipeline for 2-atom BCC Fe.

    BCC Fe in a 2-atom conventional cell yields exactly 2 magnetic
    orderings (FM + AFM), both within the default n_max=4 cap.
    """
    _run_magnetic_bulk_pipeline(fe_bcc_atoms, n_expected=2, tmp_path=tmp_path)
