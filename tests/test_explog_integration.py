# Integration tests for the Explog workflow.
#
# These tests verify that the full Explog pipeline — from candidate registration
# through bulk relaxation, surface relaxation, and O-adsorption — works correctly
# end-to-end in "test" mode (no real VASP jobs; all submissions complete immediately).
#
# Key behavioural contract tested here:
#  - add_process("bulk_relaxation") returns list[int], one ID per magnetic config.
#    For NM materials this is always a single-element list, but the tests treat
#    it generically so they remain valid if magnetic materials are introduced.
#  - Overpotential metrics (idealOverPotential, Overpotential_from_scaling) stay
#    NaN until both O and OH stages are complete — only G(O) deviation is populated
#    after the O stage.
#  - Process IDs are globally unique; duplicate submissions raise rather than silently
#    overwrite existing records.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from gnome_dreams_oer_screening import Explog, OER_catalyst_study, explog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_study_from_atoms(atoms: Atoms) -> OER_catalyst_study:
    """Create a fresh OER_catalyst_study from an ASE Atoms object.
    The study has no relaxed bulk yet — that is handled by the Explog workflow."""
    return OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )


def _prepare_candidate_for_surface_stage(explog: Explog, candidate_id: str) -> None:
    """Predict surfaces and initialise the surface study for termination 0.
    Called between the bulk-relaxation update_log and the surface-relaxation
    add_process — mirrors the user-facing workflow."""
    study = explog.relational_frame.candidates[candidate_id].study_obj
    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)


def _prepare_candidate_for_o_adsorption_stage(
    explog: Explog, candidate_id: str, only_reduced_coord_o_sites: bool = True
) -> int:
    """Determine adsorption sites and initialise the first site study.
    Returns the site_index used so the caller can add the O_adsorption process."""
    study = explog.relational_frame.candidates[candidate_id].study_obj
    surface_study = study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=only_reduced_coord_o_sites
    )
    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)
    return site_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tio2_rutile_atoms() -> Atoms:
    dir_path = Path(__file__).resolve().parent
    path = dir_path / "../notebooks/TiO2_rutile.cif"
    return read(path.resolve(), format="cif")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_end_to_end_test_mode_process_keeps_candidate_overpotential_missing_until_oh(
    tmp_path,
    tio2_rutile_atoms,
):
    """Full bulk → surface → O pipeline in test mode.

    Verifies the core Explog contract:
      - After the O stage, G(O) deviation is a finite number.
      - idealOverPotential and Overpotential_from_scaling remain NaN because
        the OH stage has not been run — overpotentials require both O and OH.
    """
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "some_candidate_blablabla"
    study = OER_catalyst_study(
        tio2_rutile_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="blablabla",
        notes="validate job flow",
        study_obj=study,
    )

    # add_process("bulk_relaxation") returns list[int] — one process per magnetic
    # config. For TiO2 (NM) this is a single-element list, but we iterate generically.
    ids = explog.add_process(candidate_id=candidate_id, calculation_type="bulk_relaxation")
    id = ids[0]
    explog.submit_process(id)
    updates = explog.update_log()
    assert updates[id] == "completed"

    candidate_study = explog.relational_frame.candidates[candidate_id].study_obj
    candidate_study.predict_most_likely_surfaces(max_miller=1)
    candidate_study.initialize_oer_surface_study(0)

    p_surface = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="surface_relaxation",
        termination_index=0,
    )
    explog.submit_process(p_surface)
    updates = explog.update_log()
    assert updates[p_surface] == "completed"

    surface_study = candidate_study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=True)
    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)

    p_o = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="O_adsorption",
        termination_index=0,
        site_index=site_index,
    )
    explog.submit_process(p_o)
    updates = explog.update_log()
    assert updates[p_o] == "completed"

    candidate_row = explog.relational_frame.candidates[candidate_id]
    assert np.isfinite(float(candidate_row.df["G(O) deviation"].iloc[0]))
    assert pd.isna(candidate_row.idealOverPotential)
    assert pd.isna(candidate_row.Overpotential_from_scaling)


def test_many_candidates_many_bulk_processes_are_unique_and_not_overwritten(
    tmp_path, tio2_rutile_atoms
):
    """Process integrity under concurrent bulk submissions for many candidates.

    Registers 100 candidates and submits a bulk_relaxation process for each.
    Verifies:
      - Duplicate candidate registration raises ValueError and does not modify
        the candidates table.
      - Every bulk process ID is unique across all candidates.
      - All processes complete and the processes table contains exactly one row
        per process ID, correctly mapped to its candidate.
    """
    explog = Explog()
    explog.init(tmp_path, mode="test")

    num_candidates = 11
    candidate_ids = [f"cand_bulk_{i}" for i in range(num_candidates)]

    for candidate_id in candidate_ids:
        explog.add_candidate(
            candidate_id=candidate_id,
            reason_or_hypothesis="bulk screening candidate",
            notes="many candidates test",
            study_obj=_new_study_from_atoms(tio2_rutile_atoms),
        )

    cdf = explog.relational_frame.candidates.df
    assert len(cdf) == num_candidates
    assert set(cdf["candidate_id"].tolist()) == set(candidate_ids)

    with pytest.raises(ValueError, match="already exists"):
        explog.add_candidate(
            candidate_id=candidate_ids[0],
            reason_or_hypothesis="duplicate",
            notes="duplicate",
            study_obj=_new_study_from_atoms(tio2_rutile_atoms),
        )
    # Test that the failed duplicate did not enter the candidates table:
    assert len(explog.relational_frame.candidates.df) == num_candidates

    # add_process returns list[int] for bulk_relaxation; flatten to a single list
    # and build a reverse map from process_id back to its candidate.
    expected_pid_to_candidate = {}
    process_ids = []
    for candidate_id in candidate_ids:
        ids = explog.add_process(
            candidate_id=candidate_id, calculation_type="bulk_relaxation"
        )
        process_ids.extend(ids)
        for pid in ids:
            expected_pid_to_candidate[pid] = candidate_id

    for process_id in process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()

    assert len(process_ids) == num_candidates  # exactly one process per candidate
    assert len(set(process_ids)) == len(process_ids)  # all IDs are unique
    assert set(updates.keys()) == set(process_ids) # The updated processes are exactly those we submitted
    assert set(updates.values()) == {"completed"} # All are completed

    pdf = explog.relational_frame.processes.df.copy()
    assert len(pdf) == len(process_ids)
    assert not pdf["process_id"].duplicated().any()

    for process_id, candidate_id in expected_pid_to_candidate.items():
        row = pdf[pdf["process_id"] == process_id].iloc[0]
        assert row["candidate_id"] == candidate_id
        assert row["job_type"] == "bulk_relaxation"
        assert row["status"] == "completed"


def test_many_candidates_many_processes_compute_o_energies_but_not_overpotentials_without_oh(
    tmp_path, tio2_rutile_atoms
):
    """Full bulk → surface → O pipeline for multiple candidates in parallel.

    Verifies that after the O stage:
      - G(O) and G(O) deviation are populated for all candidates.
      - ideal overpotential remains NaN for all sites (OH not yet computed).
      - All process IDs are globally unique and no row is duplicated in the
        processes table.
    """
    explog = Explog()
    explog.init(tmp_path, mode="test")

    num_candidates = 4
    candidate_ids = [f"cand_full_{i}" for i in range(num_candidates)]

    for candidate_id in candidate_ids:
        explog.add_candidate(
            candidate_id=candidate_id,
            reason_or_hypothesis="full workflow candidate",
            notes="many process regression",
            study_obj=_new_study_from_atoms(tio2_rutile_atoms),
        )

    # Bulk: add_process returns list[int]; flatten across all candidates.
    bulk_process_ids = []
    for candidate_id in candidate_ids:
        bulk_process_ids.extend(
            explog.add_process(candidate_id=candidate_id, calculation_type="bulk_relaxation")
        )
    for process_id in bulk_process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()
    assert set(updates.keys()) == set(bulk_process_ids)
    assert set(updates.values()) == {"completed"}

    for candidate_id in candidate_ids:
        _prepare_candidate_for_surface_stage(explog, candidate_id)

    # Surface: add_process returns a single int for surface_relaxation.
    surface_process_ids = []
    for candidate_id in candidate_ids:
        surface_process_ids.append(
            explog.add_process(
                candidate_id=candidate_id,
                calculation_type="surface_relaxation",
                termination_index=0,
            )
        )
    for process_id in surface_process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()
    assert set(updates.keys()) == set(surface_process_ids)
    assert set(updates.values()) == {"completed"}

    site_indices = {}
    for candidate_id in candidate_ids:
        site_indices[candidate_id] = _prepare_candidate_for_o_adsorption_stage(
            explog, candidate_id, only_reduced_coord_o_sites=True
        )

    # O adsorption: add_process returns a single int for O_adsorption.
    o_process_ids = []
    for candidate_id in candidate_ids:
        o_process_ids.append(
            explog.add_process(
                candidate_id=candidate_id,
                calculation_type="O_adsorption",
                termination_index=0,
                site_index=site_indices[candidate_id],
            )
        )
    for process_id in o_process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()
    assert set(updates.keys()) == set(o_process_ids)
    assert set(updates.values()) == {"completed"}

    all_process_ids = bulk_process_ids + surface_process_ids + o_process_ids
    assert len(set(all_process_ids)) == len(all_process_ids)  # globally unique

    pdf = explog.relational_frame.processes.df.copy()
    assert len(pdf) == len(all_process_ids)
    assert not pdf["process_id"].duplicated().any()
    assert set(pdf["status"].tolist()) == {"completed"}

    for candidate_id in candidate_ids:
        candidate_row = explog.relational_frame.candidates[candidate_id]
        assert np.isfinite(float(candidate_row.df["G(O) deviation"].iloc[0]))
        assert pd.isna(candidate_row.idealOverPotential)
        assert pd.isna(candidate_row.Overpotential_from_scaling)

        study = candidate_row.study_obj
        surface_study = study.get_surface_studies()[0]
        df = surface_study.get_adsorption_sites_df()

        assert "G(O)" in df.columns
        assert "G(O) deviation" in df.columns
        assert "ideal overpotential" in df.columns
        assert df["G(O)"].notna().any()
        assert df["G(O) deviation"].notna().any()
        assert df["ideal overpotential"].isna().all()


def test_processes_are_not_overwritten_when_duplicate_process_requested(
    tmp_path, tio2_rutile_atoms
):
    """Duplicate process guard: re-requesting a completed bulk process raises RuntimeError.

    After a bulk_relaxation completes successfully, a second add_process call for
    the same candidate and job type must raise rather than silently add a second row.
    The processes table must be identical before and after the failed call.
    """
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "cand_duplicate_process"
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="duplicate process guard",
        notes="ensure no overwrite",
        study_obj=_new_study_from_atoms(tio2_rutile_atoms),
    )

    process_ids = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="bulk_relaxation",
    )
    for p_id in process_ids:
        explog.submit_process(p_id)
    explog.update_log()

    before = explog.relational_frame.processes.df.copy()
    with pytest.raises(RuntimeError, match="non-failed calculation"):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="bulk_relaxation",
        )
    after = explog.relational_frame.processes.df.copy()

    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_update_candidate_oer_metrics_handles_no_valid_o_energies_without_raising(
    tmp_path, tio2_rutile_atoms
):
    """OER metric update is a no-op when no O energies have been computed.

    Sets up a study with a relaxed bulk and surface (bypassing VASP via
    use_init_as_relaxed_bulk=True and require_magmom=False) but does not run
    any O-adsorption calculations. Calls _update_candidate_oer_metrics directly
    and verifies that it returns without raising and leaves all metrics NaN.

    This guards against a regression where the metric update function crashed
    when iterating over an empty or all-NaN G(O) column.
    """
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "cand_no_valid_o"
    # use_init_as_relaxed_bulk bypasses the magnetic pipeline — appropriate here
    # because we are testing the OER metric update path, not the bulk workflow.
    study = OER_catalyst_study(
        tio2_rutile_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="no valid O energies path",
        notes="regression for candidate oer metric updates",
        study_obj=study,
    )

    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    # require_magmom=False: no VASP calculator attached in test mode
    surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.135,
        require_magmom=False,
    )
    surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=True)

    candidate_row_before = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(candidate_row_before.df["G(O) deviation"].iloc[0])
    assert pd.isna(candidate_row_before.idealOverPotential)

    explog.job_handler._update_candidate_oer_metrics(candidate_id)

    candidate_row_after = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(candidate_row_after.df["G(O) deviation"].iloc[0])
    assert pd.isna(candidate_row_after.idealOverPotential)
