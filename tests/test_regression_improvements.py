import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from gnome_dreams_oer_screening import Explog, OER_catalyst_study
from gnome_dreams_oer_screening.oer.ad_site_determination import (
    determine_adsorption_sites,
)


def _new_study_from_atoms(atoms: Atoms) -> OER_catalyst_study:
    return OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )


def _prepare_candidate_for_surface_stage(explog: Explog, candidate_id: str) -> None:
    study = explog.relational_frame.candidates[candidate_id].study_obj
    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)


def _prepare_candidate_for_o_adsorption_stage(
    explog: Explog, candidate_id: str, only_reduced_coord_o_sites: bool = True
) -> int:
    study = explog.relational_frame.candidates[candidate_id].study_obj
    surface_study = study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=only_reduced_coord_o_sites
    )
    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)
    return site_index


@pytest.fixture(scope="module")
def tio2_rutile_atoms() -> Atoms:
    dir_path = Path(__file__).resolve().parent
    path = dir_path / "../notebooks/TiO2_rutile.cif"
    return read(path.resolve(), format="cif")


@pytest.fixture(scope="module")
def prepared_surface_study(tio2_rutile_atoms):
    study = OER_catalyst_study(
        tio2_rutile_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    study.set_relaxed_structure(study.get_init_atoms())
    study.predict_most_likely_surfaces(method="O-coordination")
    study.initialize_oer_surface_study(0)

    surface_study = study.get_surface_studies()[0]
    surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.135,
    )
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites=False,
    )
    return study, surface_study


def test_getters_return_none_when_not_initialized():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
    )

    assert study.get_bulk_atoms() is None
    assert study.get_bulk_structure() is None
    assert study.get_distinct_millers() is None


def test_getters_return_copies_after_initialization():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    study = OER_catalyst_study(
        atoms,
        H2_gas_free_energy=-2.0,
        H2O_gas_free_energy=-4.0,
    )
    study.set_relaxed_structure(atoms.copy())

    bulk_atoms = study.get_bulk_atoms()
    assert bulk_atoms is not None
    assert bulk_atoms is not study.bulk_atoms

    bulk_atoms.positions[0, 0] += 0.1
    assert not np.isclose(
        bulk_atoms.positions[0, 0], study.bulk_atoms.positions[0, 0]
    )

    bulk_structure = study.get_bulk_structure()
    assert bulk_structure is not None
    assert bulk_structure is not study.bulk_structure


def test_determine_adsorption_sites_rejects_atoms_input():
    atoms = Atoms(
        "TiO2",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[5, 5, 5],
        pbc=True,
    )
    with pytest.raises(
        AssertionError, match="Input surface must be a pymatgen Structure or Slab object"
    ):
        determine_adsorption_sites(atoms, normal_vector=np.array([0.0, 0.0, 1.0]))


def test_adsorption_site_dataframe_has_stable_column_name(prepared_surface_study):
    _, surface_study = prepared_surface_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    assert isinstance(df, pd.DataFrame)
    assert "ad site element" in df.columns
    assert "add site element" not in df.columns


def test_update_adsorption_energies_skips_incomplete_studies(prepared_surface_study):
    _, surface_study = prepared_surface_study
    surface_study = copy.deepcopy(surface_study)

    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert pd.isna(row["G(O)"])
    assert pd.isna(row["ideal overpotential"])


def test_update_adsorption_energies_computes_o_only_overpotential(
    prepared_surface_study,
):
    study, surface_study = prepared_surface_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    on_top_site_index = int(df[df["site type"] == "on-top"]["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(on_top_site_index)

    ad_study = surface_study.get_adsorption_site_studies_dict()[on_top_site_index]
    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    target_g_o = 2.46
    constants = study.get_gas_and_energy_corrections()
    e_add = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )
    ad_study.set_relaxed_O_calc(ad_study.get_atoms_for_O_calc(), energy=e_add)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[on_top_site_index]

    assert row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert row["ideal overpotential"] == pytest.approx(0.0, abs=1e-8)
    assert pd.isna(row["G(OH)"])


def test_end_to_end_test_mode_process_updates_candidate_overpotential(
    tmp_path,
    tio2_rutile_atoms,
):
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "candidate_regression"
    study = OER_catalyst_study(
        tio2_rutile_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
    )
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="regression test candidate",
        notes="validate job flow",
        study_obj=study,
    )

    p_bulk = explog.add_process(candidate_id=candidate_id, calculation_type="bulk_relaxation")
    explog.submit_process(p_bulk)
    updates = explog.update_log()
    assert updates[p_bulk] == "completed"

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
    assert candidate_row.idealOverPotential is not None
    assert np.isfinite(float(candidate_row.idealOverPotential))


def test_many_candidates_many_bulk_processes_are_unique_and_not_overwritten(
    tmp_path, tio2_rutile_atoms
):
    explog = Explog()
    explog.init(tmp_path, mode="test")

    num_candidates = 10
    candidate_ids = [f"cand_bulk_{i}" for i in range(num_candidates)]

    for candidate_id in candidate_ids:
        explog.add_candidate(
            candidate_id=candidate_id,
            reason_or_hypothesis="bulk screening candidate",
            notes="many candidate regression",
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
    assert len(explog.relational_frame.candidates.df) == num_candidates

    expected_pid_to_candidate = {}
    process_ids = []
    for candidate_id in candidate_ids:
        process_id = explog.add_process(
            candidate_id=candidate_id, calculation_type="bulk_relaxation"
        )
        process_ids.append(process_id)
        expected_pid_to_candidate[process_id] = candidate_id

    for process_id in process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()

    assert len(process_ids) == num_candidates
    assert len(set(process_ids)) == len(process_ids)
    assert set(updates.keys()) == set(process_ids)
    assert set(updates.values()) == {"completed"}

    pdf = explog.relational_frame.processes.df.copy()
    assert len(pdf) == num_candidates
    assert not pdf["process_id"].duplicated().any()

    for process_id, candidate_id in expected_pid_to_candidate.items():
        row = pdf[pdf["process_id"] == process_id].iloc[0]
        assert row["candidate_id"] == candidate_id
        assert row["job_type"] == "bulk_relaxation"
        assert row["status"] == "completed"


def test_many_candidates_many_processes_compute_o_energies_and_overpotentials(
    tmp_path, tio2_rutile_atoms
):
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

    bulk_process_ids = []
    for candidate_id in candidate_ids:
        bulk_process_ids.append(
            explog.add_process(candidate_id=candidate_id, calculation_type="bulk_relaxation")
        )
    for process_id in bulk_process_ids:
        explog.submit_process(process_id)
    updates = explog.update_log()
    assert set(updates.keys()) == set(bulk_process_ids)
    assert set(updates.values()) == {"completed"}

    for candidate_id in candidate_ids:
        _prepare_candidate_for_surface_stage(explog, candidate_id)

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
    assert len(all_process_ids) == 3 * num_candidates
    assert len(set(all_process_ids)) == len(all_process_ids)

    pdf = explog.relational_frame.processes.df.copy()
    assert len(pdf) == 3 * num_candidates
    assert not pdf["process_id"].duplicated().any()
    assert set(pdf["status"].tolist()) == {"completed"}

    for candidate_id in candidate_ids:
        candidate_row = explog.relational_frame.candidates[candidate_id]
        assert candidate_row.idealOverPotential is not None
        assert np.isfinite(float(candidate_row.idealOverPotential))

        study = candidate_row.study_obj
        surface_study = study.get_surface_studies()[0]
        df = surface_study.get_adsorption_sites_df()

        assert "G(O)" in df.columns
        assert "ideal overpotential" in df.columns
        assert df["G(O)"].notna().any()
        assert df["ideal overpotential"].notna().any()


def test_processes_are_not_overwritten_when_duplicate_process_requested(
    tmp_path, tio2_rutile_atoms
):
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "cand_duplicate_process"
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="duplicate process guard",
        notes="ensure no overwrite",
        study_obj=_new_study_from_atoms(tio2_rutile_atoms),
    )

    process_id = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="bulk_relaxation",
    )
    explog.submit_process(process_id)
    explog.update_log()

    before = explog.relational_frame.processes.df.copy()
    with pytest.raises(RuntimeError, match="non-failed calculation"):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="bulk_relaxation",
        )
    after = explog.relational_frame.processes.df.copy()

    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_update_ideal_over_pot_handles_no_valid_o_energies_without_raising(
    tmp_path, tio2_rutile_atoms
):
    explog = Explog()
    explog.init(tmp_path, mode="test")

    candidate_id = "cand_no_valid_o"
    study = OER_catalyst_study(
        tio2_rutile_atoms,
        H2_gas_free_energy=-6.77,
        H2O_gas_free_energy=-14.217,
        use_init_as_relaxed_bulk=True,
    )
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="no valid O energies path",
        notes="regression for _update_ideal_over_pot",
        study_obj=study,
    )

    study.predict_most_likely_surfaces(max_miller=1)
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(),
        energy=-278.135,
    )
    surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=True)

    candidate_row_before = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(candidate_row_before.idealOverPotential)

    explog.job_handler._update_ideal_over_pot(candidate_id)

    candidate_row_after = explog.relational_frame.candidates[candidate_id]
    assert pd.isna(candidate_row_after.idealOverPotential)


def test_update_ideal_over_pot_skips_invalid_surfaces_and_uses_min_valid_value(
    tmp_path,
):
    class DummySurfaceStudy:
        def __init__(self, df):
            self._df = df
            self.update_calls = 0

        def update_adsorption_energies(self):
            self.update_calls += 1

        def get_adsorption_sites_df(self):
            return self._df

    class DummyStudy:
        def __init__(self, surfaces):
            self._surfaces = surfaces

        def get_surface_studies(self):
            return self._surfaces

    surface_none = DummySurfaceStudy(None)
    surface_no_column = DummySurfaceStudy(pd.DataFrame({"x": [1, 2]}))
    surface_valid_1 = DummySurfaceStudy(
        pd.DataFrame({"ideal overpotential": [np.nan, 0.47, 0.53]})
    )
    surface_valid_2 = DummySurfaceStudy(
        pd.DataFrame({"ideal overpotential": [0.333, 0.4]})
    )

    dummy_study = DummyStudy(
        {
            0: surface_none,
            1: surface_no_column,
            2: surface_valid_1,
            3: surface_valid_2,
        }
    )

    explog = Explog()
    explog.init(tmp_path, mode="test")
    candidate_id = "cand_dummy_surfaces"
    explog.add_candidate(
        candidate_id=candidate_id,
        reason_or_hypothesis="dummy surface test",
        notes="test _update_ideal_over_pot guards",
        study_obj=dummy_study,
    )

    explog.job_handler._update_ideal_over_pot(candidate_id)

    candidate_row = explog.relational_frame.candidates[candidate_id]
    assert candidate_row.idealOverPotential == pytest.approx(0.33, abs=1e-12)

    assert surface_none.update_calls == 1
    assert surface_no_column.update_calls == 1
    assert surface_valid_1.update_calls == 1
    assert surface_valid_2.update_calls == 1
