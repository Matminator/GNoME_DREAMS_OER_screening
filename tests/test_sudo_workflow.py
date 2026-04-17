# --- standard library imports ---
import numpy as np
import pandas as pd
from pathlib import Path

from ase import Atoms
from ase.io import read, Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
import pytest

# --- IMPORTS FROM THIS PROJECT ---
from gnome_dreams_oer_screening import OER_catalyst_study
from gnome_dreams_oer_screening import Explog

from gnome_dreams_oer_screening.vasp.job_handler import Job_handler


def _new_oer_study(atoms: Atoms) -> OER_catalyst_study:
    return OER_catalyst_study(
        atoms.copy(),
        H2_gas_free_energy=-2,
        H2O_gas_free_energy=-4,
    )


@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    if atoms is None:
        pytest.fail("Slab generation failed")
    return atoms

def test_multiple_bulk_jobs_submitted(tmp_path, TiO2_r_bulk):
    NUM_PROCESSES = 10

    atoms = TiO2_r_bulk
    EXPLOG = Explog()
    EXPLOG.init(tmp_path, 'test')

    cand_ids = []
    for i in range(NUM_PROCESSES):
        cand_id = "dummy_id_" + str(i)
        cand_ids.append(cand_id)

        EXPLOG.add_candidate(
            candidate_id=cand_id,
            reason_or_hypothesis="Ir sites should be great", 
            notes="this is a great candidate", 
            study_obj=_new_oer_study(atoms)
        )

    p_ids = []
    for cand_id in cand_ids:
        ids = EXPLOG.add_process(
            candidate_id=cand_id,
            calculation_type="bulk_relaxation",
        )
        for pid in ids:
            EXPLOG.submit_process(process_id=pid)
        p_ids.extend(ids)

    pdf = EXPLOG.relational_frame.processes.df
    pdf_process_ids = pdf['process_id'].tolist()

    # Test that the correct number of processes have been submitted:
    assert len(p_ids) == NUM_PROCESSES
    assert len(pdf_process_ids) == NUM_PROCESSES

    # Test that all process ids are in the dataframe of processes:
    for p_id in p_ids:
        assert p_id in pdf_process_ids

    # Test that all process ids are unique:
    for i in range(NUM_PROCESSES):
        p_id = p_ids.pop()
        assert p_id not in p_ids

    
def test_simple_sudo_workflow(tmp_path, TiO2_r_bulk):

    atoms = TiO2_r_bulk
    study = _new_oer_study(atoms)
    EXPLOG = Explog()
    EXPLOG.init(tmp_path, 'test')

    cand_id = "dummy_id"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="Ir sites should be great", 
        notes="this is a great candidate", 
        study_obj=study
    )

    ids = EXPLOG.add_process(
        candidate_id=cand_id,
        calculation_type="bulk_relaxation",
    )
    assert len(ids) == 1, "Expected one process, as TiO2 is non-magnetic"

    # Submit the process:
    for pid in ids:
        EXPLOG.submit_process(process_id=pid)

    updates_dict = EXPLOG.update_log()

    l_keys = list(updates_dict.keys())
    assert len(l_keys) == 1
    assert l_keys[0] == 0
    assert updates_dict[0] == 'completed'

    # Args:
    cant_id = cand_id
    max_miller = 1      # Max miller index

    # Args to leve out (for now):
    method = 'all'      # What coordination to consider
    stoichiometry_tolerance = 0.2 # Allowed stoichiometry diviation from bulk
    all_species_present = True # Only surfaces with all bulk species present
    symmetrize = True # Whether to symmetrize the surfaces
    select_closest_O_stoichiometry = True # Select surfaces with closest O stoichiometry to bulk
    min_slab_thickness = 9 # Minimum slab thickness in Å
    max_slab_thickness = 20 # Maximum slab thickness in Å
    min_atoms = 20 # Minimum number of atoms in the slab
    max_atoms = 120 # Maximum number of atoms in the slab
    max_layers = 6 # Maximum number of layers, considered when building the slabs

    study = EXPLOG.relational_frame.candidates[cant_id].study_obj
    ranking = study.get_termination_rankings() # None on first call
    assert ranking is None, "Expected initial ranking to be None before prediction"

    study.predict_most_likely_surfaces(
    max_miller = max_miller,
    method = method,
    stoichiometry_tolerance = stoichiometry_tolerance,
    all_species_present = all_species_present,
    symmetrize = symmetrize,
    select_closest_O_stoichiometry = select_closest_O_stoichiometry,
    min_slab_thickness = min_slab_thickness,
    max_slab_thickness = max_slab_thickness,
    min_atoms = min_atoms,
    max_atoms = max_atoms,
    max_layers = max_layers
    )
    ranking = study.get_termination_rankings()
    assert isinstance(ranking, pd.DataFrame)

    termination_index = 0
    df = study.get_init_adsorption_sites_df(termination_index)
    assert isinstance(df, pd.DataFrame)

    study = EXPLOG.relational_frame.candidates.df['study_obj'][0]
    study.initialize_oer_surface_study(0)

    p_id = EXPLOG.add_process( 
    candidate_id = cant_id,
    calculation_type = "surface_relaxation",
    termination_index = termination_index
    )

    EXPLOG.submit_process(process_id = p_id)
    EXPLOG.update_log()
    surface_study = study.get_surface_studies()[0]

    # =========================================================================
    # Stage A — Site determination and initial state checks
    # =========================================================================
    # Use only_reduced_coord_O_sites=False to get both site types (on-top and
    # lattice O). TiO2 rutile at miller=1 produces both.
    surface_study.determine_adsorption_sites()

    df = surface_study.get_adsorption_sites_df()
    assert isinstance(df, pd.DataFrame)

    on_top_rows = df[df['site type'] == 'on-top']
    lattice_o_rows = df[df['site type'] == 'lattice O']
    assert len(on_top_rows) > 0, "Expected at least one on-top site for TiO2"
    assert len(lattice_o_rows) > 0, "Expected at least one lattice-O site for TiO2"

    on_top_idx = int(on_top_rows['Site index'].iloc[0])
    lattice_o_idx = int(lattice_o_rows['Site index'].iloc[0])

    surface_study.initialize_adsorption_site_study(on_top_idx)
    surface_study.initialize_adsorption_site_study(lattice_o_idx)

    ad_studies = surface_study.get_adsorption_site_studies_dict()
    on_top_study = ad_studies[on_top_idx]
    lattice_o_study = ad_studies[lattice_o_idx]

    n_surface_atoms = len(surface_study.get_relaxed_surface())

    # --- Structure shape: O-calc input atoms ---
    # on-top: surface + 1 O adsorbate (tag=4) placed on top
    o_calc_on_top = on_top_study.get_atoms_for_O_calc()
    assert len(o_calc_on_top) == n_surface_atoms + 1, (
        f"on-top O calc: expected surface+1={n_surface_atoms+1} atoms, "
        f"got {len(o_calc_on_top)}"
    )
    assert sum(1 for a in o_calc_on_top if a.symbol == 'O' and a.tag == 4) == 1, (
        "on-top O calc structure must contain exactly one O adsorbate with tag=4"
    )

    # lattice-O: surface MINUS the lattice O (tag=3 O removed for the reference calc)
    o_calc_lat = lattice_o_study.get_atoms_for_O_calc()
    assert len(o_calc_lat) == n_surface_atoms - 1, (
        f"lattice-O O calc: expected surface-1={n_surface_atoms-1} atoms "
        f"(lattice O removed), got {len(o_calc_lat)}"
    )
    assert sum(1 for a in o_calc_lat if a.symbol == 'O' and a.tag == 3) == 0, (
        "Lattice O (tag=3) must be absent from the lattice-O O calc structure"
    )

    # NOTE: oh_batch_finalized defaults to True by design.
    # Direct/manual workflows (e.g. this test) expose OH metrics immediately.
    # job_handler sets it to False when OH jobs are submitted and restores True
    # when all 3 OH processes reach terminal status.
    assert on_top_study.is_oh_batch_finalized() == True, (
        "oh_batch_finalized should default to True (direct-workflow convention)"
    )
    assert lattice_o_study.is_oh_batch_finalized() == True

    # --- No relaxed structures set yet ---
    assert on_top_study.relaxed_O_atoms is None, "No O_adsorption run yet"
    assert on_top_study.relaxed_OH_atoms is None, "No OH_adsorption run yet"
    assert lattice_o_study.relaxed_O_atoms is None, "No O_adsorption run yet"
    assert lattice_o_study.relaxed_OH_atoms is None, "No OH_adsorption run yet"

    # --- Initial energies_dict ---
    # on-top: constructor stores relaxed_surface_energy as clean_surface_energy
    on_top_e = on_top_study.get_adsorption_site_energies()
    assert on_top_e['clean_surface_energy'] is not None, (
        "on-top: clean_surface_energy set at construction from relaxed surface energy"
    )
    assert on_top_e['O_adsorbed_energy'] is None, "O_adsorption not run yet"
    assert on_top_e['OH_adsorbed_energy'] is None, "OH_adsorption not run yet"

    # lattice-O: constructor stores relaxed_surface_energy as O_adsorbed_energy
    # (the surface WITH the lattice O IS the 'O-adsorbed' state in this model).
    # clean_surface_energy stays None until O_adsorption removes the lattice O.
    lattice_o_e = lattice_o_study.get_adsorption_site_energies()
    assert lattice_o_e['O_adsorbed_energy'] is not None, (
        "lattice-O: O_adsorbed_energy set at construction (surface WITH lattice O)"
    )
    assert lattice_o_e['clean_surface_energy'] is None, (
        "lattice-O: clean_surface_energy is None until O_adsorption job completes "
        "(job removes lattice O and relaxes the remaining surface)"
    )
    assert lattice_o_e['OH_adsorbed_energy'] is None, "OH_adsorption not run yet"

    # --- All G values None before any DFT jobs ---
    surface_study.update_adsorption_energies()
    df = surface_study.get_adsorption_sites_df()
    on_top_row = df[df['Site index'] == on_top_idx].iloc[0]
    lattice_o_row = df[df['Site index'] == lattice_o_idx].iloc[0]
    assert pd.isna(on_top_row['G(O)']), "G(O) None — O_adsorption not run"
    assert pd.isna(on_top_row['G(OH)']), "G(OH) None — OH_adsorption not run"
    assert pd.isna(on_top_row['ideal overpotential'])
    assert pd.isna(on_top_row['overpotential from OH-OOH scaling relation'])
    assert pd.isna(lattice_o_row['G(O)'])
    assert pd.isna(lattice_o_row['G(OH)'])
    assert pd.isna(lattice_o_row['ideal overpotential'])
    assert pd.isna(lattice_o_row['overpotential from OH-OOH scaling relation'])

    # =========================================================================
    # Stage B — O_adsorption for on-top site
    # add_process("O_adsorption") returns int (single process)
    # =========================================================================
    p_id_o = EXPLOG.add_process(
        candidate_id=cant_id,
        calculation_type="O_adsorption",
        termination_index=termination_index,
        site_index=on_top_idx,
    )
    EXPLOG.submit_process(process_id=p_id_o)
    updates_b = EXPLOG.update_log()
    # In "test" mode every submitted job completes immediately:
    assert all(v == 'completed' for v in updates_b.values()), (
        f"Expected all processes to complete in test mode, got: {updates_b}"
    )

    # Relaxed O structure stored; O_adsorbed_energy populated:
    assert on_top_study.relaxed_O_atoms is not None, (
        "relaxed_O_atoms must be set after O_adsorption completes"
    )
    on_top_e = on_top_study.get_adsorption_site_energies()
    assert on_top_e['O_adsorbed_energy'] is not None, (
        "O_adsorbed_energy set by O_adsorption job (surface + O energy)"
    )
    assert on_top_e['OH_adsorbed_energy'] is None, "OH_adsorption not run yet"

    df = surface_study.get_adsorption_sites_df()
    on_top_row = df[df['Site index'] == on_top_idx].iloc[0]
    assert np.isfinite(float(on_top_row['G(O)'])), (
        "G(O) must be finite after O_adsorption completes"
    )
    assert np.isfinite(float(on_top_row['G(O) deviation']))
    assert pd.isna(on_top_row['G(OH)']), "G(OH) still None — OH_adsorption not run"
    assert pd.isna(on_top_row['ideal overpotential']), (
        "ideal overpotential requires both G(O) and G(OH)"
    )
    assert pd.isna(on_top_row['overpotential from OH-OOH scaling relation'])

    # OH initialization structures are available after O relaxation (on-top).
    # Each OH init = surface + O adsorbate (tag=4) + H (tag=4) → N+2 atoms.
    oh_init_on_top = on_top_study.get_OH_adsorbed_atoms()
    assert len(oh_init_on_top) == 3, "Expected 3 OH initialization structures"
    assert all(len(s) == n_surface_atoms + 2 for s in oh_init_on_top), (
        f"on-top OH inits should have N+2 atoms (surface+O+H), "
        f"got {[len(s) for s in oh_init_on_top]}"
    )
    assert all(
        sum(1 for a in s if a.symbol == 'H' and a.tag == 4) == 1
        for s in oh_init_on_top
    ), "Each on-top OH init must have exactly one H adsorbate (tag=4)"

    # =========================================================================
    # Stage C — OH_adsorption for on-top site
    # add_process("OH_adsorption") returns list[int] (3 IDs, one per OH init)
    # and immediately sets oh_batch_finalized = False
    # =========================================================================
    oh_ids = EXPLOG.add_process(
        candidate_id=cant_id,
        calculation_type="OH_adsorption",
        termination_index=termination_index,
        site_index=on_top_idx,
    )
    assert len(oh_ids) == 3, (
        "OH_adsorption creates 3 processes (one per OH initialization)"
    )
    assert on_top_study.is_oh_batch_finalized() == False, (
        "add_process('OH_adsorption') sets oh_batch_finalized=False to hide "
        "partial OH results while async jobs are running"
    )

    for pid in oh_ids:
        EXPLOG.submit_process(process_id=pid)
    updates_c = EXPLOG.update_log()
    assert all(v == 'completed' for v in updates_c.values()), (
        f"Expected all 3 on-top OH processes to complete in test mode, "
        f"got: {updates_c}"
    )

    # All 3 OH processes terminal → _refresh_oh_batch_finalization_state → True:
    assert on_top_study.is_oh_batch_finalized() == True, (
        "oh_batch_finalized restored to True once all 3 OH processes are terminal"
    )
    assert on_top_study.relaxed_OH_atoms is not None, (
        "Best accepted relaxed OH structure stored after OH batch finalizes"
    )
    on_top_e = on_top_study.get_adsorption_site_energies()
    assert on_top_e['OH_adsorbed_energy'] is not None, (
        "OH_adsorbed_energy set to lowest-energy accepted OH structure"
    )

    df = surface_study.get_adsorption_sites_df()
    on_top_row = df[df['Site index'] == on_top_idx].iloc[0]
    assert np.isfinite(float(on_top_row['G(OH)'])), (
        "G(OH) finite after OH batch finalizes"
    )
    assert np.isfinite(float(on_top_row['G(OH) deviation']))
    assert np.isfinite(float(on_top_row['ideal overpotential'])), (
        "ideal overpotential available once G(O) and G(OH) are both known"
    )
    assert np.isfinite(float(on_top_row['overpotential from OH-OOH scaling relation']))

    # Candidate-level metrics are updated when at least one site has both G(O) and G(OH):
    candidate_row = EXPLOG.relational_frame.candidates[cant_id]
    assert np.isfinite(float(candidate_row.idealOverPotential))
    assert np.isfinite(float(candidate_row.Overpotential_from_scaling))

    # Exactly one of the 3 OH process rows must carry OER metrics;
    # the other two must remain NaN (_update_oh_batch_process_oer_metrics
    # populates only the first completed row).
    pdf = EXPLOG.relational_frame.processes.df
    oh_rows = pdf[
        (pdf["candidate_id"] == cant_id)
        & (pdf["job_type"] == "OH_adsorption")
        & (pdf["termination_index"] == termination_index)
        & (pdf["site_index"] == on_top_idx)
    ]
    assert len(oh_rows) == 3
    populated   = oh_rows[oh_rows["G(OH)"].notna()]
    unpopulated = oh_rows[oh_rows["G(OH)"].isna()]
    assert len(populated) == 1, (
        f"Exactly 1 OH process row must carry G(OH), got {len(populated)}"
    )
    assert len(unpopulated) == 2, (
        "The other 2 OH process rows must have NaN G(OH)"
    )
    assert populated.iloc[0]["status"] == "completed", (
        "The populated OH row must be the completed one"
    )
    assert np.isfinite(float(populated.iloc[0]["ideal overpotential"])), (
        "Populated OH row must have finite ideal overpotential"
    )

    # =========================================================================
    # Stage D — OH_adsorption for lattice-O site (submitted BEFORE O_adsorption)
    # lattice-O OH does not require prior O_adsorption — OH is initialized
    # directly from the lattice surface (intact lattice O) + H placed nearby.
    # =========================================================================

    # OH initialization structures for lattice-O:
    # init_surface (N atoms, lattice O intact) + H (tag=4) → N+1 atoms
    oh_init_lat = lattice_o_study.get_OH_adsorbed_atoms()
    assert len(oh_init_lat) == 3
    assert all(len(s) == n_surface_atoms + 1 for s in oh_init_lat), (
        f"lattice-O OH inits should have N+1 atoms (surface+H), "
        f"got {[len(s) for s in oh_init_lat]}"
    )
    assert all(
        sum(1 for a in s if a.symbol == 'H' and a.tag == 4) == 1
        for s in oh_init_lat
    ), "Each lattice-O OH init must have exactly one H adsorbate (tag=4)"

    oh_ids_lattice = EXPLOG.add_process(
        candidate_id=cant_id,
        calculation_type="OH_adsorption",
        termination_index=termination_index,
        site_index=lattice_o_idx,
    )
    assert len(oh_ids_lattice) == 3
    assert lattice_o_study.is_oh_batch_finalized() == False, (
        "add_process('OH_adsorption') sets oh_batch_finalized=False"
    )

    for pid in oh_ids_lattice:
        EXPLOG.submit_process(process_id=pid)
    updates_d = EXPLOG.update_log()
    assert all(v == 'completed' for v in updates_d.values()), (
        f"Expected all 3 lattice-O OH processes to complete in test mode, "
        f"got: {updates_d}"
    )

    assert lattice_o_study.is_oh_batch_finalized() == True
    assert lattice_o_study.relaxed_OH_atoms is not None
    lattice_o_e = lattice_o_study.get_adsorption_site_energies()
    assert lattice_o_e['OH_adsorbed_energy'] is not None

    # G(OH) is STILL None even though OH batch is finalized:
    # clean_surface_energy is not set until O_adsorption removes the lattice O.
    assert lattice_o_e['clean_surface_energy'] is None, (
        "clean_surface_energy is None until O_adsorption job runs "
        "(that job removes lattice O and stores the relaxed surface energy)"
    )
    df = surface_study.get_adsorption_sites_df()
    lattice_o_row = df[df['Site index'] == lattice_o_idx].iloc[0]
    assert pd.isna(lattice_o_row['G(O)']), (
        "G(O) is None: E_pure (clean_surface_energy) not yet set"
    )
    assert pd.isna(lattice_o_row['G(OH)']), (
        "G(OH) is None: clean_surface_energy missing even though OH batch is "
        "finalized and OH_adsorbed_energy is set"
    )
    assert pd.isna(lattice_o_row['ideal overpotential'])
    assert pd.isna(lattice_o_row['overpotential from OH-OOH scaling relation'])

    # =========================================================================
    # Stage E — O_adsorption for lattice-O site
    # This removes the lattice O and relaxes the remaining surface.
    # set_relaxed_O_calc stores that energy as clean_surface_energy.
    # After this, G(O) and G(OH) both become finite (OH was done in Stage D).
    # add_process("O_adsorption") returns int (single process)
    # =========================================================================
    p_id_o_lat = EXPLOG.add_process(
        candidate_id=cant_id,
        calculation_type="O_adsorption",
        termination_index=termination_index,
        site_index=lattice_o_idx,
    )
    EXPLOG.submit_process(process_id=p_id_o_lat)
    updates_e = EXPLOG.update_log()
    assert all(v == 'completed' for v in updates_e.values()), (
        f"Expected O_adsorption to complete in test mode, got: {updates_e}"
    )

    assert lattice_o_study.relaxed_O_atoms is not None, (
        "relaxed_O_atoms (surface-without-lattice-O) stored after O_adsorption"
    )
    lattice_o_e = lattice_o_study.get_adsorption_site_energies()
    assert lattice_o_e['clean_surface_energy'] is not None, (
        "clean_surface_energy now set (energy of surface with lattice O removed)"
    )
    assert lattice_o_e['O_adsorbed_energy'] is not None, (
        "O_adsorbed_energy was set at construction (surface WITH lattice O)"
    )

    # Both G(O) and G(OH) now finite:
    # - G(O): O_adsorbed_energy (Stage A) + clean_surface_energy (Stage E)
    # - G(OH): OH_adsorbed_energy (Stage D) + clean_surface_energy (Stage E)
    df = surface_study.get_adsorption_sites_df()
    lattice_o_row = df[df['Site index'] == lattice_o_idx].iloc[0]
    assert np.isfinite(float(lattice_o_row['G(O)'])), (
        "G(O) finite: both O_adsorbed_energy and clean_surface_energy now set"
    )
    assert np.isfinite(float(lattice_o_row['G(O) deviation']))
    assert np.isfinite(float(lattice_o_row['G(OH)'])), (
        "G(OH) finite: OH_adsorbed_energy (Stage D) + clean_surface_energy (Stage E)"
    )
    assert np.isfinite(float(lattice_o_row['G(OH) deviation']))
    assert np.isfinite(float(lattice_o_row['ideal overpotential']))
    assert np.isfinite(float(lattice_o_row['overpotential from OH-OOH scaling relation']))

