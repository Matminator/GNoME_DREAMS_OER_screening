# --- standard library imports ---
import numpy as np
import pandas as pd
from pathlib import Path

from ase import Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
import pytest

# --- IMPORTS FROM THIS PROJECT ---
from gnome_dreams_oer_screening import OER_catalyst_study
from gnome_dreams_oer_screening import Explog

from gnome_dreams_oer_screening.vasp.job_handler import Job_handler

    
@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True)
    if atoms is None:
        pytest.fail("Slab generation failed")
    return atoms, oer_study

def test_multiple_bulk_jobs_submitted(tmp_path, TiO2_r_bulk):
    NUM_PROCESSES = 10

    atoms, study = TiO2_r_bulk
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
            study_obj=study
        )

    p_ids = []
    for cand_id in cand_ids:
        p_id = EXPLOG.add_process(
        candidate_id=cand_id, 
        calculation_type="bulk_relaxation"
        )
        EXPLOG.submit_process(process_id=p_id)
        p_ids.append(p_id)

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

    only_reduced_coord_O_sites = True # Use True as default for now


    atoms, study = TiO2_r_bulk
    EXPLOG = Explog()
    EXPLOG.init(tmp_path, 'test')

    cand_id = "dummy_id"
    EXPLOG.add_candidate(
        candidate_id=cand_id,
        reason_or_hypothesis="Ir sites should be great", 
        notes="this is a great candidate", 
        study_obj=study
    )

    p_id = EXPLOG.add_process(
        candidate_id=cand_id, 
        calculation_type="bulk_relaxation"
    )
    EXPLOG.submit_process(process_id=p_id)

    updates_dict = EXPLOG.update_log()

    l_keys = list(updates_dict.keys())
    assert len(l_keys) == 1
    assert l_keys[0] == str(0)
    assert updates_dict['0'] == 'completed'

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
    if ranking is None:
        # No arguments needs to be provided (all have defaults as shown above):
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

    df = ranking.sort_values('Normalized score', ascending=False)
    assert isinstance(df, pd.DataFrame)

    cand_id = cand_id # will be needed to get the study object...
    termination_index = 0 # What termination to yield the adsorption sites for

    # Arg which may be left out (for now):

    df = study.get_init_adsorption_sites_df(termination_index, 
                    only_reduced_coord_O_sites=only_reduced_coord_O_sites)
    
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
    EXPLOG.relational_frame.processes.df
    # EXPLOG.relational_frame.processes.df


    surface_study = study.get_surface_studies()[0]
    surface_study.determine_adsorption_sites(
        only_reduced_coord_O_sites = only_reduced_coord_O_sites)
    
    df = surface_study.get_adsorption_sites_df()
    assert isinstance(df, pd.DataFrame)

    site_indices = df['Site index'].to_list()
    assert len(site_indices) > 0
    
    surface_study.initialize_adsorption_site_study(site_indices[0])
        

    p_id = EXPLOG.add_process( 
        candidate_id = cant_id,
        calculation_type = "O_adsorption",
        termination_index = termination_index,
        site_index = 0
    )

    # EXPLOG.relational_frame.processes.df

    EXPLOG.update_log()
