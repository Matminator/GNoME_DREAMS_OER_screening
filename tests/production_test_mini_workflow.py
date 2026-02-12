# --- standard library imports ---
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

from ase import Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor

# --- IMPORTS FROM THIS PROJECT ---
from gnome_dreams_oer_screening import OER_catalyst_study
from gnome_dreams_oer_screening import Explog

def waite_for_processe_to_finish(process_id):
    iter = 0
    while True:
        EXPLOG.update_log()
        pdf = EXPLOG.relational_frame.processes.df
        status = pdf[pdf['process_id'] == process_id]['status'].values[0]
        print(
            '---------------------------------------------------\n',
            'Update iteration:', iter, '\n\n',
            pdf[pdf['process_id'] == process_id][['slurmID', 'status',
                                                    'job_type']],
            '\n___________________________________________________\n'
            , flush=True)
        
        os.system('squeue --me')
        

        print(
        '\n---------------------------------------------------\n',
        flush=True
            )
        if status in ['completed']:
            break
        time.sleep(30*60)
        iter += 1
    print('*'* 80)

# Get the directory where the current Python file is located
current_dir = Path(__file__).resolve().parent
new_dir = current_dir / 'TEMP_production_test_mini_workflow'
new_dir.mkdir(exist_ok=True)

EXPLOG = Explog()
EXPLOG.init(new_dir, 'production_test')

test_cands = Trajectory('notebooks/test_cands.traj')
atoms = test_cands[18]
S = AseAtomsAdaptor.get_structure(atoms)

study = OER_catalyst_study(atoms, H2O_gas_free_energy = -14.217, 
                           H2_gas_free_energy = -6.77)

cant_id = "07286qnfd"

EXPLOG.add_candidate(
    candidate_id=cant_id,
    reason_or_hypothesis="Ir sites should be great", 
    notes="this is a great candidate", 
    study_obj=study
)

p_id = EXPLOG.add_process(candidate_id=cant_id, 
                          calculation_type="bulk_relaxation")
EXPLOG.submit_process(process_id=p_id)
waite_for_processe_to_finish(p_id)

# --- Tools: -----------------------------------------------------------
study = EXPLOG.relational_frame.candidates[cant_id].study_obj
study.predict_most_likely_surfaces()

# ranking: only here for future use, not needed in this test:
ranking = study.get_termination_rankings()
ranking = ranking.sort_values('Normalized score', ascending=False)

study.initialize_oer_surface_study(0)
# ----------------------------------------------------------------------

p_id = EXPLOG.add_process( 
    candidate_id = cant_id,
    calculation_type = "surface_relaxation",
    termination_index = 0
)
EXPLOG.submit_process(process_id = p_id)
waite_for_processe_to_finish(p_id)

# --- Tools: -----------------------------------------------------------
study = EXPLOG.relational_frame.candidates.df['study_obj'][0]
surface_study = study.get_surface_studies()[0]
surface_study.determine_adsorption_sites()
surface_study.initialize_adsorption_site_study(0)
# ----------------------------------------------------------------------

p_id = EXPLOG.add_process( 
    candidate_id = cant_id,
    calculation_type = "O_adsorption",
    termination_index = 0,
    site_index = 0
)
EXPLOG.submit_process(process_id = p_id)
waite_for_processe_to_finish(p_id)

print('\n\n --->>> Results:')
# --- Tools: -----------------------------------------------------------
surface_study.update_adsorption_energies()
print(surface_study.get_adsorption_sites_df()[['site type', 
                                    'add site element', 'G_approx(O)']])

print(EXPLOG.relational_frame.candidates.df)
# ----------------------------------------------------------------------
print('\n\n --- END --- \n\n')