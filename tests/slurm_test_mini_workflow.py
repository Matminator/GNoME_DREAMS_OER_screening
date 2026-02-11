# --- standard library imports ---
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

# Get the directory where the current Python file is located
current_dir = Path(__file__).resolve().parent
new_dir = current_dir / 'TEMP_slurm_test_mini_workflow'
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

processes_df = EXPLOG.relational_frame.processes.df
print(processes_df)
print('-------------------')
print(processes_df['slurmID'])

