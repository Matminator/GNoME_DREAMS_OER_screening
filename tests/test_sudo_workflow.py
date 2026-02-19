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

def test_multibul_bulk_jobs_submitted(tmp_path, TiO2_r_bulk):
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

    


