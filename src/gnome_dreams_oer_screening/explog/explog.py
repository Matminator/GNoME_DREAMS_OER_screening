import os
import pandas as pd
from typing import Literal
from pathlib import Path
from pysqa import QueueAdapter
from sympy import rf

from .relational_frame import RelationalFrame
from gnome_dreams_oer_screening.vasp.Job_handler import Job_handler


class Explog:
    def __init__(self, 
        vasp_calculation_dir: Path | str = Path("vasp_calculations"),
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
        ):
        # Setup of relational frame, with tables for candidates and 
        # processes (both of which are needed for the Job_handler etc.)
        rf = RelationalFrame(root="candidates")

        study_df = pd.DataFrame(columns=["candidate_id",
            "reason_or_hypothesis", "notes", "study_obj", "OHDone", 
            "idealOverPotential"])
        rf.add_table("candidates", study_df, pk="candidate_id")

        process_df = pd.DataFrame(columns=["process_id", "candidate_id", 
            "job_type", "slurmID", "status", "VASP_dir", "processNote"])
        rf.add_table("processes", process_df, pk="process_id")

        # Add relation between candidates and processes:
        rf.candidates.add_relation(name="processes", child_table="processes",
            fk_in_child="candidate_id")
        
        self.relational_frame = rf
        
        # Ensure vasp_calculation_dir is a Path object
        if isinstance(vasp_calculation_dir, str):
            vasp_calculation_dir = Path(vasp_calculation_dir)
        self.job_handler = Job_handler(rf, vasp_calculation_dir)
        




# --- Copilot suggested mehtods: ---------------------------------------

    # def process_message(self, message):
    #     # Process the message and update the relational frame
    #     self.relational_frame.update(message)

    # def run(self):
    #     while True:
    #         message = self.queue_adapter.get_message()
    #         if message is not None:
    #             self.process_message(message)
                
    # def submit_job(self, job):
    #     self.submitter.submit(job)
        
    # def get_info(self, query):
    #     return self.relational_frame.query(query)
