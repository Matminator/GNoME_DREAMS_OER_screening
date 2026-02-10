import os
import pandas as pd
from typing import Literal
from pathlib import Path
from pysqa import QueueAdapter
from sympy import rf

from .relational_frame import RelationalFrame
from gnome_dreams_oer_screening.oer.oer_study import OER_catalyst_study
from gnome_dreams_oer_screening.vasp.job_handler import Job_handler


class Explog:
    def __init__(self, 
        vasp_calculation_dir: Path | str = Path("vasp_calculations"),
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
        prefix_vasp_dirs: bool = True
        ):
        # Setup of relational frame, with tables for candidates and 
        # processes (both of which are needed for the Job_handler etc.)
        rf = RelationalFrame(root="candidates")

        study_df = pd.DataFrame(columns=["candidate_id",
            "reason_or_hypothesis", "notes", "study_obj", "OHDone", 
            "idealOverPotential"])
        rf.add_table("candidates", study_df, pk="candidate_id")

        process_df = pd.DataFrame(columns=["process_id", "candidate_id", 
            "job_type", "slurmID", "status", 'termination_index', 
            'site_index', "VASP_dir", "processNote"])
        rf.add_table("processes", process_df, pk="process_id")

        # Add relation between candidates and processes:
        rf.candidates.add_relation(name="processes", child_table="processes",
            fk_in_child="candidate_id")
        
        self.relational_frame = rf
        
        # Ensure vasp_calculation_dir is a Path object
        if isinstance(vasp_calculation_dir, str):
            vasp_calculation_dir = Path(vasp_calculation_dir)
        self.job_handler = Job_handler(rf, vasp_calculation_dir,
            mode=mode, prefix_vasp_dirs_with_prefix=prefix_vasp_dirs)
        
    def add_candidate(self, 
                      candidate_id: str, 
                      study_obj: OER_catalyst_study,
                      reason_or_hypothesis: str | None = None, 
                      notes: str | None = None) -> None:
        
        if candidate_id in \
            self.relational_frame.candidates.df['candidate_id'].values:
            raise ValueError(f"Candidate with id {candidate_id} " \
                              "already exists in the log.")

        new_candidate_row = {"candidate_id": candidate_id,
                             "reason_or_hypothesis": reason_or_hypothesis,
                             "notes": notes,
                             "study_obj": study_obj}
        self.relational_frame.candidates.add_row(new_candidate_row, 
                                                 allow_update=True)

    def add_process(self, 
        candidate_id: str, 
        calculation_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        termination_index: int | None = None,
        site_index: int | None = None,
        note: str | None = None
        ) -> None:
        self.job_handler.enter_process(candidate_id, 
            calculation_type, termination_index, site_index, note)
        
    def submit_process(self, candidate_id: str, process_id: str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto']
        = 'auto') -> None:
        self.job_handler.submit_vasp_calculation(candidate_id,
            process_id, partition=partition)
        
    def update_log(self):
        self.job_handler.update_log()

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
