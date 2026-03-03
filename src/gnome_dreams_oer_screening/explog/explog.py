import os
import pandas as pd
from typing import Literal
from pathlib import Path
from pysqa import QueueAdapter
from sympy import rf
import pickle

from .relational_frame import RelationalFrame
from gnome_dreams_oer_screening.oer.oer_study import OER_catalyst_study
from gnome_dreams_oer_screening.vasp.job_handler import Job_handler


class Explog:
    # we'll manually init later
    def init(self, 
        vasp_calculation_dir: Path | str = Path("vasp_calculations"),
        mode: Literal["test", "slurm_test", "production_test", 
                      "production"] = "production",
        prefix_vasp_dirs: bool = True,
        sleep_min = 2,
        ):
        # Setup of relational frame, with tables for candidates and 
        # processes (both of which are needed for the Job_handler etc.)
        rf = RelationalFrame(root="candidates")

        # study_df = pd.DataFrame(columns=["candidate_id",
        #     "reason_or_hypothesis", "notes", "study_obj", "OHDone", 
        #     "idealOverPotential"])
        study_df = pd.DataFrame({
            "candidate_id": pd.Series(dtype="string"),
            "reason_or_hypothesis": pd.Series(dtype="string"),
            "notes": pd.Series(dtype="string"),
            "study_obj": pd.Series(dtype="object"), # can store any Python object
            "OHDone": pd.Series(dtype="boolean"),   # True if OH adsorption is done
            "idealOverPotential": pd.Series(dtype="float64") # nullable float (supports <NA>)
        })
        rf.add_table("candidates", study_df, pk="candidate_id")

        # process_df = pd.DataFrame(columns=["process_id", "candidate_id", 
        #     "job_type", "slurmID", "status", 'termination_index', 
        #     'site_index', "VASP_dir", "processNote"]).astype({
        #         "process_id": "string",
        #         "candidate_id": "string",
        #         "job_type": "string",
        #         "slurmID": "string",
        #         "status": "string",
        #         'termination_index': 'Int64', # nullable integer (supports <NA>)
        #         'site_index': 'Int64',       # nullable integer (supports <NA>)
        #         "VASP_dir": "string",
        #         "processNote": "string"
        #     })
        process_df = pd.DataFrame({
            "process_id": pd.Series(dtype="string"),
            "candidate_id": pd.Series(dtype="string"),
            "job_type": pd.Series(dtype="string"),
            "slurmID": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            'termination_index': pd.Series(dtype='Int64'), # nullable integer (supports <NA>)
            'site_index': pd.Series(dtype='Int64'),       # nullable integer (supports <NA>)
            "VASP_dir": pd.Series(dtype="string"),
            "processNote": pd.Series(dtype="string")
        })
        rf.add_table("processes", process_df, pk="process_id")

        # Add relation between candidates and processes:
        rf.candidates.add_relation(name="processes", child_table="processes",
            fk_in_child="candidate_id")
        
        self.relational_frame = rf
        self.vasp_calculation_dir = vasp_calculation_dir
        
        # Ensure vasp_calculation_dir is a Path object
        if isinstance(vasp_calculation_dir, str):
            vasp_calculation_dir = Path(vasp_calculation_dir)
        self.job_handler = Job_handler(rf, vasp_calculation_dir,
            mode=mode, prefix_vasp_dirs_with_prefix=prefix_vasp_dirs,
            sleep_min=sleep_min)
        
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
                                                 allow_update=False)

    def add_process(self, 
        candidate_id: str, 
        calculation_type: Literal["bulk_relaxation", 
            "surface_relaxation", "O_adsorption", 'OH_adsorption'],
        termination_index: int | None = None,
        site_index: int | None = None,
        note: str | None = None
        ) -> None:
        
        result =  self.job_handler.enter_process(candidate_id, 
            calculation_type, termination_index, site_index, note)
        # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
        with open(self.vasp_calculation_dir / "explog.pkl", "wb") as f:
            pickle.dump({"candidates_df":self.relational_frame.candidates.df, "processes_df":self.relational_frame.processes.df}, f)
        return result
        
    def submit_process(self, process_id: str,
        partition: Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto']
        = 'auto') -> None:
        self.job_handler.submit_vasp_calculation(process_id, 
                                                 partition=partition)
        # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
        with open(self.vasp_calculation_dir / "explog.pkl", "wb") as f:
            pickle.dump({"candidates_df":self.relational_frame.candidates.df, "processes_df":self.relational_frame.processes.df}, f)
        
    def update_log(self):
        result = self.job_handler.update_log()
        # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
        with open(self.vasp_calculation_dir / "explog.pkl", "wb") as f:
            pickle.dump({"candidates_df":self.relational_frame.candidates.df, "processes_df":self.relational_frame.processes.df}, f)
        return result
    
    
        #  SOULD RETURN A DICT, WITH ALL UPDATES... <<<-----------------------------------------

EXPLOG = Explog()
    
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
