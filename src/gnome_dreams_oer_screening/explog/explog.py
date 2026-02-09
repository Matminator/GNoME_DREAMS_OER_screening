import os
from typing import Literal
from pathlib import Path
from pysqa import QueueAdapter

from .relational_frame import RelationalFrame
from gnome_dreams_oer_screening.vasp.job_submitter import Job_submitter


class Explog:
    def __init__(self, queue_adapter: QueueAdapter):
        with open(os.path.join(WORKING_DIRECTORY, "slurm.sh"), "w") as file:
            file.write(r"""#!/bin/bash
#SBATCH -J {{job_name}} # Job name
#SBATCH -n {{cores_max}} # Number of total cores
#SBATCH -N {{nodes_max}} # Number of nodes
#SBATCH --time={{run_time_max | int}}
#SBATCH -p {{partition}}
#SBATCH --mem-per-cpu={{memory_max}}M # Memory pool for all cores in MB
#SBATCH -e {{errNoutName}}.err #change the name of the err file 
#SBATCH -o {{errNoutName}}.out # File to which STDOUT will be written %j is the job #

{{command}}

                    """)
            
        with open(os.path.join(WORKING_DIRECTORY, "queue.yaml"), "w") as file:
            file.write(r"""queue_type: SLURM
queue_primary: slurm
queues:
slurm: {
    job_name: testPysqa,
    cores_max: 4, 
    cores_min: 1, 
    nodes_max: 1,
    memory_max: 2000,
    partition: venkvis-cpu,
    script: slurm.sh
    }
                    """)
        
        self.qa = QueueAdapter()
        self.queue_adapter = queue_adapter
        self.relational_frame = RelationalFrame()
        self.submitter = Job_submitter(self.queue_adapter, self.relational_frame)

    def process_message(self, message):
        # Process the message and update the relational frame
        self.relational_frame.update(message)

    def run(self):
        while True:
            message = self.queue_adapter.get_message()
            if message is not None:
                self.process_message(message)
                
    def submit_job(self, job):
        self.submitter.submit(job)
        
    def get_info(self, query):
        return self.relational_frame.query(query)