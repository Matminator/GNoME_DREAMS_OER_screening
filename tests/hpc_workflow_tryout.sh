#!/bin/bash
#SBATCH --job-name OER_wf_test
#SBATCH --mail-type None
#SBATCH -e hpc_workflow_test_%j.err
#SBATCH -o hpc_workflow_test_%j.out
#SBATCH -n 1        # Total cores
#SBATCH -N 1        # Number of nodes
#SBATCH -p xeon24el8
#SBATCH -t 05:10:00

module purge
module use /home/modules/energy/modules/all
module load VASP/6.5.1-intel-2023a

module load impi/2021.9.0-intel-compilers-2023.1.0

ml Python/3.11.3-GCCcore-12.3.0

source /home/energy/matnis/projects/dreams_colab/material_agent/venv2/bin/activate

# Ensure enum.x is in PATH
which enum.x
which makestr.x

python3 tests/hpc_workflow_tryout.py MLIP_test
