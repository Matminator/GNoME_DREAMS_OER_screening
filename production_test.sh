#!/bin/bash
#SBATCH --job-name P_test
#SBATCH --mail-type None
#SBATCH -e production_test.err
#SBATCH -o production_test.out
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p xeon24el8_week
#SBATCH -t 7-00:00:00

module purge
module use /home/modules/energy/modules/all
module load VASP/6.5.1-intel-2023a

module load impi/2021.9.0-intel-compilers-2023.1.0

ml Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

python3 tests/production_test_mini_workflow.py