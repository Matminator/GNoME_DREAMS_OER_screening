#!/bin/bash
#SBATCH --job-name DFT_AGENT
#SBATCH -e ./slabord.err
#SBATCH -o ./slabord.out
#SBATCH --mail-type None
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p xeon24el8_test
#SBATCH -t 00:02:00

module purge
module use /home/modules/energy/modules/all
module load VASP/6.5.1-intel-2023a

module load impi/2021.9.0-intel-compilers-2023.1.0

ml Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

{{command}}