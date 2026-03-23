#!/bin/bash
#SBATCH --job-name A_DFT
#SBATCH --mail-type None
#SBATCH -e {{slurm_out_name}}.err
#SBATCH -o {{slurm_out_name}}.out
#SBATCH -n {{cores}}        # Total cores
#SBATCH -N {{nodes}}        # Number of nodes
#SBATCH -p {{partition}}
#SBATCH -t {{walltime}}

module purge
module use /home/modules/energy/modules/all
module load VASP/6.5.1-intel-2023a

module load impi/2021.9.0-intel-compilers-2023.1.0

ml Python/3.11.3-GCCcore-12.3.0
source venv2/bin/activate

{{command}}