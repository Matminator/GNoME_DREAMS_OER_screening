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
module load VASP/6.5.1-intel-2025b
module load impi/2021.16.1-intel-compilers-2025.2.0

# No Python module: venv_el9 ships its own CPython 3.11.13. Loading one would
# shadow it.
source venv_el9/bin/activate || exit 1

command -v vasp_std >/dev/null && command -v mpiexec >/dev/null || {
    echo "FATAL: VASP/impi not on PATH on $(hostname)" >&2; exit 1; }

{{command}}