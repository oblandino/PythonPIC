#!/bin/bash
#SBATCH --job-name=numba
#SBATCH --output=numba_100k_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-6c.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun python ./configuration_scripts/benchmark.py 100000
