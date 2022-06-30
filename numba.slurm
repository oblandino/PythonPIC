#!/bin/bash
#SBATCH --job-name=numba
#SBATCH --output=numba_400k_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-05.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun python ./configuration_scripts/benchmark.py 400000
