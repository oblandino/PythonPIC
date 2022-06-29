#!/bin/bash
#SBATCH --job-name=numba2cores
#SBATCH --output=numba_2cores_400k_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-05.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-1 python ./configuration_scripts/benchmark.py 400000
