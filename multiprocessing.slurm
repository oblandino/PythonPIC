#!/bin/bash
#SBATCH --job-name=mp8cores
#SBATCH --output=mp_8cores_400k_opt_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=10
#SBATCH --time=3:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-7 python ./configuration_scripts/benchmark.py 7
