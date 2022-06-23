#!/bin/bash
#SBATCH --job-name=mp16cores
#SBATCH --output=mp_16cores_200k_opt_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=10
#SBATCH --time=3:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-15 python ./configuration_scripts/benchmark.py 16
