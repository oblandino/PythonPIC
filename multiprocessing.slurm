#!/bin/bash
#SBATCH --job-name=mp32cores
#SBATCH --output=mp_32cores_400k_opt_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-31 python ./configuration_scripts/benchmark.py 32 400000
