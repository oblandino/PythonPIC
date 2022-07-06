#!/bin/bash
#SBATCH --job-name=mp16c32k
#SBATCH --output=mp_16cores_3200k_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-15 python ./configuration_scripts/benchmark.py 16 3200000
