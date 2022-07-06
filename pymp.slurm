#!/bin/bash
#SBATCH --job-name=pymp4c8k
#SBATCH --output=pymp_4cores_800k_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-3 python ./configuration_scripts/benchmark.py 4 800000
