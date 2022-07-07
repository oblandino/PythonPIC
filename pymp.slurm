#!/bin/bash
#SBATCH --job-name=pymp64c128k
#SBATCH --output=pymp_64cores_12800k_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-6a.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py 64 12800000
