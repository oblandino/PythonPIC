#!/bin/bash
#SBATCH --job-name=pymp64_320
#SBATCH --output=pymp_64cores_320k_nu_weak_rev0.txt
#SBATCH --partition=nu --nodelist=nu-3c.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py 64 320000
