#!/bin/bash
#SBATCH --job-name=pymp64_600
#SBATCH --output=pymp_64cores_600k_nu_rev0.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py 64 600000
