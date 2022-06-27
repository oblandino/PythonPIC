#!/bin/bash
#SBATCH --job-name=pymp16cores
#SBATCH --output=pymp_16cores_400k_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-6d.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-15 python ./configuration_scripts/benchmark.py 16
