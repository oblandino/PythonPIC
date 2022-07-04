#!/bin/bash
#SBATCH --job-name=mp2cores
#SBATCH --output=mp_2cores_100k_cs5000_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-6a.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-1 python ./configuration_scripts/benchmark.py 2 100000
