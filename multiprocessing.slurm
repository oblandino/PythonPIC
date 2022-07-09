#!/bin/bash
#SBATCH --job-name=mpi64c128k
#SBATCH --output=mp_64cores_12800k_nu_rev2.txt
#SBATCH --partition=nu --nodelist=nu-5a.cnca
#SBATCH --ntasks=10
#SBATCH --time=48:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py 64 12800000
