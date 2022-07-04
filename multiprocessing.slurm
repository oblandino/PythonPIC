#!/bin/bash
#SBATCH --job-name=mp64cores
#SBATCH --output=mp_64cores_100k_cs1000_nu_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py 64 100000
