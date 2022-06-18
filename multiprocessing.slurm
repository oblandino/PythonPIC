#!/bin/bash
#SBATCH --job-name=mp64cores
#SBATCH --output=mp_reduced_64cores_200k.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

srun taskset --cpu-list 0-63 python ./configuration_scripts/benchmark.py
