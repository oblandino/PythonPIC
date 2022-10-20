#!/bin/bash
#SBATCH --job-name=pymp8_1000
#SBATCH --output=pymp_8cores_1000k_dribe_rev0.txt
#SBATCH --partition=dribe --nodelist=dribe-05.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-7 python ./configuration_scripts/benchmark.py 8 1000000
