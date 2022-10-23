#!/bin/bash
#SBATCH --job-name=mp16_320k
#SBATCH --output=mp_16cores_320k_dribe_weak_rev0.txt
#SBATCH --partition=dribe --nodelist=dribe-05.cnca
#SBATCH --ntasks=10
#SBATCH --time=48:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-15 python ./configuration_scripts/benchmark.py 16 320000
