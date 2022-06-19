#!/bin/bash
#SBATCH --job-name=mp2cores
#SBATCH --output=mp_2cores_200k.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=10
#SBATCH --time=4:00:00

srun taskset --cpu-list 0-1 python ./configuration_scripts/benchmark.py 2
