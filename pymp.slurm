#!/bin/bash
#SBATCH --job-name=pymp2cores
#SBATCH --output=pymp_2cores_200k_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=10
#SBATCH --time=2:00:00
#SBATCH --exclusive

srun taskset --cpu-list 0-1 python ./configuration_scripts/benchmark.py 2
