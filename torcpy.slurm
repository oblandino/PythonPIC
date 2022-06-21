#!/bin/bash
#SBATCH --job-name=torcpy2cores
#SBATCH --output=torcpy_2cores_10k.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=3
#SBATCH --time=4:00:00

srun mpiexec -n 2 -env TORCPY_WORKERS=2 --bind-to core --map-by core -hosts=dribe-06.cnca python configuration_scripts/benchmark.py
