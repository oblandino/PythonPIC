#!/bin/bash
#SBATCH --job-name=torcpy16dribe
#SBATCH --output=torcpy_16cores_dribe_100k_cs5000_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 16 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 100000
