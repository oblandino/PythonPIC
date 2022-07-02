#!/bin/bash
#SBATCH --job-name=torcpy8dribe
#SBATCH --output=torcpy_8cores_dribe_100k_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-05.cnca
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 8 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 100000
