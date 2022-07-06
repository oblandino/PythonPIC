#!/bin/bash
#SBATCH --job-name=t8c16K
#SBATCH --output=torcpy_8cores_dribe_1600k_rev1.txt
#SBATCH --partition=dribe --nodelist=dribe-06.cnca
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 8 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 1600000
