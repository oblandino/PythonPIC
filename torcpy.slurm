#!/bin/bash
#SBATCH --job-name=torcpy64nu
#SBATCH --output=torcpy_64cores_nu_100k_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5b.cnca
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 64 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 100000
