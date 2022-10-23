#!/bin/bash
#SBATCH --job-name=t64_320
#SBATCH --output=torcpy_64cores_320k_nu_weak_rev0.txt
#SBATCH --partition=nu --nodelist=nu-6b.cnca
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 64 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 320000
