#!/bin/bash
#SBATCH --job-name=torcpy32nu
#SBATCH --output=torcpy_32cores_nu_100k_cs1_rev1.txt
#SBATCH --partition=nu --nodelist=nu-5a.cnca
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --exclusive

mpiexec -n 32 -env TORCPY_WORKERS=1 --bind-to core --map-by core python configuration_scripts/benchmark.py 100000
