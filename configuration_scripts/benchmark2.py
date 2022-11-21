# coding=utf-8
from pythonpic.configs.benchmark_run import initial
from pythonpic import plotting_parser
import numpy as np
import itertools

n_grid = [1000]
n_particles = [100]
args_animation = False, False, True, False, False
args = plotting_parser("Hydrogen shield")
times_array = np.zeros((len(n_grid), len(n_particles)), dtype=float)

for j, number_particles in enumerate(n_particles):
    for i, number_grid in enumerate(n_grid):
        s = initial(f"{number_particles}_{number_grid}", number_particles,
                    number_grid).lazy_run().plots_3d(*args_animation)
        # s.plots_3d(True, False, True, False)
