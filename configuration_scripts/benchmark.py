# coding=utf-8
from pythonpic.configs.benchmark_run import initial
import numpy as np
import itertools
import pandas as pd

number_grid = 1000
n_particles = [100]

list_particles = []
list_cells = []
list_runtimes = []
for number_particles in n_particles:
    # s = initial(f"{number_particles}_{number_grid}", number_particles,
    #             number_grid).lazy_run()
    runtime = initial(f"{number_particles}_{number_grid}", number_particles,
                number_grid).run_lite()
    print(number_particles, runtime, sep=",")
    list_particles.append(number_particles)
    list_cells.append(number_grid)
    list_runtimes.append(runtime)

dict = {'n_particles':list_particles, 'n_grid': list_cells, 'runtime':list_runtimes}
pd.DataFrame.from_dict(dict).to_csv('/work/oblandino/data/inz_data_jit.csv')

