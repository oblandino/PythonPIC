# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.helper_functions.physics import did_it_thermalize
from pythonpic.configs.run_twostream import initial
from pythonpic.visualization import animation, static_plots


args = plotting_parser("Two stream instability")
args_animation = False, False, True, False, False

#S = initial("TS_STABLE",
#            v0 = 0.01,
#            N_electrons=5000,
#            plasma_frequency=0.001,
#            T = 6000,
#            ).lazy_run().plots_1d(*args_animation)

S = initial("TS_UNSTABLE",
            v0 = 0.01,
            N_electrons=5000,
            plasma_frequency=0.1,
            T = 6000,
            ).lazy_run().plots_1d(*args_animation)



