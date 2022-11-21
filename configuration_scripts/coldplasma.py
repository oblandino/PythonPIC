# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_coldplasma import initial
from pythonpic.visualization import animation

args_animation = False, False, True, False, False

args = plotting_parser("Cold plasma oscillations")
plasma_frequency = 1
push_mode = 2
N_electrons = 1024
NG = 64
qmratio = -1
T = 10
scaling = 1
c = 10
epsilon_zero = 1

S = initial(f"CO_LINEAR", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
            N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode,
            push_amplitude=1,
            save_data=False, T = T, scaling=scaling, c=c).lazy_run().plots_1d(*args_animation)



