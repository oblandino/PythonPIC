# coding=utf-8
import numpy as np

from pythonpic import plotting_parser
from pythonpic.configs.run_beamplasma import initial

args_animation = False, False, True, False, False
args = plotting_parser("Weak beam instability")
np.random.seed(0)
s = initial("beamplasma1").lazy_run().plots_1d(*args_animation, alpha=0.5)