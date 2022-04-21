# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs import run_coldplasma, run_laser
from pythonpic.visualization import static_plots, time_snapshots
from pythonpic.configs.run_laser import initial, impulse_duration, n_macroparticles, number_cells
import pathlib

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

plot_folder = pathlib.Path("/home/dominik/Inzynierka/ThesisText/Images/")
S = run_coldplasma.initial(f"energy_plot", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
            N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode, save_data=False, T = T, scaling=scaling, c=c).lazy_run()
static_plots.publication_plots(S, str(plot_folder/"ESE_energy_plot.pdf"), [static_plots.electrostatic_energy_time_plots])
del S

S = run_coldplasma.initial(f"energy_plot_2", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
            N_electrons=N_electrons, epsilon_zero=epsilon_zero, push_mode=push_mode, save_data=False, T = T*10, scaling=scaling, c=c).lazy_run()
static_plots.publication_plots(S, str(plot_folder/"ESE_energy_plot_long.pdf"), [static_plots.electrostatic_energy_time_plots])
del S

S = run_laser.initial(f"75000_1378_run_21_Circular", n_macroparticles, number_cells, impulse_duration, 1e21, "Circular").lazy_run()
static_plots.publication_plots(S, str(plot_folder/"preplazma.pdf"), [time_snapshots.SpatialDistributionPlot]) # TODO fix
