"""Implements interaction of the laser with a hydrogen shield plasma"""
# coding=utf-8
import numpy as np
from pythonpic.algorithms import BoundaryCondition
from pythonpic.classes import NonperiodicGrid, Simulation, Species
from pythonpic.helper_functions.physics import epsilon_zero, electric_charge, lightspeed, proton_mass, electron_rest_mass, \
    critical_density

VERSION = 32
laser_wavelength = 1.064e-6 # meters
laser_intensity = 1e21 # watt/meters squared
impulse_duration = 1e-13 # seconds

length = 1.0655e-5 # meters
total_time = 2e-13 # seconds
spatial_step = 7.7325e-9 # meters
number_cells = 1378

moat_length_left_side = 3.093e-6 # meters
# linear preplasma
preplasma_length = 7.73e-7 # meters
main_plasma_length = 7.73e-7 + preplasma_length # meters

print("crit density", critical_density(laser_wavelength))
maximum_electron_concentration = 5 * critical_density(laser_wavelength) # m^-3

# assert np.isclose(maximum_electron_concentration, 5.24e27), maximum_electron_concentration # m^-3
# maximum_electron_concentration = 5.24e27 # CHECK: this is a crutch

npic = 0.01 * critical_density(laser_wavelength)


N_MACROPARTICLES = int(maximum_electron_concentration * 1.5 * preplasma_length / npic / spatial_step)
n_macroparticles = N_MACROPARTICLES
scaling = npic

laser_polarization = "Ez"
individual_diagnostics = False

category_name = "benchmark_run"
# assert False
class initial(Simulation):
    def __init__(self, filename, n_macroparticles, n_cells):
        """
        A simulation of laser-hydrogen shield interaction.

        Parameters
        ----------
        filename : str
            Filename for the simulation.
        n_macroparticles : int
            Number of macroparticles for each species. The simulation is
            normalized to 75000 macroparticles by default,
        impulse_duration : float
            Duration of the laser impulse.
        laser_intensity : float
            Laser impulse intensity, in W/m^2. A good default is 1e21.
        perturbation_amplitude : float
            Amplitude of the initial position perturbation.
        """
        if laser_intensity:
            bc_laser = BoundaryCondition.bcs[laser_polarization](laser_intensity=laser_intensity,
                                         laser_wavelength=laser_wavelength,
                                         envelope_center_t = total_time/2,
                                         envelope_width=impulse_duration,
                                         envelope_power=6,
                                         c=lightspeed,
                                         epsilon_0=epsilon_zero,
                                         )
            print(f"Laser amplitude: {bc_laser.laser_amplitude:e}")
            bc = bc_laser
        else:
            bc = BoundaryCondition.BC
        grid = NonperiodicGrid(T=total_time, L=length, NG=n_cells, c =lightspeed, epsilon_0 =epsilon_zero, bc=bc)

        cells_per_wl = laser_wavelength / grid.dx
        print(f"{cells_per_wl:.1f} grid cells per laser wavelength.")
        vtherm = 2 * np.pi / cells_per_wl * lightspeed
        print(f"Thermal velocity for this simulation should be on the order of {vtherm / lightspeed:.3f}c.")

        if n_macroparticles:
            electrons = Species(-electric_charge, electron_rest_mass,
                                n_macroparticles, grid, "electrons", scaling,
                                individual_diagnostics=individual_diagnostics)
            protons = Species(electric_charge, proton_mass, n_macroparticles,
                              grid, "protons", scaling,
                              individual_diagnostics=individual_diagnostics)
            list_species = [electrons, protons]
        else:
            list_species = []

        description = "Benchmark run for laser interaction"

        super().__init__(grid, list_species,
                         filename=filename,
                         category_type="benchmark",
                         config_version=VERSION,
                         title=description,
                         considered_large = True)
        print("Simulation prepared.")

    def grid_species_initialization(self):
        for species in self.list_species:
            print(f"Distributing {species.name} nonuniformly.")
            species.distribute_uniformly(self.grid.L, 0,
                                         moat_length_left_side, moat_length_left_side)
        print("Finished initial distribution of particles.")
        super().grid_species_initialization(False)
        print("Finished initialization.")

