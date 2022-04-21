# coding=utf-8
import pytest
import numpy as np
from ..classes.species import n_saved_particles
from pythonpic.classes import Species, PeriodicTestGrid
from pythonpic.helper_functions.physics import electric_charge, electron_rest_mass, lightspeed, epsilon_zero

@pytest.fixture(params=np.logspace(0, 6, 12, dtype=int))
def n_available(request):
    return request.param

max_saved = n_available

def test_n_saved(n_available, max_saved):
    save_every_n, n_saved = n_saved_particles(n_available, max_saved)

    assert n_saved <= max_saved
    assert np.arange(n_available)[::save_every_n].size == n_saved

def test_n_saved_equal(n_available):
    save_every_n, n_saved = n_saved_particles(n_available, n_available)

    assert n_saved == n_available

@pytest.mark.parametrize("scaling", [1, 10, 100, 1000])
def test_species(scaling):
    g = PeriodicTestGrid(1e-8, 1, 100, c=lightspeed, epsilon_0=epsilon_zero)
    species = Species(electric_charge, electron_rest_mass, 1, g, scaling=scaling)
    species.v[:, 0] = 10
    kinetic_energy_single_electron = 4.554692e-29
    assert np.isclose(species.kinetic_energy, kinetic_energy_single_electron*scaling)