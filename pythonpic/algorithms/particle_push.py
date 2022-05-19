# coding=utf-8
"""mathematical algorithms for the particle pusher, Leapfrog and Boris"""
import numpy as np
from numba import jit, njit
from multiprocessing import Pool, Array, Value

import time

@jit()
def boris_velocity_kick(v, eff_q, E, B, dt, eff_m):
    """
    The velocity update portion of the Boris pusher. Updates the velocity in place so as to conserve memory.

    Parameters
    ----------
    v : `numpy.ndarray`
        Array of velocities, of shape `(N, 3)`, `N` being the number of macroparticles
    eff_q : `float`
        The effective charge of the particles (total charge in the macroparticle)
    E : `numpy.ndarray`
        Interpolated or calculated values of the electric field. Shape `(N, 3)`.
    B : `numpy.ndarray`
        Interpolated or calculated values of the magnetic field. Shape `(N, 3)`.
    dt : `float`
        Timestep duration.
    eff_m : `float`
        The effective mass of the particles (total mass in the macroparticle)

    Returns
    -------
    float
        The kinetic energy of the particles being pushed.

    """
    # calculate u
    # gamma = 1 / ()
    # u = v /
    vminus = v + eff_q * E / eff_m * dt * 0.5

    # rotate to add magnetic field
    t = B * eff_q / eff_m * dt * 0.5
    s = 2 * t / (1 + (t * t).sum(axis=1, keepdims=True))

    vprime = vminus + np.cross(vminus, t)
    vplus = vminus + np.cross(vprime, s)
    v_new = vplus + eff_q * E / eff_m * dt * 0.5

    energy = (v_new * v * (0.5 * eff_m)).sum()
    v[:] = v_new
    return energy

def init(_half_force_0, _half_force_1, _half_force_2, _E_0, _E_1, _E_2, _constant_arr):
    global half_force_0, half_force_1, half_force_2, E_0, E_1, E_2, constant_arr
    half_force_0 = _half_force_0
    half_force_1 = _half_force_1
    half_force_2 = _half_force_2
    E_0 = _E_0
    E_1 = _E_1
    E_2 = _E_2
    constant_arr = _constant_arr

def calculate_half_force(i):
    half_force_0[i] = constant_arr[i] * E_0[i]
    half_force_1[i] = constant_arr[i] * E_1[i]
    half_force_2[i] = constant_arr[i] * E_2[i]

#@jit("f8(f8[:,:],f8,f8,f8[:,:],f8[:,:],f8,f8)")
def rela_boris_velocity_kick(v, c, eff_q, E, B, dt, eff_m):
    """
    The velocity update portion of the Boris pusher. Updates the velocity in place so as to conserve memory.

    Parameters
    ----------
    v : `numpy.ndarray`
        Array of velocities, of shape `(N, 3)`, `N` being the number of macroparticles
    c : `float`
        The speed of light
    eff_q : `float`
        The effective charge of the particles (total charge in the macroparticle)
    E : `numpy.ndarray`
        Interpolated or calculated values of the electric field. Shape `(N, 3)`.
    B : `numpy.ndarray`
        Interpolated or calculated values of the magnetic field. Shape `(N, 3)`.
    dt : `float`
        Timestep duration.
    eff_m : `float`
        The effective mass of the particles (total mass in the macroparticle)

    Returns
    -------
    float
        The kinetic energy of the particles being pushed.

    """
    # calculate u
    v /= np.sqrt(1 - (v ** 2).sum(axis=1, keepdims=True) / c ** 2)  # below eq 22 LPIC

    N = len(v)

    constant = (eff_q * 0.5 / eff_m * dt)

    half_force_0 = Array('f', range(N))
    half_force_1 = Array('f', range(N))
    half_force_2 = Array('f', range(N))
    E_0 = Array('f', E[:,0])
    E_1 = Array('f', E[:,1])
    E_2 = Array('f', E[:,2])
    constant_arr = Array('f', np.full(N, constant))

    #pp = Pool(initializer=init, initargs=(half_force_0, half_force_1, half_force_2, E_0, E_1, E_2, constant_arr))
    #Pool(4)
    #pp.map(calculate_half_force, range(N))

    half_force = np.ndarray(shape=(N,3), dtype='float64')
    #np.insert(half_force, 0, half_force_0[:])
    #np.insert(half_force, 1, half_force_1[:])
    #np.insert(half_force, 2, half_force_2[:])

    #start_time = time.time()

    #with pymp.Parallel(4) as p:
    for i in range(N):
        half_force[i]= (eff_q * 0.5 / eff_m * dt) * E[i]

    #runtime = time.time() - start_time
    #print("Runtime: ", runtime)

    #half_force = (eff_q * 0.5 / eff_m * dt) * E  # eq. 21 LPIC # array of shape (N_particles, 3)
    # add first half of electric force

    # calculate uminus: initial velocity with added half impulse
    v += half_force

    # rotate to add magnetic field
    # this effectively takes relativistic mass into account
    #t = B * eff_q * dt / (2 * eff_m * np.sqrt(1 + (v ** 2).sum(axis=1, keepdims=True) / c ** 2))

    t = np.ndarray(shape=(N,3), dtype='float64')

    #with pymp.Parallel(4) as p:
    for i in range(N):
        t[i] = B[i] * eff_q * dt / (2 * eff_m * np.sqrt(1 + (v[i] ** 2).sum() / c ** 2))

    # u' = u- + u- x t
    uprime = v + np.cross(v, t)
    # rotate second time, by s = 2t/(1+t*t)
    t *= 2
    t /= 1 + (t * t).sum(axis=1, keepdims=True)
    # u+ = u- + u' x s
    v += np.cross(uprime, t)

    # add second half of electric force
    v += half_force

    final_gamma = np.sqrt(1 + ((v ** 2).sum(axis=1, keepdims=True) / c ** 2))
    v /= final_gamma
    total_velocity = final_gamma - 1
    return total_velocity.sum() * eff_m * c ** 2

def boris_push(species, E: np.ndarray, dt: float, B: np.ndarray):
    """
    Implements the relativistic Boris pusher.
    Mostly a wrapper function for the compiled version in `rela_boris_velocity_kick`.

    Note that velocity is updated in-place to conserve memory!

    Parameters
    ----------
    species : `pythonpic.classes.Species`
    E : `numpy.ndarray`
        Interpolated or calculated values of the electric field. Shape `(N, 3)`.
    dt : float
        Timestep duration
    B : `numpy.ndarray`
        Interpolated or calculated values of the magnetic field. Shape `(N, 3)`.
    Returns
    -------
    `numpy.ndarray`
        Updated positions, shape `(N, )`.
    `numpy.ndarray`
        Updated velocity, shape `(N, 3)`.
    `float`
        Total kinetic energy of the particles.
    """
    energy = boris_velocity_kick(species.v, species.eff_q,
                                 E, B, dt, species.eff_m)
    return energy
def rela_boris_push(species, E: np.ndarray, dt: float, B: np.ndarray):
    """
    Implements the relativistic Boris pusher.
    Mostly a wrapper function for the compiled version in `rela_boris_velocity_kick`.

    Note that velocity is updated in-place to conserve memory!

    Parameters
    ----------
    species : `pythonpic.classes.Species`
    E : `numpy.ndarray`
        Interpolated or calculated values of the electric field. Shape `(N, 3)`.
    dt : float
        Timestep duration
    B : `numpy.ndarray`
        Interpolated or calculated values of the magnetic field. Shape `(N, 3)`.
    Returns
    -------
    `numpy.ndarray`
        Updated positions, shape `(N, )`.
    `numpy.ndarray`
        Updated velocity, shape `(N, 3)`.
    `float`
        Total kinetic energy of the particles.
    """
    energy = rela_boris_velocity_kick(species.v, species.c, species.eff_q,
                                      E, B, dt, species.eff_m)
    return energy

