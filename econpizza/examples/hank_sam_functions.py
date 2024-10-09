#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 30.09.2024

This file contains functions necessary for the solution of the HANK-SAM model.

Based on example functions file from Gregor Boehl, see:
https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank_functions.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from grgrjax import jax_print
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid


def egm_init(a_grid, skills_grid):
    """Initialise EGM.
    """
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2


def egm_step(Wa_p, a_grid, skills_grid, w, T, xi, R, beta, sigma_c):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Wa_p

    # benefits
    zeros_vector = jnp.zeros((1, len(skills_grid)-1))
    xi_reshaped = jnp.array([[xi]])
    benefit_schedule_temp = jnp.concatenate(
        [xi_reshaped, zeros_vector], axis=1)
    benefit_schedule = benefit_schedule_temp[0]

    # labour income and consumption
    # labor_inc = skills_grid[:, None]*n*w + benefit_schedule[:, None]
    labor_inc = skills_grid[:, None]*w + benefit_schedule[:, None]
    c_nextgrid = ux_nextgrid**(-1/sigma_c)

    # get consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = R * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)

    # get todays distribution of assets
    a = rhs + labor_inc + T[:, None] - c

    # fix consumption and labor for constrained households
    c = jnp.where(a < a_grid[0], labor_inc + rhs +
                  T[:, None] - a_grid[0], c)
    a = jnp.where(a < a_grid[0], a_grid[0], a)

    # calculate new MUC
    Wa = R * (c**(-sigma_c))

    return Wa, a, c


def mean(x, pi):
    """Mean of discretized random variable with support x and probability mass function pi."""
    return jnp.sum(pi * x)


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return jnp.sum(pi * (x - jnp.sum(pi * x)) ** 2)


def markov_rouwenhorst(rho, sigma, N):
    """Rouwenhorst method"""

    # Explicitly typecast N as an integer, since when the grid constructor functions
    # (e.g. the function that makes a_grid) are implemented as blocks, they interpret the integer-valued calibration
    # as a float.
    N = int(N)

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = jnp.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (jnp.zeros((n, n)) for _ in range(4))
        P1 = P1.at[:-1, :-1].set(p * Pi)
        P2 = P2.at[:-1, 1:].set((1 - p) * Pi)
        P3 = P3.at[1:, :-1].set((1 - p) * Pi)
        P4 = P4.at[1:, 1:].set(p * Pi)
        Pi = P1 + P2 + P3 + P4
        Pi = Pi.at[1:-1].divide(2)

    # invariant distribution and scaling
    pi = stationary_distribution(jnp.array(Pi.T))
    y = rouwenhorst_grid_from_stationary(sigma, pi)
    return y, pi, Pi


def rouwenhorst_grid_from_stationary(sigma, stationary_distribution):
    s = jnp.linspace(-1, 1, len(stationary_distribution))
    s *= (sigma / jnp.sqrt(variance(s, stationary_distribution)))
    return jnp.exp(s) / jnp.sum(stationary_distribution * jnp.exp(s))


def get_sam_grid_from_stationary(sigma, stationary_distribution):
    s = jnp.linspace(-1, 1, len(stationary_distribution)-1)
    s *= (sigma / jnp.sqrt(variance(s, stationary_distribution[1:])))
    scaled_grid = jnp.exp(
        s) / jnp.sum(stationary_distribution[1:] * jnp.exp(s))

    augmented_grid = jnp.concatenate(
        [jnp.array([0.0]), scaled_grid])  # unemployment state
    return augmented_grid


def stationary_distribution(T):
    """Find invariant distribution of a Markov chain by unit eigenvector.
    NOTE: jax has no autodiff support for eig. (there is a version with custom_jvp in grgrwip)
    """

    v, w = jnp.linalg.eig(T)

    # using sorted args instead of np.isclose is neccessary for jax-jitting
    args = jnp.argsort(v)
    unit_ev = w[:, args[-1]]

    return unit_ev.real / unit_ev.real.sum()


def sam_markov(rho, sigma, N, find_rate, separate_rate):
    """Obtain transition matrix, stationary distribution and skill levels with
    unemployment state (which corresponds to skill = 0).
    """
    # Get transition matrix and its stationary distribution for the employed
    # agents switching skills
    _, pi, Pi = markov_rouwenhorst(rho, sigma, N)

    # Get number of states of standard grid
    pi_len = len(pi)

    # Create new transition matrix
    new_row_1 = jnp.array([1 - find_rate])
    new_row_2 = find_rate * pi
    new_row = jnp.concatenate((new_row_1, new_row_2), axis=0)
    new_row = new_row.reshape(1, len(new_row))

    new_col = separate_rate * jnp.array((jnp.ones(pi_len)))
    new_col = new_col.reshape(len(new_col), 1)
    # Scale the original transition matrix
    scaled_Pi = (1 - separate_rate) * Pi
    new_lower = jnp.concatenate((new_col, scaled_Pi), axis=1)

    Pi_new = jnp.concatenate((new_row, new_lower), axis=0)
    row_sums = jnp.sum(Pi_new, axis=1, keepdims=True)
    sam_Pi = Pi_new / row_sums

    # Get new stationary distribution
    sam_pi = stationary_distribution(jnp.array(sam_Pi.T))

    # Get new grid of skill levels, incl. unemployment state
    sam_y = get_sam_grid_from_stationary(sigma, sam_pi)

    return sam_y, sam_pi, pi, sam_Pi


def get_unemployment(dist):
    """Calculate unemployment rate, given the distribution of households over states.
    """
    dist_over_skills = jnp.sum(dist, axis=1)
    return dist_over_skills[0]
