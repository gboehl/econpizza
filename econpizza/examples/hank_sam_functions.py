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
from econpizza.utilities.grids import log_grid, markov_rouwenhorst, variance, rouwenhorst_grid_from_stationary
from econpizza.utilities.dists import stationary_distribution


def egm_init(a_grid, skills_grid):
    """Initialise EGM.
    """
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2


def egm_step(Wa_p, a_grid, skills_grid, w, T, xi, R, beta, sigma_c):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Wa_p

    # labour income and consumption
    benefit_schedule = jnp.zeros_like(skills_grid).at[0].set(xi)
    labor_inc = skills_grid[:, None]*w + benefit_schedule[:, None]
    c_nextgrid = ux_nextgrid**(-1/sigma_c)

    # get consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = R * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)

    # get todays distribution of assets
    a = rhs + labor_inc + T[:, None] - c

    # fix consumption and labor for constrained households
    c = jnp.where(a < a_grid[0], labor_inc + rhs + T[:, None] - a_grid[0], c)
    a = jnp.where(a < a_grid[0], a_grid[0], a)

    # calculate new MUC
    Wa = R * (c**(-sigma_c))

    return Wa, a, c


def sam_markov(rho, sigma, N, find_rate, separate_rate):
    """Obtain transition matrix, stationary distribution and skill levels with
    unemployment state (which corresponds to skill = 0).
    """
    # get transition matrix & stationary distribution for employed agents switching skills
    _, pi, Pi = markov_rouwenhorst(rho, sigma, N)
    # create new transition matrix
    sam_Pi = jnp.block([[1-find_rate, find_rate*pi], [separate_rate*jnp.ones((N,1)), (1 - separate_rate)*Pi]])

    # get new stationary distribution
    sam_pi = stationary_distribution(sam_Pi.T)
    # get new grid of skill levels, incl. unemployment state
    scaled_grid = rouwenhorst_grid_from_stationary(sigma, sam_pi[1:])
    sam_y = jnp.hstack((0, scaled_grid))  # unemployment state

    return sam_y, sam_pi, sam_Pi


def get_unemployment(dist):
    """Calculate unemployment rate, given the distribution of households over states.
    """
    dist_over_skills = jnp.sum(dist, axis=1)
    return dist_over_skills[0]
