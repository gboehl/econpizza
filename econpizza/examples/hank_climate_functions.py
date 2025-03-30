# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from grgrjax import jax_print
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid


def egm_init_ones(a_grid, skills_grid):
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))


def egm_step(VPrime, VaPrime, a_grid, skills_grid, w, n, T, R, beta, sigma_c, sigma_l):
    """A single backward step via EGM
    """

    # MU of composite good X as implied by next periods value function
    ux_nextgrid = beta * VaPrime
    # consumption can already be obtained from MUC and MU of labor
    labor_inc = skills_grid[:, None]*n*w
    c_nextgrid = ux_nextgrid**(-1/sigma_c) + labor_inc/(1 + sigma_l)

    # get consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = R * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)
    # get current distribution of assets
    a = rhs + labor_inc + T[:, None] - c

    # fix consumption and labor for constrained households
    c = jnp.where(a < a_grid[0], labor_inc + rhs +
                  T[:, None] - a_grid[0], c)
    a = jnp.where(a < a_grid[0], a_grid[0], a)

    # calculate new Values & MUX 
    x = c - labor_inc/(1 + sigma_l)
    V = x ** (1-sigma_c) / (1-sigma_c) + beta*VPrime
    Va = R * x ** (-sigma_c)

    return V, Va, a, c


def please_the_rich(skills_stationary, income):
    # hardwired incidence rules are proportional to skill; scale does not matter
    out = jnp.zeros_like(skills_stationary)
    out = out.at[-1].set(income/skills_stationary[-1])
    return out


def transfers(skills_stationary, income, rule):
    # hardwired incidence rules are proportional to skill; scale does not matter
    return income / jnp.sum(skills_stationary * rule) * rule


def special_grid(amax, n, amin):
    """A log grid that always contains 0
    """
    if amin == 0:
        return log_grid(amax, n, amin)
    grid = log_grid(amax, n-1, amin)
    grid = jnp.hstack((grid,0))
    return jnp.sort(grid)


