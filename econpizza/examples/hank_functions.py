# -*- coding: utf-8 -*-
"""functions for the one-asset HANK model without labor choice. Heavily inspired by https://github.com/shade-econ/sequence-jacobian/#sequence-space-jacobian
"""

import jax
import jax.numpy as jnp
from grgrjax import jax_print
from econpizza.utilities.interp import interpolate


def egm_init(a_grid, skills_grid):
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2


def egm_step(Wa_p, a_grid, skills_grid, w, n, T, R, beta, sigma_c, sigma_l):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Wa_p

    # consumption can be readily obtained from MUC and MU of labor
    labor_inc = skills_grid[:, None]*n*w
    c_nextgrid = ux_nextgrid**(-1/sigma_c) + labor_inc/(1 + sigma_l)

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
    Wa = R * (c - labor_inc/(1 + sigma_l)) ** (-sigma_c)

    return Wa, a, c


def transfers(skills_stationary, Div, Tax, skills_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    rule = skills_grid
    div = Div / jnp.sum(skills_stationary * rule) * rule
    tax = Tax / jnp.sum(skills_stationary * rule) * rule
    T = div - tax
    return T
