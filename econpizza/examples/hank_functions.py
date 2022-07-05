#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from grgrlib.jaxed import jax_print
from econpizza.utilities.interp import interpolate


def hh_init(a_grid, e_grid):
    return jnp.ones((e_grid.shape[0], a_grid.shape[0]))*1e-2


def hh(Va_p, a_grid, skills_grid, w, n, T, R, beta, eis, frisch):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Va_p

    # consumption can be readily obtained from MUC and MU of labor
    labor_inc = skills_grid[:, None]*n*w
    c_nextgrid = ux_nextgrid**-eis + labor_inc/(1 + 1/frisch)

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
    Va = R * (c - labor_inc/(1 + 1/frisch)) ** (-1 / eis)

    return Va, a, c


def transfers(age_stationary, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    rule = e_grid
    div = Div / jnp.sum(age_stationary * rule) * rule
    tax = Tax / jnp.sum(age_stationary * rule) * rule
    T = div - tax
    return T
