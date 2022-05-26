#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from grgrlib.jaxed import jax_print
from econpizza.utilities.interp import interpolate


def hh_init(a_grid, we, R, eis, T):
    """The initialization for the value function
    """

    coh = R * a_grid[jnp.newaxis, :] + \
        we[:, jnp.newaxis] + T[:, jnp.newaxis]
    Va = R * (0.1 * coh) ** (-1 / eis)
    return Va


def hh(Va_p, a_grid, we, T, R, beta, eis, frisch, vphi):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(
        uc_nextgrid, we[:, jnp.newaxis], eis, frisch, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid - we[:, jnp.newaxis] * n_nextgrid + \
        a_grid[jnp.newaxis, :] - T[:, jnp.newaxis]
    rhs = R * a_grid

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, jnp.newaxis] * n + T[:, jnp.newaxis] - c
    # fix consumption and labor for constrained households
    c, n = jnp.where(a < a_grid[0], solve_cn(
        we[:, jnp.newaxis], rhs + T[:, jnp.newaxis] - a_grid[0], eis, frisch, vphi, Va_p), jnp.array((c, n)))
    a = jnp.where(a > a_grid[0], a, a_grid[0])

    # calculate new MUC
    Va = R * c ** (-1 / eis)

    return Va, a, c, n


def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters
    """
    return jnp.array((uc ** (-eis), (w * uc / vphi) ** frisch))


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """

    def solve_uc_cond(cont):
        return jnp.abs(cont[0]).max() > 1e-11

    def solve_uc_body(cont):
        ne, log_uc = cont
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        log_uc -= ne / ne_p
        return ne, log_uc

    log_uc = jnp.log(uc_seed)

    _, log_uc = jax.lax.while_loop(
        solve_uc_cond, solve_uc_body, (uc_seed, log_uc))

    return jnp.exp(log_uc)


def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative
    """
    c, n = cn(jnp.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    tax_rule, div_rule = e_grid, e_grid
    div = Div / jnp.sum(pi_e * div_rule) * div_rule
    tax = Tax / jnp.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


def wages(w, e_grid):
    we = w * e_grid
    return we


def labor_supply(n, e_grid):
    ne = e_grid[:, jnp.newaxis] * n
    return ne
