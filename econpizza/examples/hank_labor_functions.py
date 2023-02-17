# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from grgrjax import jax_print, amax
from econpizza.utilities.interp import interpolate


def hh_init(a_grid, we, R, sigma_c, T):
    """The initialization for the value function
    """

    coh = R * a_grid[None, :] + \
        we[:, None] + T[:, None]
    Va = R * (0.1 * coh) ** (-1 / sigma_c)
    return Va


def hh(Va_p, a_grid, we, T, R, beta, sigma_c, frisch, vphi):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(
        uc_nextgrid, we[:, None], sigma_c, frisch, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid - we[:, None] * n_nextgrid + \
        a_grid[None, :] - T[:, None]
    rhs = R * a_grid

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + T[:, None] - c
    # fix consumption and labor for constrained households
    c, n = jnp.where(a < a_grid[0], solve_cn(
        we[:, None], rhs + T[:, None] - a_grid[0], sigma_c, frisch, vphi, Va_p), jnp.array((c, n)))
    a = jnp.where(a > a_grid[0], a, a_grid[0])

    # calculate new MUC
    Va = R * c ** (-1 / sigma_c)

    return Va, a, c, n


def cn(uc, w, sigma_c, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters
    """
    return jnp.array((uc ** (-sigma_c), (w * uc / vphi) ** frisch))


def solve_cn(w, T, sigma_c, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, sigma_c, frisch, vphi, uc_seed)
    return cn(uc, w, sigma_c, frisch, vphi)


def solve_uc_cond(carry):
    ne, _, _ = carry
    return amax(ne) > 1e-8


def solve_uc_body(carry):
    ne, log_uc, pars = carry
    ne, ne_p = netexp(log_uc, *pars)
    log_uc -= ne / ne_p
    return ne, log_uc, pars


def solve_uc(w, T, sigma_c, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/sigma_c) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc = jnp.log(uc_seed)
    pars = w, T, sigma_c, frisch, vphi
    _, log_uc, _ = jax.lax.while_loop(
        solve_uc_cond, solve_uc_body, (uc_seed, log_uc, pars))
    return jnp.exp(log_uc)


def netexp(log_uc, w, T, sigma_c, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative
    """
    c, n = cn(jnp.exp(log_uc), w, sigma_c, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -sigma_c and frisch wrt log u'(c)
    c_loguc = -sigma_c * c
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
    ne = e_grid[:, None] * n
    return ne
