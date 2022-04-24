#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)


def hh_init(a_grid, we, r, eis, T):
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + \
        we[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return fininc, Va

# this is double checked against the original and works
# there is potentially room for more vectorization


@jax.jit
def hh(Va_p, a_grid, we, T, r, beta, eis, frisch, vphi):
    '''Single backward step via EGM.'''

    uc_nextgrid = beta * Va_p
    c_nextgrid, n_nextgrid = cn(
        uc_nextgrid, we[:, np.newaxis], eis, frisch, vphi)

    lhs = c_nextgrid - we[:, np.newaxis] * n_nextgrid + \
        a_grid[np.newaxis, :] - T[:, np.newaxis]
    rhs = (1 + r) * a_grid

    # interpolation was hand-taylored in ssj. So there certainly is a better solution than this
    c = jax.vmap(np.interp)(rhs.broadcast((lhs.shape[0],)), lhs, c_nextgrid)
    n = jax.vmap(np.interp)(rhs.broadcast((lhs.shape[0],)), lhs, n_nextgrid)

    a = rhs + we[:, np.newaxis] * n + T[:, np.newaxis] - c
    # inds = a < a_grid[0]
    # inds = jax.lax.lt(a, a_grid[0])
    # print(inds.shape)
    # solve_cn(we[:, np.newaxis], rhs + T[:, np.newaxis] - a_grid, eis, frisch, vphi, Va_p)
    # solve_cn(we.broadcast((lhs.shape[1],)).T[inds], rhs[inds] + T.broadcast((lhs.shape[1],)).T[inds] - a_grid[inds], eis, frisch, vphi, Va_p)
    # cna = solve_cn(we.broadcast((lhs.shape[1],)).T[inds], rhs.broadcast((lhs.shape[0],))[inds] + T.broadcast((lhs.shape[1],)).T[inds] - a_grid.broadcast((lhs.shape[0],))[inds], eis, frisch, vphi, Va_p[inds])

    # c = c.at[inds].set(cna[0])
    # n = n.at[inds].set(cna[1])
    c, n = np.where(a < a_grid[0], solve_cn(
        we[:, np.newaxis], rhs + T[:, np.newaxis] - a_grid, eis, frisch, vphi, Va_p), np.array((c, n)))
    a = np.where(a > a_grid[0], a, a_grid)

    Va = (1 + r) * c ** (-1 / eis)

    return Va, a, c, n


'''Supporting functions for HA block'''


def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return np.array((uc ** (-eis), (w * uc / vphi) ** frisch))


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """

    def solve_uc_cond(cont):
        return np.abs(cont[0]).max() > 1e-11

    def solve_uc_body(cont):
        ne, log_uc = cont
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        log_uc -= ne / ne_p
        return (ne, log_uc)

    log_uc = np.log(uc_seed)

    _, log_uc = jax.lax.while_loop(
        solve_uc_cond, solve_uc_body, (uc_seed, log_uc))

    return np.exp(log_uc)


def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


def wages(w, e_grid):
    we = w * e_grid
    return we


def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne
