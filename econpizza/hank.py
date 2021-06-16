#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from interpolation import interp


@njit(cache=True, fastmath=True)
def egm_ces_1asset(bs, par, max_iter=1000, tol=1e-8):

    rb, rl, sigma, beta, prob_g, prob_b, inc_g, inc_b = par
    x = np.vstack((bs, bs))
    cnt = 0
    flag = False

    rPrime = np.ones_like(bs) * rl
    rPrime[bs < 0] = rb
    r = np.empty_like(x)

    while True:

        cnt += 1

        if cnt > max_iter:
            flag = True
            break

        x_old = x.copy()

        fg = interp(x[0], bs, bs)
        fb = interp(x[1], bs, bs)

        cg = (
            beta
            * rPrime
            * (
                prob_g * (inc_g - fg + rPrime * bs) ** -sigma
                + (1 - prob_g) * (inc_b - fb + rPrime * bs) ** -sigma
            )
        ) ** (1 / -sigma)
        cb = (
            beta
            * rPrime
            * (
                prob_b * (inc_g - fg + rPrime * bs) ** -sigma
                + (1 - prob_b) * (inc_b - fb + rPrime * bs) ** -sigma
            )
        ) ** (1 / -sigma)

        x[0] = bs + cg - inc_g
        x[1] = bs + cb - inc_b

        r[:] = rb
        r[0][x[0] >= 0] = rl
        r[1][x[1] >= 0] = rl

        x /= r

        if np.abs(x - x_old).max() < tol:
            break

    c = np.vstack((cg, cb))

    return x, c, flag


@njit(cache=True, fastmath=True)
def find_stst_dist(bins, x, fx, probs, max_iter=1000, tol=1e-8, weights=None):
    """find stationary distribution of 2-state markov chain by iterating on the distribution.
    This is way faster than constructing the transition matrix and calculating the stationary distribution via unity eigenvalues!
    """

    n = len(bins)
    tmat = np.vstack((np.array(probs), 1 - np.array(probs)))

    if weights is None:
        wths = np.ones((2, n)) / n
        wths *= stationary_dist_mc(probs).reshape(-1, 1)
    else:
        wths = weights

    new_bins = np.empty(n + 1)
    # chose new bins such that the original bins are their midpoints
    new_bins[1:-1] = bins[:-1] + np.diff(bins) / 2
    # first and last bin catch everything that is off-grid (they won't carry any weight)
    new_bins[0] = -np.inf
    new_bins[-1] = np.inf

    grid_g = interp(x[0], fx, bins)
    grid_b = interp(x[1], fx, bins)
    bin_inds_g = np.searchsorted(grid_g, new_bins)
    bin_inds_b = np.searchsorted(grid_b, new_bins)

    cnt = 0
    flag = False

    while True:

        cnt += 1

        if cnt > max_iter:
            flag = True
            break

        old_wths = wths.copy()
        wths = tmat @ wths

        cum_wths_g = np.hstack((np.zeros(1), wths[0].cumsum()))
        cum_wths_b = np.hstack((np.zeros(1), wths[1].cumsum()))
        wths[0, :] = np.diff(cum_wths_g[bin_inds_g])
        wths[1, :] = np.diff(cum_wths_b[bin_inds_b])

        if np.abs(wths - old_wths).max() < tol:
            break

    return wths, flag


@njit(cache=True, fastmath=True)
def stationary_dist_mc(*probs):

    probs = np.array(probs)
    mat = np.vstack((probs, 1 - probs))
    v, w = np.linalg.eig(mat)

    return np.ravel(w[:, v == 1] / np.sum(w[:, v == 1]))
