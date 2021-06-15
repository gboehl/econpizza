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


# @njit(cache=True, fastmath=True)
def find_stst_dist(bins, x, fx, probs, max_iter=1000, tol=1e-8, weights=None):

    n = len(bins)
    prob_g, prob_b = probs

    if weights is None:
        wths = np.ones((2, n)) / n
        wths *= stationary_dist_mc(probs).reshape(-1, 1)
    else:
        wths = weights

    new_bins = np.empty(n + 1)
    # chose new bins such that the original bins are their midpoints
    new_bins[1:-1] = bins[:-1] + np.diff(bins) / 2
    # first and last bin catch everything that is off-grid
    new_bins[0] = -np.inf
    new_bins[-1] = np.inf

    grid_g = interp(x[0], fx, bins)
    grid_b = interp(x[1], fx, bins)
    sorting_index_g = np.argsort(grid_g)
    sorting_index_b = np.argsort(grid_b)
    sorted_grid_g = grid_g[sorting_index_g]
    sorted_grid_b = grid_b[sorting_index_b]
    bin_index_g = np.searchsorted(sorted_grid_g, new_bins)
    bin_index_b = np.searchsorted(sorted_grid_b, new_bins)

    new_wths = np.empty_like(wths)

    cnt = 0
    flag = False

    while True:

        cnt += 1

        if cnt > max_iter:
            flag = True
            break

        old_wths = wths.copy()

        new_wths[0, :] = prob_g * wths[0] + prob_b * wths[1]
        new_wths[1, :] = (1 - prob_g) * wths[0] + (1 - prob_b) * wths[1]

        sorted_new_wths_g = new_wths[0][sorting_index_g]
        sorted_new_wths_b = new_wths[1][sorting_index_b]
        cum_new_wths_g = np.hstack((np.zeros(1), sorted_new_wths_g.cumsum()))
        cum_new_wths_b = np.hstack((np.zeros(1), sorted_new_wths_b.cumsum()))
        wths[0, :] = np.diff(cum_new_wths_g[bin_index_g])
        wths[1, :] = np.diff(cum_new_wths_b[bin_index_b])

        if np.abs(wths - old_wths).max() < tol:
            break

    return wths, flag


@njit(cache=True, fastmath=True)
def stationary_dist_mc(*probs):

    probs = np.array(probs)
    mat = np.vstack((probs, 1 - probs))
    v, w = np.linalg.eig(mat)

    return np.ravel(w[:, v == 1] / np.sum(w[:, v == 1]))
