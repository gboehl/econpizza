#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from interpolation import interp


@njit(cache=True, fastmath=True)
def egm_ces_1asset(bs, par, max_iter=1000, tol=1e-8):

    rb, rl, sigma, beta, p, inc_b, inc_g = par
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

        c = (
            beta
            * rPrime
            * (
                p * (inc_g - fg + rPrime * bs) ** -sigma
                + (1 - p) * (inc_b - fb + rPrime * bs) ** -sigma
            )
        ) ** (1 / -sigma)

        x[0] = bs + c - inc_g
        x[1] = bs + c - inc_b

        r[:] = rb
        r[0][x[0] >= 0] = rl
        r[1][x[1] >= 0] = rl

        x /= r

        if np.abs(x - x_old).max() < tol:
            break

    return x, flag


@njit(cache=True, fastmath=True)
def find_stst_dist(bins, x, fx, prob, max_iter=1000, tol=1e-8, weights=None):

    n = len(bins)

    if weights is None:
        wths = np.ones(n) / n
    else:
        wths = weights

    new_bins = np.empty(n + 1)
    # chose new bins such that the original bins are their midpoints
    new_bins[1:-1] = bins[:-1] + np.diff(bins) / 2
    # first and last bin catch everything that is off-grid
    new_bins[0] = -np.inf
    new_bins[-1] = np.inf

    grid = np.hstack((interp(x[1], fx, bins), interp(x[0], fx, bins)))
    sorting_index = np.argsort(grid)
    sorted_grid = grid[sorting_index]
    bin_index = np.searchsorted(sorted_grid, new_bins)

    new_wths = np.empty_like(grid)

    cnt = 0
    flag = False

    while True:

        cnt += 1

        if cnt > max_iter:
            flag = True
            break

        old_wths = wths.copy()

        new_wths[:n] = (1 - prob) * wths
        new_wths[n:] = prob * wths

        sorted_new_wths = new_wths[sorting_index]
        cum_new_wths = np.hstack((np.zeros(1), sorted_new_wths.cumsum()))
        wths = np.diff(cum_new_wths[bin_index])

        if np.abs(wths - old_wths).max() < tol:
            break

    return wths, flag
