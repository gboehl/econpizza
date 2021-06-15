#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from interpolation import interp
from grgrlib.njitted import histogram_weights


@njit
def egm_ces_one_asset(bs, par, tol=1e-8):

    rb, rl, gamma, beta, p, x, y = par
    gy = bs.copy()
    g0 = bs.copy()
    cnt = 0

    while True:

        cnt += 1

        gy_old = gy
        g0_old = g0

        giy = lambda x: interp(gy, bs, x)
        gi0 = lambda x: interp(g0, bs, x)

        r = np.ones_like(bs) * rb
        ry = r.copy()
        r0 = r.copy()
        r[bs >= 0] = rl

        c = (
            beta
            * r
            * (
                p * (y + x - giy(bs) + r * bs) ** -gamma
                + (1 - p) * (x - gi0(bs) + r * bs) ** -gamma
            )
        ) ** (1 / -gamma)

        gy = bs + c - y - x
        g0 = bs + c - x
        ry[gy >= 0] = rl
        r0[g0 >= 0] = rl
        gy /= ry
        g0 /= r0

        erry = np.abs(gy - gy_old).max()
        err0 = np.abs(g0 - g0_old).max()
        err = max(err0, erry)

        if err < tol:
            break

    return gy, g0, c, cnt


@njit
def find_stst_dist(bins, g0, gy, bs, prob, weights=None):
    # must be the funcs and the probs

    giy = lambda x: interp(gy, bs, x)
    gi0 = lambda x: interp(g0, bs, x)

    if weights is None:
        wths = np.ones(n) / n
    else:
        wths = weights

    n = len(bins)

    # chose new bins such that the original bins are approximately their midpoints
    new_bins = np.empty(n + 1)
    diff = np.diff(bins) / 2
    new_bins[0] = bins[0] - diff[0]
    new_bins[1:-1] = bins[:-1] + diff
    new_bins[-1] = bins[-1] + diff[-1]

    m = np.empty(n * 2)
    w = m.copy()
    cnt = 0

    while True:

        cnt += 1
        wths_old = wths.copy()

        m[:n] = gi0(bins)
        m[n:] = giy(bins)
        w[:n] = (1 - prob) * wths
        w[n:] = prob * wths

        wths, _ = histogram_weights(m, bins=new_bins, weights=w)

        err = np.abs(wths - wths_old).max()

        if err < 1e-8:
            break

    return bins, wths, cnt
