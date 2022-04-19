#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as np


@jax.jit
def interpolate_coord_robust_vector(x, xq):
    """Does interpolate_coord_robust where xq must be a vector, more general function is wrapper"""

    xqi = np.searchsorted(x, xq, side='right') - 1
    xqpi = (x[xqi+1] - xq) / (x[xqi+1] - x[xqi])

    return xqi, xqpi


@jax.jit
def interpolate_coord_robust(x, xq, check_increasing=False):
    """Linear interpolation exploiting monotonicity only in data x, not in query points xq.
    Simple binary search, less efficient but more robust.
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Main application intended to be universally-valid interpolation of policy rules.
    Dimension k is optional.

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (k, nq), query points (in any order)

    Returns
    ----------
    xqi  : array (k, nq), indices of lower bracketing gridpoints
    xqpi : array (k, nq), weights on lower bracketing gridpoints
    """
    if x.ndim != 1:
        raise ValueError(
            'Data input to interpolate_coord_robust must have exactly one dimension')

    if check_increasing and np.any(x[:-1] >= x[1:]):
        raise ValueError(
            'Data input to interpolate_coord_robust must be strictly increasing')

    if xq.ndim == 1:
        return interpolate_coord_robust_vector(x, xq)
    else:
        i, pi = interpolate_coord_robust_vector(x, xq.ravel())
        return i.reshape(xq.shape), pi.reshape(xq.shape)


@jax.jit
def tmat_from_exog(probs, D):

    nZ, nX = D.shape
    tmat = np.zeros((nX*nZ, nX*nZ))

    i = np.arange(nZ)*nX
    xi = np.arange(nX)[:, np.newaxis]
    zi = np.arange(nZ)[:, np.newaxis, np.newaxis]
    tmat = tmat.at[xi+zi*nX, xi +
                   i].set(probs.broadcast_in_dim((nZ, nX, nZ), (0, 2)))

    return tmat


@jax.jit
def tmat_from_endo(x_i, probs):

    nZ, nX = x_i.shape
    tmat = np.zeros((nX*nZ, nX*nZ))

    ix = np.arange(nX*nZ)
    j = np.arange(nZ).repeat(nX)*nX
    i = x_i.ravel()
    pi = probs.ravel()

    tmat = tmat.at[i+j, ix].add(pi)
    tmat = tmat.at[i+1+j, ix].add(1-pi)

    return tmat


@jax.jit
def forward_policy_1d(D, x_i, x_pi):
    nZ, nX = D.shape
    Dnew = np.zeros_like(D)

    j = np.arange(nZ)[:, np.newaxis]

    Dnew = Dnew.at[j, x_i].add(D * x_pi)

    return Dnew.at[j, x_i+1].add(D * (1 - x_pi))


@jax.jit
def stationary_distribution_forward_policy_1d_fixed_n(endog_inds, endog_probs, exog_probs, n=1000):

    dist = np.ones_like(endog_inds)
    dist /= dist.sum()

    def body(x, dist): return exog_probs.T @ forward_policy_1d(dist,
                                                               endog_inds, endog_probs)

    return jax.lax.fori_loop(0, 1000, body, dist)


@jax.jit
def stationary_distribution_forward_policy_1d(endog_inds, endog_probs, exog_probs, tol=1e-10):

    dist = np.ones_like(endog_inds)
    dist /= dist.sum()

    def cond_func(cont):
        return np.abs(cont[0]-cont[1]).max() > tol

    def body_func(cont):
        dist, _ = cont
        return exog_probs.T @ forward_policy_1d(dist, endog_inds, endog_probs), dist

    return jax.lax.while_loop(cond_func, body_func, (dist, dist+1))[0]


@jax.jit
def stationary_distribution(T):
    """Find invariant distribution of a Markov chain by unit eigenvector.
    NOTE: jax has no autodiff support for eig.
    """

    v, w = np.linalg.eig(T)

    # using sorted args instead of np.isclose is neccessary for jax-jitting
    args = np.argsort(v)
    unit_ev = w[:, args[-1]]

    return unit_ev.real / unit_ev.real.sum()


# @jax.jit
# def stationary_distribution_iterative(T, n=1000):
    # """Find invariant distribution of a Markov chain by brute force ('cause jax won't find the jacobian of eigenvectors)."""

    # a = np.ones(T.shape[0])/T.shape[0]
    # return np.linalg.matrix_power(T, n) @ a
