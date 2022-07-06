#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp


def tmat_from_exog(probs, D):

    nZ, nX = D.shape
    tmat = jnp.zeros((nX*nZ, nX*nZ))

    i = jnp.arange(nZ)*nX
    xi = jnp.arange(nX)[:, None]
    zi = jnp.arange(nZ)[:, None, None]
    tmat = tmat.at[xi+zi*nX, xi +
                   i].set(probs.broadcast_in_dim((nZ, nX, nZ), (0, 2)))

    return tmat


def tmat_from_endo(x_i, probs):

    nZ, nX = x_i.shape
    tmat = jnp.zeros((nX*nZ, nX*nZ))

    ix = jnp.arange(nX*nZ)
    j = jnp.arange(nZ).repeat(nX)*nX
    i = x_i.ravel()
    pi = probs.ravel()

    tmat = tmat.at[i+j, ix].add(pi)
    tmat = tmat.at[i+1+j, ix].add(1-pi)

    return tmat


def expect_transition(Pi, X):
    """If Pi is a matrix, multiply Pi times the ith dimension of X and return"""

    shape = X.shape
    # iterate forward using Pi
    X = Pi @ X.reshape((shape[0], -1))
    # reverse steps
    X = X.reshape((Pi.shape[0], *shape[1:]))

    return X


def forward_policy_1d(D, x_i, x_pi):

    nZ, _ = D.shape
    Dnew = jnp.zeros_like(D)

    j = jnp.arange(nZ)[:, None]

    Dnew = Dnew.at[j, x_i].add(D * x_pi)
    Dnew = Dnew.at[j, x_i+1].add(D * (1 - x_pi))

    return Dnew


def stationary_distribution_forward_policy_1d(endog_inds, endog_probs, exog_probs, tol=1e-10, maxit=1000):

    dist = jnp.ones_like(endog_inds, dtype=jnp.float64)
    dist /= dist.sum()

    def cond_func(cont):
        dist, dist_old, cnt = cont
        cond0 = jnp.abs(dist-dist_old).max() > tol
        cond1 = cnt < maxit
        return cond0 & cond1

    def body_func(cont):
        dist, _, cnt = cont
        return exog_probs.T @ forward_policy_1d(dist, endog_inds, endog_probs), dist, cnt + 1

    dist, _, cnt = jax.lax.while_loop(cond_func, body_func, (dist, dist+1, 0))
    return dist, cnt


def forward_policy_2d(D, x_i, y_i, x_pi, y_pi):

    nZ, _, _ = D.shape
    Dnew = jnp.zeros_like(D)

    j = jnp.arange(nZ)[:, None, None]

    Dnew = Dnew.at[j, x_i, y_i].add(y_pi * x_pi * D)
    Dnew = Dnew.at[j, x_i+1, y_i].add(y_pi * (1 - x_pi) * D)
    Dnew = Dnew.at[j, x_i, y_i+1].add((1 - y_pi) * x_pi * D)
    Dnew = Dnew.at[j, x_i+1, y_i+1].add((1 - y_pi) * (1 - x_pi) * D)

    return Dnew


def stationary_distribution_forward_policy_2d(endog_inds0, endog_inds1, endog_probs0, endog_probs1, exog_probs, tol=1e-10, maxit=1000):
    # TODO: can be merged with stationary_distribution_forward_policy_1d

    dist = jnp.ones_like(endog_inds0, dtype=jnp.float64)
    dist /= dist.sum()

    def cond_func(cont):
        dist, dist_old, cnt = cont
        cond0 = jnp.abs(dist-dist_old).max() > tol
        cond1 = cnt < maxit
        return cond0 & cond1

    def body_func(cont):
        dist, _, cnt = cont
        pre_exo_dist = forward_policy_2d(
            dist, endog_inds0, endog_inds1, endog_probs0, endog_probs1)
        new_dist = expect_transition(exog_probs.T, pre_exo_dist)
        return new_dist, dist, cnt + 1

    dist, _, cnt = jax.lax.while_loop(cond_func, body_func, (dist, dist+1, 0))
    return dist, cnt


def stationary_distribution(T):
    """Find invariant distribution of a Markov chain by unit eigenvector.
    NOTE: jax has no autodiff support for eig. (there is a version with custom_jvp in grgrwip)
    """

    v, w = jnp.linalg.eig(T)

    # using sorted args instead of np.isclose is neccessary for jax-jitting
    args = jnp.argsort(v)
    unit_ev = w[:, args[-1]]

    return unit_ev.real / unit_ev.real.sum()
