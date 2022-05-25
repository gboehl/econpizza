#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def interpolate_coord_robust_vector(x, xq):
    """Does interpolate_coord_robust where xq must be a vector, more general function is wrapper"""

    xqi = jnp.searchsorted(x, xq, side='right') - 1
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

    if check_increasing and jnp.any(x[:-1] >= x[1:]):
        raise ValueError(
            'Data input to interpolate_coord_robust must be strictly increasing')

    if xq.ndim == 1:
        return interpolate_coord_robust_vector(x, xq)
    else:
        i, pi = interpolate_coord_robust_vector(x, xq.ravel())
        return i.reshape(xq.shape), pi.reshape(xq.shape)


@partial(jnp.vectorize, signature='(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y):
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """

    nx = x.shape[0]

    # xi = jnp.minimum(jnp.searchsorted(x, xq, side='right'), nx-2)
    xi = jnp.searchsorted(x, xq, side='right') - 1
    x_low = x[xi]
    x_high = x[xi + 1]

    xqpi_cur = (x_high - xq) / (x_high - x_low)
    yq = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

    return yq
