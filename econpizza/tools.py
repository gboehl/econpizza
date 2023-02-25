"""Tools accessible during runtime
"""

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print as jax_print
from .utilities.interp import interpolate
from .utilities.grids import markov_rouwenhorst, rouwenhorst_grid_from_stationary


def percentile(x, dist, share):
    """percentiles function
    """
    # reshape inputs flattend for each period t
    horizon = x.shape[-1]
    x_flat = x.reshape(-1, horizon)
    dist_flat = dist.reshape(-1, horizon)

    # distribution sorted according to x
    sorted_args = jnp.argsort(x_flat, axis=0)
    dist_sorted = jnp.take_along_axis(dist_flat, sorted_args, axis=0)
    x_sorted = jnp.take_along_axis(x_flat, sorted_args, axis=0)

    # cummulative sums
    dist_cumsum = jnp.cumsum(dist_sorted, axis=0)
    x_cumsum = jnp.cumsum(x_sorted*dist_sorted, axis=0)

    # interpolate
    return interpolate(dist_cumsum.T, (share,), x_cumsum.T).flatten()/x_cumsum[-1]
