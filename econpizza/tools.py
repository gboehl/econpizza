"""Tools accessible during runtime
"""

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print as jax_print
from .utilities.interp import interpolate


def percentile(x, dist, share):

    # reshape inputs flattend for each period t
    horizon = x.shape[-1]
    x_flat = x.reshape(-1, horizon)
    dist_flat = dist.reshape(-1, horizon)

    # distribution sorted according to x
    dist_sorted = jnp.take_along_axis(
        dist_flat, jnp.argsort(x_flat, axis=0), axis=0)
    # cummulative sums
    dist_cumsum = jnp.cumsum(dist_sorted, axis=0)
    x_cumsum = jnp.cumsum(x_flat*dist_flat, axis=0)
    px = interpolate(dist_cumsum.T, (share,),
                     x_cumsum.T).flatten()/x_cumsum[-1]

    return px
