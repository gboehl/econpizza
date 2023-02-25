
"""Grids and Markov chains"""

import jax
import jax.numpy as jnp
from .dists import stationary_distribution


def log_grid(amax, n, amin=0):
    """Create grid between amin and amax that is equidistant in logs."""
    pivot = jnp.abs(amin) + 0.25
    a_grid = jnp.geomspace(amin + pivot, amax + pivot, n) - pivot
    a_grid = a_grid.at[0].set(amin)  # make sure *exactly* equal to amin
    return a_grid


def mean(x, pi):
    """Mean of discretized random variable with support x and probability mass function pi."""
    return jnp.sum(pi * x)


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return jnp.sum(pi * (x - jnp.sum(pi * x)) ** 2)


def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""

    # Explicitly typecast N as an integer, since when the grid constructor functions
    # (e.g. the function that makes a_grid) are implemented as blocks, they interpret the integer-valued calibration
    # as a float.
    N = int(N)

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = jnp.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (jnp.zeros((n, n)) for _ in range(4))
        P1 = P1.at[:-1, :-1].set(p * Pi)
        P2 = P2.at[:-1, 1:].set((1 - p) * Pi)
        P3 = P3.at[1:, :-1].set((1 - p) * Pi)
        P4 = P4.at[1:, 1:].set(p * Pi)
        Pi = P1 + P2 + P3 + P4
        Pi = Pi.at[1:-1].divide(2)

    # invariant distribution and scaling
    pi = stationary_distribution(jnp.array(Pi.T))
    s = jnp.linspace(-1, 1, N)
    s *= (sigma / jnp.sqrt(variance(s, pi)))
    y = jnp.exp(s) / jnp.sum(pi * jnp.exp(s))

    return y, pi, Pi


def rouwenhorst_grid_from_stationary(sigma, stationary_distribution):
    s = jnp.linspace(-1, 1, len(stationary_distribution))
    s *= (sigma / jnp.sqrt(variance(s, stationary_distribution)))
    return jnp.exp(s) / jnp.sum(stationary_distribution * jnp.exp(s))


def create_grids(distributions, context):
    """Get the strings of functions that define the grids.
    """

    if not distributions:
        return

    grid_strings = ()

    # create strings of the function that define the grids
    for dist_name, dist in distributions.items():
        for grid_name, g in dist.items():

            # shortcuts
            g['type'] = 'endogenous_log' if g['type'] == 'endogenous' else g['type']
            g['type'] = 'exogenous_rouwenhorst' if g['type'] == 'exogenous' else g['type']

            if 'exogenous' in g['type'] and 'transition_name' not in g:
                g['transition_name'] = grid_name + '_transition'

            if 'endogenous' in g['type'] and 'grid_name' not in g:
                g['grid_name'] = grid_name + '_grid'

            if g['type'] == 'exogenous_rouwenhorst':
                grid_strings += f"{', '.join(v for v in g['grid_variables'])} = grids.markov_rouwenhorst(rho={g['rho']}, sigma={g['sigma']}, N={g['n']})",

            if g['type'] == 'endogenous_log':
                grid_strings += f"{g['grid_name']} = grids.log_grid(amin={g['min']}, amax={g['max']}, n={g['n']})",

    # execute all of them
    for grid_str in grid_strings:
        exec(grid_str, context)

    return
