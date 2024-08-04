"""Tools accessible during runtime
"""

import jax
import jax.numpy as jnp
from grgrjax import jax_print
from .parser import load
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


def traverse_dict_and_set(d, path, value):
    dd = d
    for q in path:
        if isinstance(dd[q], dict):
            dd = dd[q]
        else:
            dd[q] = value


def traverse_dict_and_get(d, path):
    dd = d
    for q in path:
        dd = dd[q]
    return eval(str(dd))


def anneal_stst(model_dict, target_value, dict_path):

    current = target_value
    last_working = traverse_dict_and_get(model_dict, dict_path)
    sequence = current,

    while True:
        try:
            # set current guess to last value in sequence
            current = sequence[-1]
            print(f"(anneal_stst:) Trying '{dict_path[-1]}'={current}...")

            # update dict and try to solve
            traverse_dict_and_set(model_dict, dict_path, current)
            current_model = load(model_dict, raise_errors=True, verbose=len(sequence) == 1)
            res_stst = current_model.solve_stst(tol=1e-7, maxit=10, tol_forwards=1e-11, verbose=True)

            # update initial guesses if successful
            model_dict['steady_state']['init_guesses'].update(current_model['steady_state']['found_values'])
            if current == target_value:
                break
            last_working = current
            sequence = sequence[:-1]
            print(f"(anneal_stst:) Success with {last_working}! Guesses updated...\n    sequence is {sequence}")
        except Exception as e:
            print(e)
            sequence += (current/2 + last_working/2),

    # print final values
    print('(anneal_stst:) Success! Values are:\n')
    [print(f"        {k}: {v}") for k,v in current_model['steady_state']['found_values'].items()]
    return current_model

