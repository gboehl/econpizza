"""Tools accessible during runtime
"""

import jax
import jax.numpy as jnp
from grgrjax import jax_print
from .parser import load
from .utilities.interp import interpolate
from .utilities.grids import markov_rouwenhorst, rouwenhorst_grid_from_stationary


def percentile(x, dist, share, normalize=True):
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
    res = interpolate(dist_cumsum.T, (share,), x_cumsum.T).flatten()
    if normalize:
        return res/x_cumsum[-1]
    return res


def traverse_dict(d, path, value=None):
    dd = d
    for q in path:
        if isinstance(dd[q], dict) or value is None:
            dd = dd[q]
        else:
            dd[q] = value
    if value is None:
        return eval(str(dd))


def anneal_stst(mdict, dict_path, target_value, max_sequence=10, **kwargs):
    """Anneal steady state by iteratively updating initial guesses. Assumes that the current model dictionary provides valid results.

    Parameters
    ----------
    mdict : dict
        the parsed yaml as a dictionary
    target_value : float
        desired value of the target
    dict_path : strng or tuple of strings
        recursive dictionary keywords which are the path to the target value (e.g. `('steady_state', 'fixed_values')`.
    max_sequence : int
        maximum lenght of the sequence before aborting
    **kwargs : optional
        arguments passed on to `find_stst`

    Returns
    -------
    current_model: PizzaModel instance
        the target model
    """

    current = target_value
    last_working = traverse_dict(mdict, dict_path)
    sequence = current,

    while True:
        try:
            # set current guess to last value in sequence
            current = sequence[-1]
            print(f"(anneal_stst:) {len(sequence)} value(s) in queue. Trying {dict_path[-1]}={current}...")

            # update dict and try to solve
            traverse_dict(mdict, dict_path, current)
            current_model = load(mdict, raise_errors=True, verbose=False)
            res_stst = current_model.solve_stst(verbose=True, **kwargs)

            # update initial guesses if successful
            mdict['steady_state']['init_guesses'].update(current_model['steady_state']['found_values'])
            if current == target_value:
                break
            last_working = current
            sequence = sequence[:-1]
            print(f"(anneal_stst:) Success with {last_working}! Guesses updated...\n    queue is {sequence}")
        except Exception as e:
            if len(sequence) == max_sequence:
                raise Exception(f"(anneal_stst:) FAILED because lenght of sequence exceeds {max_sequence}. Best guess was {dict_path[-1]}={last_working}")
            sequence += (current/2 + last_working/2),
            print(str(e) + 'Adding value to queue.')

    # print final values
    print('(anneal_stst:) Success! Values are:\n')
    [print(f"        {k}: {v}") for k,v in current_model['steady_state']['found_values'].items()]
    return current_model, mdict

