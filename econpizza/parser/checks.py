"""Checking functions for parsing.py.
"""

import sys
import jax.numpy as jnp
from .build_generic_functions import func_pre_stst
import re


def _strip_comments(code):
    code = str(code)
    return re.sub(r'(?m)^ *#.*\n?', '', code)


def check_if_defined(evars, eqns, decisions, skipped_vars):
    """Check if all variables are defined in period t.
    """
    skipped_vars = [] if skipped_vars is None else skipped_vars
    calls = _strip_comments(
        decisions['calls']) if decisions is not None else ''

    for v in evars:
        v_in_eqns = [
            v in e.replace(v + "SS", "").replace(v + "Lag",
                                                 "").replace(v + "Prime", "")
            for e in eqns
        ]
        v_in_calls = v in calls.replace(
            v + "SS", "").replace(v + "Lag", "").replace(v + "Prime", "")
        if not any(v_in_eqns) and not v_in_calls and not v in skipped_vars:
            raise Exception(f"Variable `{v}` is not defined for time t.")
    return


def check_dublicates(evars):
    """Check if there are dublicates
    """
    evars = [*evars] if isinstance(evars, dict) else evars
    dubs = [x for i, x in enumerate(evars) if x in evars[:i]]
    dubmess = (
        " The variables list contains dublicate(s): %s" % ", ".join(
            dubs) if dubs else ""
    )
    if dubs:
        print("(parse:) Warning%s" % dubmess)
    return


def check_determinancy(evars, eqns):
    """Check if the numbers of eqns/vars match.
    """
    evars = [*evars] if isinstance(evars, dict) else evars
    sorted_evars = evars[:] = sorted(list(set(evars)), key=str.lower)
    if len(sorted_evars) != len(eqns):
        raise Exception(
            f"Model has {len(sorted_evars)} variables but {len(eqns)} equations."
        )
    return sorted_evars


def check_initial_values(model, shocks, init_guesses, fixed_values, init_wf, pre_stst_mapping):

    from . import d2jnp

    # run func_pre_stst to translate init values into vars & pars
    transform_back = model['options'].get('transform_back') or (lambda x: x)
    init_vals, par = func_pre_stst(transform_back(d2jnp(init_guesses)), transform_back(d2jnp(fixed_values)), pre_stst_mapping)

    # collect some information needed later
    model['context']['init_run'] = {}

    mess = ''
    if model.get('decisions'):
        # make a test backward and forward run
        _, decisions_output_init = model['context']['func_backw'](
            init_vals, init_vals, init_vals, init_vals, par, init_wf, jnp.zeros(len(shocks)))
        dists_init, _ = model['context']['func_forw_stst'](
            decisions_output_init, 1e-8, 1)

        if any(jnp.isnan(doi).any() for doi in decisions_output_init):
            mess = 'Outputs of decision stage contains `NaN`s'
        elif any(jnp.isinf(doi).any() for doi in decisions_output_init):
            mess = 'Outputs of decision stage are not finite'
        elif jnp.isnan(dists_init).any():
            mess = 'Distribution contains `NaN`s'
        elif jnp.isinf(dists_init).any():
            mess = 'Distribution is not finite'
    else:
        decisions_output_init = dists_init = []

    model['context']['init_run']['decisions_output'] = decisions_output_init
    model['context']['init_run']['dists'] = dists_init

    # final test of main function
    init_vals = init_vals[..., None]
    test = model['context']['func_eqns'](init_vals, init_vals, init_vals, init_vals, par, jnp.zeros(
        len(shocks)), jnp.array(dists_init)[..., None], (doi[..., None] for doi in decisions_output_init))

    if mess:
        pass
    elif jnp.isnan(test).any():
        mess += 'Output of final stage contains `NaN`s'
    elif jnp.isinf(test).any():
        mess += 'Output of final stage is not finite'

    if mess:
        raise Exception(mess + ' for initial values.')
    return


def check_shapes(distributions, init_decisions, dist_names):
    decisions_shape = init_decisions.shape
    # so far only for one distribution
    dist_shape = tuple([d.get('n')
                       for d in distributions[dist_names[0]].values()])
    check = [(ds == decisions_shape[-len(dist_shape):][i] or ds is None)
             for i, ds in enumerate(dist_shape)]
    if not all(check):
        raise Exception(
            f"Initial decisions and the distribution have different shapes in last dimensions: {decisions_shape}, {dist_shape}")
    return


def check_if_compiled(model, horizon, pars, stst):
    try:
        assert model['cache']['horizon'] == horizon
        assert jnp.allclose(model['cache']['stst'], stst)
        return jnp.allclose(model['cache']['pars'], pars)
    except:
        return False

def check_for_lags(calls, evars):
   for v in evars:
       if v+'Lag' in calls: 
           raise Exception(f"`{v}Lag` in decisions calls detected. For efficiency reasons, the use of lagged values is not supported here. This can be circumvented by defining an auxilliary variable, e.g. `{v}_lagged = {v}Lag`.")


