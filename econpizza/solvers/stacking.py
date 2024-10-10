# -*- coding: utf-8 -*-

import os
import jax
import time
import jax.numpy as jnp
from grgrjax import val_and_jacrev
from ..parser import d2jnp
from ..parser.build_generic_functions import build_aggr_het_agent_funcs, get_stst_derivatives
from ..parser.checks import check_if_compiled
from ..utilities.jacobian import get_stst_jacobian, get_jac_and_value_sliced
from ..utilities.newton import newton_for_jvp, newton_for_tridiag_jac, newton_jax_jit_wrapper


def write_cache(model, horizon, pars, stst):
    model['cache']['horizon'] = horizon
    model['cache']['pars'] = pars
    model['cache']['stst'] = stst
    return


def find_path_stacking(
    self,
    shock=None,
    init_state=None,
    init_dist=None,
    init_guess=None,
    pars=None,
    horizon=200,
    use_solid_solver=False,
    skip_jacobian=False,
    verbose=True,
    raise_errors=True,
    **newton_args
):
    """Find the expected trajectory given an initial state.

    Parameters
    ----------
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`
    init_state : array, optional
        initial state, defaults to the steady state values
    init_dist : array, optional
        initial distribution, defaults to the steady state distribution
    init_guess : array, optional
        initial guess on the sequence trajectory, defaults to the steady state
    pars : dict, optional
        alternative parameters. Warning: do only change those parameters that are invariant to the steady state.
    horizon : int, optional
        number of periods until the system is assumed to be back in the steady state. Defaults to ``200``
    use_solid_solver : bool, optional
        calculate the full jacobian and use a standard Newton method. Defaults to ``False``
    skip_jacobian : bool, optional
        whether to skip the calculation of the steady state sequence space Jacobian. If True, the last cached Jacobian will be used. Defaults to ``False``
    verbose : bool, optional
        degree of verbosity. ``0``/``False`` is silent. Defaults to ``False``
    raise_errors : bool, optional
        whether to raise errors as exceptions, or just inform about them. Defaults to ``True``
    newton_args : optional
        any additional arguments to be passed on to the Newton solver (see the documentations of the solvers)

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        Error flag. Returns `False` if the solver was successful, otherwise returns `True`
    """

    st = time.time()
    # only skip jacobian calculation if it exists
    skip_jacobian = skip_jacobian if self['cache'].get(
        'jac_factorized') else False

    # get variables
    stst = d2jnp(self["stst"])
    nvars = len(self["var_names"])
    pars = d2jnp(pars if pars is not None else self["pars"])
    shocks = self.get("shocks") or ()

    # get initial guess
    x0 = jnp.array(list(init_state)) if init_state is not None else stst
    init_dist = init_dist if init_dist is not None else self['steady_state'].get(
        'distributions')
    dist0 = jnp.array(init_dist if init_dist is not None else jnp.nan)
    x_stst = jnp.ones((horizon + 1, nvars)) * stst
    x_init = init_guess if init_guess is not None else x_stst.at[0].set(x0)

    # deal with shocks if any
    shock_series = jnp.zeros((horizon-1, len(shocks)))
    if shock is not None:
        try:
            shock_series = shock_series.at[0,
                                           shocks.index(shock[0])].set(shock[1])
        except ValueError:
            raise ValueError(f"Shock '{shock[0]}' is not defined.")

    if not self.get('distributions'):

        if not check_if_compiled(self, horizon, pars, stst) or not self['context'].get('jav_func'):
            # get transition function
            func_eqns = self['context']["func_eqns"]
            jav_func_eqns = val_and_jacrev(func_eqns, (0, 1, 2))
            jav_func_eqns_partial = jax.tree_util.Partial(
                jav_func_eqns, XSS=stst, pars=pars, distributions=[], decisions_outputs=[])
            self['context']['jav_func'] = jav_func_eqns_partial
            # mark as compiled
            write_cache(self, horizon, pars, stst)

        # actual newton iterations
        jav_func_eqns_partial = self['context']['jav_func']
        x_out, f, flag, mess = newton_for_tridiag_jac(
            jav_func_eqns_partial, nvars, horizon, x_init, shock_series, verbose, **newton_args)

    else:
        if not check_if_compiled(self, horizon, pars, stst) or not self['context'].get('jvp_func'):
            # get derivatives via AD and compile functions
            zero_shocks = jnp.zeros_like(shock_series).T
            build_aggr_het_agent_funcs(self, jnp.zeros_like(
                pars), nvars, stst, zero_shocks, horizon)

            if not use_solid_solver and not skip_jacobian:
                # get steady state partial jacobians
                derivatives = get_stst_derivatives(
                    self, nvars, pars, stst, x_stst, zero_shocks, horizon, verbose)
                # accumulate steady stat jacobian
                get_stst_jacobian(self, derivatives, horizon, nvars, verbose)
            # mark as compiled
            write_cache(self, horizon, pars, stst)

        # get jvp function and steady state jacobian
        jvp_partial = jax.tree_util.Partial(
            self['context']['jvp_func'], x0=x0, dist0=dist0, shocks=shock_series.T, pars=pars)
        if not use_solid_solver:
            jacobian = self['cache']['jac_factorized']
            # actual newton iterations
            x, f, flag, mess = newton_for_jvp(
                jvp_partial, jacobian, x_init, verbose, **newton_args)
        else:
            # define function returning value and jacobian calculated in slices
            value_and_jac_func = get_jac_and_value_sliced(
                (horizon-1)*nvars, jvp_partial, newton_args)
            x, f, flag, mess = newton_jax_jit_wrapper(
                value_and_jac_func, x_init[1:-1].flatten(), **newton_args)
        x_out = x_init.at[1:-1].set(x.reshape((horizon - 1, nvars)))

    # some informative print messages
    duration = time.time() - st
    result = 'done' if not flag else 'FAILED'
    mess = f"(find_path:) Stacking {result} ({duration:1.3f}s). " + mess
    if flag and raise_errors:
        raise Exception(mess)
    elif verbose:
        print(mess)

    return x_out, (flag, f)
