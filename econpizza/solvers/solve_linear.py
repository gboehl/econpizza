# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
from .stacking import write_cache
from ..utilities.jacobian import get_stst_jacobian
from ..parser import d2jnp
from ..parser.checks import check_if_compiled
from ..parser.build_generic_functions import build_aggr_het_agent_funcs, get_stst_derivatives


def find_path_linear(self, shock=None, init_state=None, pars=None, horizon=200, verbose=True):
    """Find the linear expected trajectory given an initial state.

    Parameters
    ----------
    init_state : array
        initial state
    pars : dict, optional
        alternative parameters. Warning: do only change those parameters that are invariant to the steady state.
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`. NOTE: Not (yet) implemented.
    horizon : int, optional
        number of periods to simulate
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        for consistency. Always returns `True`
    """

    if shock is not None:
        raise NotImplementedError(
            "Shocks are not (yet) implemented for the linear solution.")

    if not self.get('distributions'):
        raise NotImplementedError(
            "Models without heterogeneous agents are not (yet) implemented for the linear solution.")

    st = time.time()

    # get model variables
    stst = d2jnp(self['stst'])
    nvars = len(self["var_names"])
    pars = d2jnp((pars if pars is not None else self["pars"]))
    shocks = self.get("shocks") or ()
    distSS = jnp.array(self['steady_state'].get('distributions'))
    doSS = [jnp.array(d)[..., None] for d in list(self['steady_state']['decisions'].values())]
    x_stst = jnp.ones((horizon + 1, nvars)) * stst

    # deal with shocks
    zero_shocks = jnp.zeros((horizon-1, len(shocks))).T

    x0 = jnp.array(list(init_state)) if init_state is not None else stst

    if not check_if_compiled(self, horizon, pars, stst):
        # get derivatives via AD and compile functions
        build_aggr_het_agent_funcs(self, jnp.zeros_like(
            pars), nvars, stst, zero_shocks, horizon)
        derivatives = get_stst_derivatives(
            self, nvars, pars, stst, x_stst, zero_shocks, horizon, verbose)

        # accumulate steady state jacobian
        get_stst_jacobian(self, derivatives, horizon, nvars, verbose)
        jac_f2xLag = derivatives[0][0]
        self['cache']['jacobian_f2x0'] = jnp.zeros(((horizon-1)*nvars, nvars)).at[:nvars].set(jac_f2xLag[...,0])
        write_cache(self, horizon, pars, stst)

    # get jacobians
    jacobian = self['cache']['jac_factorized']
    jacobian_f2x0 = self['cache']['jacobian_f2x0']
    combined_sweep = self['context']['combined_sweep']

    x = -jax._src.lax.linalg.lu_solve(*jacobian[0], jacobian_f2x0 @ (x0-stst), 0)[jacobian[1]]
    x = jnp.vstack((x0,x.reshape(-1, nvars) + stst, stst))

    if verbose:
        duration = time.time() - st
        print(
            f"(find_path_linear:) Linear solution done ({duration:1.3f}s).")

    return x, True
