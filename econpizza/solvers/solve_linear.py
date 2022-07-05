#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import jax.numpy as jnp
import numpy as np
import scipy.sparse as ssp


def get_stst_jacobian(model, horizon):
    """Get the steady state Jacobian. Only meant to be used internally.
    """

    pars = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = jnp.array(list(model['stst'].values()))
    nvars = len(model["variables"])

    if model.get('distributions'):
        vfSS = model['steady_state']['decisions']
        distSS = jnp.array(model['steady_state']['distributions'])
        decisions_outputSS = jnp.array(
            model['steady_state']['decisions_output'])

    func_eqns = model['context']["func_eqns"]
    func_backw = model['context'].get('func_backw')
    func_dist = model['context'].get('func_dist')
    zshock = jnp.zeros(len(shocks))

    # TODO: getting the partial Jacobians should be a subfunction
    @jax.jit
    def func_eqns_ssj(x, distributions, decisions_output):
        distributions = distributions.reshape(*distSS.shape, 1)
        decisions_output = decisions_output.reshape(
            *decisions_outputSS.shape, 1)
        return func_eqns(x[:nvars, jnp.newaxis], x[nvars:-nvars, jnp.newaxis], x[-nvars:, jnp.newaxis], stst, zshock, pars, distributions, decisions_output)

    @jax.jit
    def func_backw_ssj(x, vf):
        vf = vf.reshape(*vfSS.shape)
        vf, decisions_output = func_backw(
            x[:nvars, jnp.newaxis], x[nvars:-nvars, jnp.newaxis], x[-nvars:, jnp.newaxis], stst, vf, zshock, pars)[:2]
        return vf.ravel(), decisions_output.ravel()

    @jax.jit
    def func_dist_ssj(distributions, decisions_output):
        distributions = distributions.reshape(*distSS.shape)
        decisions_output = decisions_output.reshape(*decisions_outputSS.shape)
        return func_dist(distributions, decisions_output)[0].ravel()

    if model.get('distributions'):
        # TODO: could still save some time by only cracking those jacobians that we care about
        jac_dist = jax.jacfwd(func_dist_ssj, argnums=(0, 1))(
            distSS.ravel(), decisions_outputSS.ravel())
        jac_f = jax.jacrev(func_eqns_ssj, argnums=(0, 1, 2))(
            jnp.hstack([stst]*3), distSS.ravel(), decisions_outputSS.ravel())
        jac_vf, jac_do = jax.jacfwd(func_backw_ssj, argnums=(0, 1))(
            jnp.hstack([stst]*3), vfSS.ravel())

        # let all this be sparse
        jac_dist2dist = ssp.csr_array(jac_dist[0])
        jac_dist2do = ssp.csc_array(jac_dist[1])
        jac_f2x = ssp.csc_array(jac_f[0])
        jac_f2dist = ssp.csr_array(jac_f[1])
        jac_f2do = ssp.csc_array(jac_f[2])
        (jac_vf2x, jac_vf2vf), (jac_do2x, jac_do2vf) = [
            ssp.csc_array(j) for j in jac_vf], [ssp.csc_array(j) for j in jac_do]

    else:
        jac_f2x = ssp.csc_array(jax.jacrev(lambda x: func_eqns(
            x[:nvars, jnp.newaxis], x[nvars:-nvars, jnp.newaxis], x[-nvars:, jnp.newaxis], stst, zshock, pars))(jnp.hstack([stst]*3)))

    # TODO: calculating the complete Jacobian should be a subfunction
    jac = ssp.lil_array(((horizon-1)*nvars, (horizon+1)*nvars))
    if model.get('distributions'):
        jac_f2x = jac_f2x + jac_f2do.dot(jac_do2x)

    dummy = ssp.lil_array((nvars, (horizon*2+1)*nvars))

    # dummy is centered at t=0 in nvars*horizon => (horizon-1) is t=-1 block
    dummy[:, (horizon-1)*nvars:(horizon+2)*nvars] = jac_f2x

    if model.get('distributions'):
        runner = jac_f2do.dot(jac_do2vf)
        for i in range(1, horizon):
            dummy[:, (horizon+i-1)*nvars:(horizon+i+2)*nvars] = dummy[:,
                                                                      (horizon+i-1)*nvars:(horizon+i+2)*nvars] + runner.dot(jac_vf2x)
            runner = runner.dot(jac_vf2vf)

        runner = jac_f2dist
        for i in range(1, horizon):
            dummy[:, (horizon-i-1)*nvars:(horizon-i+2)*nvars] = dummy[:, (horizon-i-1)
                                                                      * nvars:(horizon-i+2)*nvars] + runner.dot(jac_dist2do).dot(jac_do2x)
            runner = runner.dot(jac_dist2dist)

    for i in range(1, horizon):
        jac[(i-1)*nvars:i*nvars] = dummy[:,
                                         (horizon-i)*nvars:(2*horizon-i+1)*nvars]

    if not model.get('distributions'):
        return jac.tocsr()

    jac_via_dists = ssp.lil_array(((horizon-1)*nvars, (horizon+1)*nvars))
    W = ssp.lil_array((distSS.size, (horizon+1)*nvars))

    runner = jac_dist2do.dot(jac_do2vf)

    for i in range(horizon-1):
        W[:, nvars*i:nvars*(i+3)] = W[:, nvars*i:nvars*(i+3)
                                      ].tocsc() + runner.dot(jac_vf2x)
        runner = runner.dot(jac_vf2vf)

    for i in range(horizon-1):
        jac_via_dists[nvars*i:nvars*(i+1)] = jac_f2dist.dot(W)
        W = jac_dist2dist.dot(W)

    for i in range(horizon-2):
        jac_via_dists[nvars*(i+1):nvars*(i+2), nvars:] = jac_via_dists[nvars*(
            i+1):nvars*(i+2), nvars:] + jac_via_dists[nvars*i:nvars*(i+1), :-nvars]

    return jac.tocsr() + jac_via_dists.tocsr()


def find_path_linear(model, x0, shocks=None, horizon=300, verbose=True):
    """Find the linear expected trajectory given an initial state.

    Parameters
    ----------
    model : dict
        model dict or PizzaModel instance
    x0 : array
        initial state
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`. NOTE: Not (yet) implemented.
    horizon : int, optional
        number of periods until the system is assumed to be back in the steady state. A good idea to set this corresponding to the respective problem. A too large value may be computationally expensive. A too small value may generate inaccurate results
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent

    Returns
    -------
    x : array
        array of the trajectory
    flag : bool
        for consistency. Always returns `True`
    """

    if shocks is not None:
        raise NotImplementedError(
            "Shocks are not (yet) implemented for the linear solution.")

    st = time.time()
    stst = jnp.array(list(model['stst'].values()))
    nvars = len(model["variables"])

    if model['stst_jacobian'] is None:
        stst_jacobian = get_stst_jacobian(model, horizon)
        model['stst_jacobian'] = stst_jacobian
        mess = 'Steady state Jacobian calculated.'
    else:
        stst_jacobian = model['stst_jacobian']
        mess = 'Steady state Jacobian loaded.'

    x0 = jnp.array(list(x0)) - stst
    x = - \
        ssp.linalg.spsolve(
            stst_jacobian[:, nvars:-nvars], stst_jacobian[:, :nvars] @ x0)
    x = jnp.hstack((x0, x)).reshape(-1, nvars) + stst

    if verbose:
        duration = time.time() - st
        print(
            f"(find_path_linear:) {mess} Linear solution done after {duration:1.3f} seconds.")

    return x, True
