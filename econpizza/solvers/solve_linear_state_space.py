#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from grgrlib import klein, speed_kills


def solve_linear_state_space(
    model,
    raise_error=True,
    check_contraction=False,
    lti_max_iter=1000,
    verbose=True,
):
    """Does half-way SGU, solves the model using linear time iteration and uses Klein's method to check for determinancy if the solution fails"""

    if model.get('distributions'):
        raise Exception(
            "A linear state-space solution for models with distributions is not implemented.")

    evars = model["variables"]
    func = model['context']["func_eqns"]
    par = jnp.array(list(model["parameters"].values()))
    shocks = model.get("shocks") or ()
    stst = jnp.array(list(model["stst"].values()))
    nshc = len(shocks)
    nsts = len(stst)

    xmult = stst.copy()
    xmult = xmult.at[jnp.isclose(stst, 0)].set(1)

    zshock = jnp.zeros(len(shocks))
    fx = func(stst, stst, stst, stst, jnp.zeros(nshc), par)

    AA = jax.jacfwd(lambda err: func(stst, stst, err,
                    stst, zshock, par))(stst) * xmult
    BB = jax.jacfwd(lambda err: func(stst, err, stst,
                    stst, zshock, par))(stst) * xmult
    CC = jax.jacfwd(lambda err: func(err, stst, stst,
                    stst, zshock, par))(stst) * xmult
    DD = jax.jacfwd(lambda err: func(stst, stst, stst, stst, err, par))(zshock)
    DD = DD.reshape((nsts, len(shocks)))

    A = jnp.pad(AA, ((0, nshc), (0, nshc)))
    B = jax.scipy.linalg.block_diag(BB, jnp.eye(nshc))
    C = jnp.block([[CC, DD], [jnp.zeros((nshc, A.shape[1]))]])

    model["ABC"] = A, B, C

    I = jnp.eye(nsts + nshc)
    Z = jnp.zeros_like(I)
    P = jnp.block([[B, A], [I, Z]])
    M = jnp.block([[C, Z], [Z, -I]])

    mess = ""
    success = True

    try:
        try:
            lam = -speed_kills(
                P, M, nsts + nshc, max_iter=lti_max_iter, verbose=verbose - 1
            )[1]

        except:
            _, lam = klein(P, M, nsts + nshc, verbose=verbose - 1)

        model["lin_pol"] = -lam[:nsts, :nsts], -lam[:nsts, nsts:]
        mess = "All eigenvalues are good"

    except Exception as error:
        success = False
        if raise_error:
            raise error
        else:
            mess = str(error).strip()
            if mess[-1] == ".":
                mess = mess[:-1]

    if check_contraction:
        A = jnp.linalg.inv(BB) @ AA
        B = jnp.linalg.inv(BB) @ CC

        Aev = jnp.abs(jnp.linalg.eig(A)[0])
        Aev_err = Aev > 1
        Bev = jnp.abs(jnp.linalg.eig(B)[0])
        Bev_err = Bev > 1
        flag += Aev_err.any() or Bev_err.any()
        mess += ", but " if success else ""
        if Aev_err.any():
            mess += "%s forward looking EV%s larger than unity (%s)" % (
                Aev_err.sum(),
                "s are" if Aev_err.sum() > 1 else " is",
                *Aev[Aev_err],
            )
        if Aev_err.any() and Bev_err.any():
            mess += " and "
        if Bev_err.any():
            mess += "%s backward looking EV%s larger than unity (%s)" % (
                Bev_err.sum(),
                "s are" if Bev_err.sum() > 1 else " is",
                *Bev[Bev_err],
            )

    if mess and verbose:
        print(f"(solve_linear:) {mess}{'' if mess[-1] in '.?!' else '.'}")

    return model["ABC"]


def find_path_linear_state_space(model, x0=None, shock=None, T=30, verbose=True):
    """Solves the expected trajectory given the linear model.

    Parameters
    ----------
    model : dict
        model dict or PizzaModel instance
    x0 : array
        initial state
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

    stst = jnp.array(list(model["stst"].values()))
    sel = jnp.isclose(stst, 0)

    x0 = jnp.array(list(x0)) if x0 is not None else stst
    x0 = x0.at[~sel].set(x0[~sel] / stst[~sel] - 1)

    shocks = model.get("shocks") or ()
    tshock = jnp.zeros(len(shocks))
    if shock is not None:
        tshock[shocks.index(shock[0])] = shock[1]

    x_lin = jnp.empty((T+1, len(stst)))
    x_lin = x_lin.at[0].set(x0)

    for t in range(T):
        x_lin = x_lin.at[t + 1].set(model["lin_pol"][0] @ x_lin[t])

        if not t:
            x_lin = x_lin.at[t + 1].add(model["lin_pol"][1] @ tshock)

    x_lin = x_lin.at[:, ~sel].set(((1 + x_lin) * stst)[:, ~sel])

    return x_lin, True
