#!/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import jax
import jax.numpy as jnp
from grgrlib.jaxed import newton_jax_jit, newton_jax_jittable


def find_path_shooting(
    model,
    x0=None,
    shock=None,
    T=30,
    init_path=None,
    max_horizon=200,
    max_loops=100,
    max_iter=None,
    tol=1e-5,
    root_options={},
    raise_error=False,
    verbose=True,
):
    """Find the expected trajectory given an initial state. A good strategy is to first set `tol` to a low value (e.g. 1e-3) and check for a good max_horizon. Then, set max_horizon to a reasonable value and let max_loops be high.

    NOTE: this is painfully slow since the migration to JAX.

    Parameters
    ----------
    model : dict
        model dict or PizzaModel instance
    x0 : array
        initial state
    shock : tuple, optional
        shock in period 0 as in `(shock_name_as_str, shock_size)`
    T : int, optional
        number of periods to simulate
    init_path : array, optional
        a first guess on the trajectory. Normally not necessary
    max_horizon : int, optional
        number of periods until the system is assumed to be back in the steady state. A good idea to set this corresponding to the respective problem. Note that a horizon too far away may cause the accumulation of numerical errors.
    max_loops : int, optional
        number of repetitions to iterate over the whole trajectory. Should eventually be high.
    max_iterations : int, optional
        number of iterations. Default is `max_horizon`. It should not be lower than that (and will raise an error). Normally it should not be higher, better use `max_loops` instead.
    tol : float, optional
        convergence criterion
    root_options : dict, optional
        dictionary with solver-specific options to be passed on to `scipy.optimize.root`
    verbose : bool, optional
        degree of verbosity. 0/`False` is silent

    Returns
    -------
    x_fin : array
        array of the trajectory
    x_lin : array or None
        array of the trajectory based on the linear model. Will return None if the linear model is unknown
    fin_flag : int
        error code
    """

    st = time.time()

    if max_iter is None:
        max_iter = max_horizon
    elif max_iter < max_horizon:
        Exception(
            "max_iter should be higher or equal max_horizon, but is %s and %s."
            % (max_iter, max_horizon)
        )

    stst = jnp.array(list(model["stst"].values()))
    nvars = len(model["variables"])
    shocks = model.get("shocks") or ()
    pars = jnp.array(list(model["parameters"].values()))
    func = jax.jit(model['context']["func_eqns"])

    if root_options:
        model["root_options"] = root_options

    # precision of root finding should be some magnitudes higher than of solver
    if "xtol" not in model["root_options"]:
        model["root_options"]["xtol"] = min(tol / max_horizon, 1e-8)

    x_fin = jnp.empty((T + 1, nvars))
    x_fin = x_fin.at[0].set(list(x0) if x0 is not None else stst)

    x = jnp.ones((T + max_horizon + 1, nvars)) * stst
    x = x.at[0].set(x_fin[0])

    if init_path is not None:
        x = x.at[1: len(init_path)].set(init_path[1:])

    tshock = jnp.zeros(len(shocks))

    fin_flag = jnp.zeros(5, dtype=bool)
    old_clock = time.time()

    msgs = (
        ", root finding did not converge",
        ", ftol not reached in root finding",
        ", contains NaNs",
        ", contains infs",
        ", max_iter reached",
    )

    @jax.jit
    def solve_current(pars, shock, XLag, XLastGuess, XPrime):
        """Solves for one period.
        """

        res = newton_jax_jittable(lambda x: func(
            XLag, x, XPrime, stst, shock, pars), XLastGuess)

        return res[0], res[2], res[3]

    try:
        for i in range(T):

            loop = 1
            cnt = 2
            flag = jnp.zeros_like(fin_flag)

            while True:

                x_old = x[1].copy()
                imax = min(cnt, max_horizon)

                flag_loc = jnp.zeros(2, dtype=bool)

                for t in range(imax):

                    if not t and not i and shock is not None:
                        tshock.at[shocks.index(shock[0])].set(shock[1])
                    else:
                        tshock.at[:].set(0)

                    x_new, flag_root, flag_ftol = solve_current(
                        pars, tshock, x[t], x[t + 1], x[t + 2])

                    x = x.at[t + 1].set(x_new)

                    flag_loc = flag_loc.at[0].set(flag_loc[0] or not flag_root)
                    flag_loc = flag_loc.at[1].set(
                        flag_loc[2] or (not flag_ftol and flag_root))

                flag = flag.at[2].set(flag[2] or jnp.any(jnp.isnan(x)))
                flag = flag.at[3].set(flag[3] or jnp.any(jnp.isinf(x)))

                if cnt == max_iter:
                    if loop < max_loops:
                        loop += 1
                        cnt = 2
                    else:
                        flag = flag.at[4].set(flag[4] or True)

                err = jnp.abs(x_old - x[1]).max()

                clock = time.time()
                if verbose and clock - old_clock > 0.5:
                    old_clock = clock
                    print(
                        "   Period{:>4d} | loop{:>5d} | iter.{:>5d} | flag{:>3d} | error: {:>1.8e}".format(
                            i, loop, cnt, 2 ** jnp.arange(5) @ fin_flag, err
                        )
                    )

                if (err < tol and cnt > 2) or flag.any():
                    flag = flag.at[:2].set(flag[2] or flag_loc)
                    if raise_error and flag.any():
                        mess = [i * bool(j) for i, j in zip(msgs, flag)]
                        raise Exception("Aborting%s" % "".join(mess))
                    fin_flag |= flag
                    break

                cnt += 1

            x_fin = x_fin.at[i + 1].set(x[1])
            x = x[1:].copy()

    except Exception as error:
        try:
            raise type(error)(
                str(error)
                + " (raised in period %s during loop %s for forecast %s steps ahead)"
                % (i, loop, t)
            ).with_traceback(sys.exc_info()[2])
        except AttributeError:
            raise error

    fin_flag = fin_flag.at[1].set(fin_flag[1] and not fin_flag[0])
    mess = [i * bool(j) for i, j in zip(msgs, fin_flag)]
    fin_flag = 2 ** jnp.arange(5) @ fin_flag

    if verbose:
        duration = jnp.round(time.time() - st, 3)
        print("(find_path:) Pizza done after %s seconds%s." %
              (duration, "".join(mess)))

    return x_fin, fin_flag
