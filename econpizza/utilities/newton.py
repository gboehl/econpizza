"""Newton heavy lifting functions and helpers
"""

import jax
import time
import jax.numpy as jnp
from grgrlib.jaxed import callback_func, amax


def iteration_step(dummy, carry):
    (y, dampening), (x, f, jvp_func, jacobian, factor) = carry
    _, v = jvp_func(x, y)
    v = jax.scipy.linalg.lu_solve(jacobian, v)
    dampening = jnp.minimum(dampening, factor*(y.T@y)/(v.T@y))
    y += dampening*(f-v)
    return (y, dampening), (x, f, jvp_func, jacobian, factor)


def while_body_jvp(carry):
    (x, _, _, cnt), (jvp_func, jacobian, maxit,
                     nsteps, tol, factor, verbose) = carry
    # first iteration
    f, _ = jvp_func(x, jnp.zeros_like(x))
    f = jax.scipy.linalg.lu_solve(jacobian, f)
    # other iterations
    (y, dampening), _ = jax.lax.fori_loop(0, nsteps,
                                          iteration_step, ((f, 1.), (x, f, jvp_func, jacobian, factor)))
    return (x-y, amax(f), dampening, cnt+1), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)


def while_cond_jvp(carry):
    (_, err, dampening, cnt), (_, _, maxit, nsteps, tol, _, verbose) = carry
    cond = jnp.logical_and(err > tol, cnt*(nsteps+1) < maxit)
    verbose = jnp.logical_and(cond, verbose)
    jax.debug.callback(callback_func, cnt*(nsteps+1),
                       err, dampening, verbose=verbose)
    return cond


def sweep_banded_down(val, i):
    jav_func, fmod, forward_mat, X, shocks = val
    # calculate value and jacobians
    fval, (jac_f2xLag, jac_f2x, jac_f2xPrime) = jav_func(
        X[i], X[i+1], X[i+2], shocks=shocks[i])
    # work on banded sequence space jacobian
    bmat = jnp.linalg.inv(jac_f2x - jac_f2xLag @ forward_mat)
    forward_mat = bmat @ jac_f2xPrime
    fmod = bmat @ (fval - jac_f2xLag @ fmod)
    return (jav_func, fmod, forward_mat, X, shocks), (fmod, forward_mat)


def sweep_banded_up(val, i):
    forward_mat, fvals, fval = val
    # go backwards in time
    fval = fvals[i] - forward_mat[i] @ fval
    return (forward_mat, fvals, fval), fval


def check_status(err, cnt, maxit, tol):
    """Check whether to exit iteration and compile error message"""

    # exit causes
    if err < tol:
        return True, (True, "The solution converged.")
    elif jnp.isnan(err):
        return True, (False, "Function returns 'NaN's.")
    elif cnt > maxit:
        return True, (False, f"Maximum number of {maxit} iterations reached.")
    else:
        return False, (False, "")


def newton_for_jvp(jvp_func, jacobian, x_init, verbose, tol=1e-8, maxit=200, nsteps=2, factor=1.5):

    start_time = time.time()
    x = x_init[1:-1].flatten()

    (x, err, dampening, cnt_loop), _ = jax.lax.while_loop(while_cond_jvp, while_body_jvp,
                                                          ((x, 1., 0., 0), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)))

    ltime = time.time() - start_time
    cnt = cnt_loop*(nsteps+1)
    callback_func(cnt, err, dampening, ltime, verbose)
    _, (success, mess) = check_status(err, cnt, maxit, tol)

    # compile error/report message
    if not success and not jnp.isnan(err):
        mess += f" Max. error is {err:1.2e}."

    return x, not success, mess


def newton_for_banded_jac(jav_func, nvars, horizon, X, shocks, verbose, maxit=30, tol=1e-8):

    st = time.time()
    cnt = 0

    while True:

        _, (fvals, forward_mat) = jax.lax.scan(sweep_banded_down, (jav_func, jnp.zeros(
            nvars), jnp.zeros((nvars, nvars)), X, shocks), jnp.arange(horizon-1))
        _, out = jax.lax.scan(sweep_banded_up, (forward_mat, fvals, jnp.zeros(
            nvars)), jnp.arange(horizon-1), reverse=True)

        X = X.at[1:-1].add(-out)
        err = amax(out)
        cnt += 1

        ltime = time.time() - st

        if verbose:
            info_str = f'    Iteration {cnt:3d} | max error {err:.2e} | lapsed {ltime:3.4f}s'
            print(info_str)

        do_break, (success, mess) = check_status(err, cnt, maxit, tol)
        if do_break:
            break

    if not success:
        mess += f" Max. error is {err:1.2e}."

    return X, False, ''
