"""Newton heavy lifting functions and helpers
"""

import jax
import time
import jax.numpy as jnp
from grgrjax import callback_func, amax


def iteration_step(carry):
    (y, dampening, cnt), (x, f, jvp_func, jacobian, factor), (_, tol, maxit) = carry
    _, v = jvp_func(x, y)
    v = jax.scipy.linalg.lu_solve(jacobian, v)
    dampening = jnp.minimum(dampening, factor*(y.T@y)/(v.T@y))
    diff = f-v
    y += dampening*diff
    eps = amax(diff)
    return (y, dampening, cnt+1), (x, f, jvp_func, jacobian, factor), (eps, tol, maxit)


def iteration_cond(carry):
    (_, _, cnt), _, (eps, tol, maxit) = carry
    return jnp.logical_and(cnt <= maxit, eps > tol)


def jvp_while_body(carry):
    (x, _, _, cnt), (jvp_func, jacobian, maxit,
                     nsteps, tol, factor, verbose) = carry
    # first iteration
    f, _ = jvp_func(x, jnp.zeros_like(x))
    f = jax.scipy.linalg.lu_solve(jacobian, f)
    # other iterations
    init = ((f, 1., 0), (x, f, jvp_func, jacobian, factor), (1e8, 1e-5, nsteps))
    (y, dampening, cnt_inner), _, _ = jax.lax.while_loop(
        iteration_cond, iteration_step, init)
    return (x-y, amax(f), dampening, cnt+cnt_inner), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)


def jvp_while_cond(carry):
    (_, err, dampening, cnt), (_, _, maxit, nsteps, tol, _, verbose) = carry
    cond = jnp.logical_and(err > tol, cnt < maxit)
    verbose = jnp.logical_and(cond, verbose)
    verbose = jnp.logical_and(cnt, verbose)
    jax.debug.callback(callback_func, cnt, err, dampening, verbose=verbose)
    return cond


def sweep_banded_down(val, i):
    jav_func, fmod, forward_mat, X, shocks = val
    # calculate value and jacobians
    fval, (jac_f2xLag, jac_f2x, jac_f2xPrime) = jav_func(
        X[i], X[i+1], X[i+2], shocks=shocks[i])
    bmat = jac_f2x - jac_f2xLag @ forward_mat
    forward_mat = jnp.linalg.solve(bmat, jac_f2xPrime)
    fmod = jnp.linalg.solve(bmat, fval - jac_f2xLag @ fmod)
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


def newton_for_jvp(jvp_func, jacobian, x_init, verbose, tol=1e-8, maxit=500, nsteps=30, factor=1.5):

    start_time = time.time()
    x = x_init[1:-1].flatten()

    (x, err, dampening, cnt), _ = jax.lax.while_loop(jvp_while_cond, jvp_while_body,
                                                     ((x, 1., 0., 0), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)))
    ltime = time.time() - start_time
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
