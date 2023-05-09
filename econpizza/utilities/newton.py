"""Newton heavy lifting functions and helpers
"""

from grgrjax import jax_print
import jax
import time
import jax.numpy as jnp
from functools import partial
from jax._src.lax.linalg import lu_solve
from grgrjax import callback_func, amax, newton_jax_jit


def callback_with_damp(cnt, err, fev, err_inner, dampening, ltime, verbose):
    inner = f' | inner {err_inner:.2e}'
    damp = f' | dampening {dampening:1.3f}'
    return callback_func(cnt, err, inner, damp, fev=fev, ltime=ltime, verbose=verbose)


def iteration_step(carry):
    (y, dampening, fev), (x, f, jvp_func, jacobian, factor), (_, tol, maxit) = carry
    _, v = jvp_func(x, y)
    v = lu_solve(*jacobian[0], v, 0)[jacobian[1]]
    dampening = jnp.minimum(dampening, factor*jnp.abs((y.T@y)/(v.T@y)))
    diff = f-v
    y += dampening*diff
    eps = amax(diff)
    return (y, dampening, fev+1), (x, f, jvp_func, jacobian, factor), (eps, tol, maxit)


def iteration_cond(carry):
    (_, _, fev), _, (eps, tol, maxit) = carry
    return jnp.logical_and(fev <= maxit, eps > tol)


def jvp_while_body(carry):
    (x, _, _, cnt, fev, _), (jvp_func, jacobian, maxit,
                             nsteps, tol, factor, verbose) = carry
    # first iteration
    f, _ = jvp_func(x, jnp.zeros_like(x))
    f = lu_solve(*jacobian[0], f, 0)[jacobian[1]]
    # other iterations
    iteration_tol = jnp.minimum(1e-5, 1e-1*amax(f))
    init = ((f, 1., 0), (x, f, jvp_func, jacobian, factor),
            (1e8, iteration_tol, nsteps))
    (y, dampening, fev_inner), _, (err_inner, _, _) = jax.lax.while_loop(
        iteration_cond, iteration_step, init)
    return (x-y, f, dampening, cnt+1, fev+fev_inner, err_inner), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)


def jvp_while_cond(carry):
    (_, f, dampening, cnt, fev, err_inner), (_,
                                             _, maxit, nsteps, tol, _, verbose) = carry
    err = amax(f)
    cond = jnp.logical_and(err > tol, cnt < maxit)
    verbose = jnp.logical_and(cond, verbose)
    verbose = jnp.logical_and(fev, verbose)
    jax.debug.callback(callback_with_damp, cnt, err, fev=fev,
                       err_inner=err_inner, dampening=dampening, ltime=None, verbose=verbose)
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
        return True, (False, "Function returns NaNs.")
    elif cnt >= maxit:
        return True, (False, f"Maximum number of {maxit} iterations reached.")
    else:
        return False, (False, "")


def newton_for_jvp(jvp_func, jacobian, x_init, verbose, tol=1e-8, maxit=20, nsteps=30, factor=1.5):
    """Newton solver for heterogeneous agents models as described in the paper.

    Parameters
    ----------
    tol : float, optional
        tolerance of the Newton method, defaults to ``1e-8``
    maxit : int, optional
        maximum of iterations for the Newton method, defaults to 20
    nsteps : int, optional
        number of function evaluations per Newton iteration, defaults to 30
    factor : float, optional
        dampening factor (gamma in the paper), Defaults to 1.5
    """

    start_time = time.time()
    x = x_init[1:-1].flatten()

    (x, f, dampening, cnt, fev, err_inner), _ = jax.lax.while_loop(jvp_while_cond, jvp_while_body,
                                                                   ((x, x, 0., 0, 0, 0), (jvp_func, jacobian, maxit, nsteps, tol, factor, verbose)))
    err = amax(f)
    ltime = time.time() - start_time
    callback_with_damp(cnt, err, fev=fev, err_inner=err_inner,
                       dampening=dampening, ltime=ltime, verbose=verbose)
    _, (success, mess) = check_status(err, cnt, maxit, tol)
    # compile error/report message
    if not success and not jnp.isnan(err):
        mess += f" Max. error is {err:1.2e}."

    return x, f, not success, mess


def newton_for_banded_jac(jav_func, nvars, horizon, X, shocks, verbose, maxit=30, tol=1e-8):
    """Newton solver for representative agents models.

    Parameters
    ----------
    tol : float, optional
        tolerance of the Newton method, defaults to ``1e-8``
    maxit : int, optional
        maximum of iterations for the Newton method, defaults to 20
    """

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

    return X, out, not success, mess


def newton_jax_jit_wrapper(func, init, **args):
    """Wrapper around grgrjax.newton.newton_jax_jit. Returns correct flags and messages.
    """

    if 'tol' not in args:
        args['tol'] = 1e-8
    if 'maxit' not in args:
        args['maxit'] = 30

    x, (f, _), cnt, flag = newton_jax_jit(func, init, **args)
    err = amax(f)
    _, (success, mess) = check_status(err, cnt, args['maxit'], args['tol'])
    flag |= not success

    # compile error/report message
    if not success and not jnp.isnan(err):
        mess += f" Max. error is {err:1.2e}."
    return x, f, flag, mess
