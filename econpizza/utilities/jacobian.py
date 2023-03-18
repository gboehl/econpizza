import jax
import time
import jax.numpy as jnp
import scipy.sparse as ssp
from jax._src.api import partial
from jax._src.typing import Array
from grgrjax import jax_print


def accumulate(i_and_j: Array, carry: (Array, Array)) -> (Array, int):
    # accumulate effects over different horizons
    jac, horizon = carry
    i = i_and_j // (horizon-2)
    j = i_and_j % (horizon-2)
    jac = jac.at[i+1, :, j+1].add(jac[i, :, j])
    return jac, horizon


@jax.jit
def get_stst_jacobian_jit(derivatives, horizon):
    # load derivatives
    (jac_f2xLag, jac_f2x, jac_f2xPrime), jac_f2do, jac_do2x = derivatives

    # jacobian is quasi-kronecker of f2do x do2x
    jac = jnp.tensordot(jac_f2do[:, ::-1],
                        jac_do2x[..., ::-1, :], jac_f2do.ndim-2)
    jac = jnp.moveaxis(jac, 0, 1)

    # add direct effects of x on f
    jac = jac.at[0, :, 0, :].add(jac_f2x[..., 0])
    jac = jac.at[0, :, 1, :].add(jac_f2xPrime[..., 0])
    jac = jac.at[1, :, 0, :].add(jac_f2xLag[..., 0])

    # accumulate
    jac, _ = jax.lax.fori_loop(0, (horizon-2)**2, accumulate, (jac, horizon))
    return jac


def lu_factor_from_sparse(lu):
    """Translate the output of scipy.sparse.linalg.splu to something that jax.scipy.linalg.lu_solve understands
    """
    pr = lu.perm_r
    pc = lu.perm_c
    n = len(pr)
    pr = jnp.empty(n, dtype=int).at[pr].set(jnp.arange(n))
    lu_factor = lu.L.A - jnp.eye(n) + lu.U.A
    return (lu_factor, pr), pc


def get_stst_jacobian(model, derivatives, horizon, nvars, verbose):
    """Calculate the steady state jacobian
    """
    st = time.time()
    # do the accumulation in jitted jax and flatten
    jac = get_stst_jacobian_jit(derivatives, horizon)
    jac = jac.reshape(((horizon-1)*nvars, (horizon-1)*nvars))
    # store result
    model['cache']['jac'] = jac
    # use sparse SuperLU because it is wayyy faster
    sparse_jac = ssp.csc_matrix(jac)
    sparse_jac_lu = ssp.linalg.splu(sparse_jac)
    model['cache']['jac_factorized'] = lu_factor_from_sparse(sparse_jac_lu)
    if verbose:
        duration = time.time() - st
        print(
            f"(get_jacobian:) Jacobian accumulation and decomposition done ({duration:1.3f}s).")

    return 0


def vmapped_jvp(jvp, primals, tangents):
    """Compact version of jvp_vmap from grgrjax
    """
    pushfwd = partial(jvp, primals)
    y, jac = jax.vmap(pushfwd, out_axes=(None, -1), in_axes=-1)(tangents)
    return y, jac


def jac_slicer(i, carry):
    """Calclulates a chunk of the jacobian
    """
    (_, jac), (x, jvp, zeros_slice, marginal_base, chunk_size) = carry
    # get base slice
    base_slice = jax.lax.dynamic_update_slice(
        zeros_slice, marginal_base, (i*chunk_size, len(x)))
    # calculate slice of the jacobian
    f, jac_slice = vmapped_jvp(jvp, x, base_slice)
    # update jacobian
    jac = jax.lax.dynamic_update_slice(jac, jac_slice, (0, i*chunk_size))
    return (f, jac), (x, jvp, zeros_slice, marginal_base, chunk_size)


def jac_and_value_sliced(jvp, chunk_size, zero_slice, eye_chunk, x):
    """Calculate the value and jacobian at `x` while only evaluating chunks of the full jacobian at times. May be necessary due to memmory requirements.
    """
    x_shape = len(x)
    nloops = jnp.ceil(x_shape/chunk_size).astype(jnp.int64)
    init_vals = x, jnp.zeros((x_shape, x_shape))
    args = x, jvp, zero_slice, eye_chunk, chunk_size
    # in essence a wrapper around a for loop over `jac_slicer`
    (f, jac), _ = jax.lax.fori_loop(0, nloops, jac_slicer, (init_vals, args))
    return f, jac


def get_jac_and_value_sliced(dimx, jvp, newton_args):
    """Get the jac_and_value_sliced function. This is necessary because objects depending on chunk_size must be defined outsite the jitted function
    """
    # get chunk_size from optional dictionary
    if 'chunk_size' in newton_args:
        chunk_size = newton_args['chunk_size']
        newton_args.pop('chunk_size')
    else:
        chunk_size = 100

    # define objects that depend on chunk_size
    zero_slice = jnp.zeros((dimx, chunk_size))
    eye_chunk = jnp.eye(chunk_size)
    return jax.tree_util.Partial(jac_and_value_sliced, jvp, chunk_size, zero_slice, eye_chunk)
