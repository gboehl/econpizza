import jax
import time
import jax.numpy as jnp


def accumulate(i_and_j, carry):
    # accumulate effects over different horizons
    jac, horizon = carry
    i = i_and_j // (horizon-2)
    j = i_and_j % (horizon-2)
    jac = jac.at[i+1, :, j+1].add(jac[i, :, j])
    return jac, horizon


def get_stst_jacobian(model, derivatives, horizon, nvars, verbose):

    st = time.time()

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

    # accumulate and flatten
    jac, _ = jax.lax.fori_loop(0, (horizon-2)**2, accumulate, (jac, horizon))
    jac = jac.reshape(((horizon-1)*nvars, (horizon-1)*nvars))

    # store result
    model['jac'] = jac
    model['jac_factorized'] = jax.scipy.linalg.lu_factor(jac)

    if verbose:
        duration = time.time() - st
        print(
            f"(get_jacobian:) Jacobian accumulation and decomposition done ({duration:1.3f}s).")

    return 0
