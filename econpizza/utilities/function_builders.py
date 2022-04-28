
"""Build dynamic functions."""

import jax
import jax.numpy as jnp


def get_func_stst_raw(par, func_pre_stst, func_backw, func_stst_dist, func_eqns, shocks, init_vf, decisions_output_init, tol):

    def func_backw_ext(x):

        def cond_func(cont):
            return jnp.abs(cont[0]-cont[2]).max() > tol

        def body_func(cont):
            vf, _, _ = cont
            return *func_backw(x, x, x, x, vf, [], par), vf

        vf, decisions_output, _ = jax.lax.while_loop(
            cond_func, body_func, (init_vf, decisions_output_init, init_vf+1))

        return vf, decisions_output

    def func_stst_raw(x):

        x = func_pre_stst(x, par)[..., jnp.newaxis]

        if not func_stst_dist:
            return func_eqns(x, x, x, x, jax.numpy.zeros(len(shocks)), par)

        vf, decisions_output = func_backw_ext(x)
        dist = func_stst_dist(decisions_output)

        # TODO: for more than one dist this should be a loop...
        decisions_output_array = jnp.array(decisions_output)[..., jnp.newaxis]
        dist_array = jnp.array(dist)[..., jnp.newaxis]

        return func_eqns(x, x, x, x, [], par, dist_array, decisions_output_array)

    return func_stst_raw, func_backw_ext
