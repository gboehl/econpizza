"""Experimental features that are not in use.
"""

"""compile_functions.py

# add to parsing.load(...):
aggr_dict = model.get('aggregation')
eqns_aggr = [f'{v} = {aggr_dict[v]}' for v in aggr_dict] if aggr_dict else []
vars_aggr = list(aggr_dict.keys()) if aggr_dict else []

# and later:
model['func_strings']["func_eqns_alt"] = compile_eqn_func_str_alt(evars, eqns, par, eqns_aux=model.get('aux_equations'), shocks=shocks, vars_aggr=vars_aggr)
model['func_strings']["func_aggr"] = compile_aggr_func_str(evars, vars_aggr, eqns_aggr, par, shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

tmpf_names += define_function(model['func_strings']["func_eqns_alt"], model['context']),
tmpf_names += define_function(model['func_strings']["func_aggr"], model['context']),
"""


def compile_eqn_func_str_alt(evars, eqns, par, eqns_aux, shocks, vars_aggr):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    # start compiling root_container
    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = f"root_container{i} = {lhs} - ({rhs})"
        else:
            eqns[i] = f"root_container{i} = {eqn}"

    eqns_aux_stack = "\n ".join(eqns_aux) if eqns_aux else ""
    eqns_stack = "\n ".join(eqns)

    # compile the final function string
    func_str = f"""def func_eqns_alt(XLag, X, XPrime, XSS, shocks, pars, vars_aggr=[]):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(v+', ' for v in vars_aggr)}) = vars_aggr
        \n {eqns_aux_stack}
        \n {eqns_stack}
        \n {"return jnp.array([" + ", ".join("root_container"+str(i) for i in range(len(evars))) + "]).ravel()"}"""

    # never use real numpy
    return func_str.replace("np.", "jnp.").replace("jjnp.", "jnp.")


def compile_aggr_func_str(evars, vars_aggr, eqns, par, shocks, distributions, decisions_outputs):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    eqns_stack = "\n ".join(eqns)

    # compile the final function string
    func_str = f"""def func_aggr(XLag, X, XPrime, XSS, shocks, pars, distributions=[], decisions_outputs=[]):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(d+', ' for d in distributions)}) = distributions
        \n ({"".join(d+', ' for d in decisions_outputs)}) = decisions_outputs
        \n {eqns_stack}
        \n {"return jnp.array([" + "".join(v+', ' for v in vars_aggr) + "])"}"""

    # never use real numpy
    return func_str.replace("np.", "jnp.").replace("jjnp.", "jnp.")


"""function_builders.py

# add to find_stack(...):
func_eqns_alt = model['context']["func_eqns_alt"]
func_aggr = model['context'].get('func_aggr')

# and later:
func_aggr_stack = get_func_aggr_stack(pars, func_backw, func_dist, func_aggr, x0, stst, vfSS, distSS, zshock, tshock, horizon, nvars, endpoint, model.get('distributions'), shock)
func_stacked_alt = get_stacked_func_alt(pars, func_eqns_alt, x0, stst, zshock, tshock, horizon, nvars, endpoint, model.get('distributions'), shock)
stacked_func_raw = lambda x: func_stacked_alt(x, func_aggr_stack(x))
"""


def get_func_aggr_stack(pars, func_backw, func_dist, func_aggr, x0, stst, vfSS, distSS, zshock, tshock, horizon, nvars, endpoint, has_distributions, shock):

    nshpe = (nvars, horizon-1)

    def backwards_step(carry, i):

        vf_old, X = carry
        vf, decisions_output = func_backw(
            X[:, i], X[:, i+1], X[:, i+2], stst, vf_old, [], pars)

        return (vf, X), jnp.array(decisions_output)

    def forwards_step(carry, i):

        dist_old, decisions_output_storage = carry
        dist = func_dist(dist_old, decisions_output_storage[..., i])
        dist_array = jnp.array(dist)

        return (dist_array, decisions_output_storage), dist_array

    def func_aggr_stack(x):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T

        if has_distributions:
            # backwards step
            _, decisions_output_storage = jax.lax.scan(
                backwards_step, (vfSS, X), jnp.arange(horizon-2, -1, -1))
            decisions_output_storage = jnp.flip(decisions_output_storage, 0)
            decisions_output_storage = jnp.moveaxis(
                decisions_output_storage, 0, -1)
            # forwards step
            _, dists_storage = jax.lax.scan(
                forwards_step, (distSS, decisions_output_storage), jnp.arange(horizon-1))
            dists_storage = jnp.moveaxis(dists_storage, 0, -1)
        else:
            decisions_output_storage, dists_storage = [], []

        return func_aggr(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(nshpe), stst, zshock, pars, dists_storage, decisions_output_storage).ravel()

    return func_aggr_stack


def get_stacked_func_alt(pars, func_eqns_alt, x0, stst, zshock, tshock, horizon, nvars, endpoint, has_distributions, shock):

    nshpe = (nvars, horizon-1)

    def stacked_func(x, vars_aggr):

        X = jax.numpy.vstack((x0, x.reshape((horizon - 1, nvars)), endpoint)).T

        out = func_eqns_alt(X[:, :-2].reshape(nshpe), X[:, 1:-1].reshape(nshpe), X[:, 2:].reshape(
            nshpe), stst, zshock, pars, vars_aggr.reshape(-1, horizon-1))

        if shock is not None:
            out = out.at[jnp.arange(nvars)*(horizon-1)].set(
                func_eqns(X[:, 0], X[:, 1], X[:, 2], stst, tshock, pars))

        return out

    return stacked_func
