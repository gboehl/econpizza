"""Functions that write other functions.
"""


def compile_func_basics_str(evars, par, shocks):

    func_str = f"""
        \n ({"".join(v + "Lag, " for v in evars)}) = XLag
        \n ({"".join(v + ", " for v in evars)}) = X
        \n ({"".join(v + "Prime, " for v in evars)}) = XPrime
        \n ({"".join(v + "SS, " for v in evars)}) = XSS
        \n ({"".join(p + ", " for p in par)}) = pars
        \n ({"".join(s + ", " for s in shocks)}) = shocks"""

    return func_str


def compile_backw_func_str(evars, par, shocks, inputs, outputs, calls):
    """Compile all information to a string that defines the backward function for 'decisions'.
    """

    func_str = f"""def func_backw_raw(XLag, X, XPrime, XSS, VFPrime, shocks, pars):
            {compile_func_basics_str(evars, par, shocks)}
            \n ({", ".join(v for v in inputs)}) = VFPrime
            \n %s
            \n return ({", ".join(v[:-5] for v in inputs)}), ({", ".join(v for v in outputs)})
            """ % '\n '.join(calls)

    # never use real numpy
    return func_str.replace("np.", "jnp.").replace("jjnp.", "jnp.")


def compile_stst_func_str(evars, eqns, par, stst_eqns):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    stst_eqns_stack = "\n ".join(stst_eqns)

    # compile the final function string
    func_pre_stst_str = f"""def func_pre_stst(X, pars):
        \n XSS = XLag = XPrime = X
        \n shocks = []
        {compile_func_basics_str(evars, par, [])}
        \n {stst_eqns_stack}
        \n X = ({"".join(v + ", " for v in evars)})
        \n return jnp.array(X)"""

    # never use real numpy
    return func_pre_stst_str.replace("np.", "jnp.").replace("jjnp.", "jnp.")


def compile_eqn_func_str(evars, eqns, par, eqns_aux, shocks, distributions, decisions_outputs):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    # start compiling root_container
    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = "root_container%s = " % i + lhs + "- (" + rhs + ")"
        else:
            eqns[i] = "root_container%s = " % i + eqn

    eqns_aux_stack = "\n ".join(eqns_aux) if eqns_aux else ""
    eqns_stack = "\n ".join(eqns)

    # compile the final function string
    func_str = f"""def func_raw(XLag, X, XPrime, XSS, shocks, pars, dists=[], decisions_outputs=[]):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(d+', ' for d in distributions)}) = dists
        \n ({"".join(d+', ' for d in decisions_outputs)}) = decisions_outputs
        \n {eqns_aux_stack}
        \n {eqns_stack}
        \n {"return jnp.array([" + ", ".join("root_container"+str(i) for i in range(len(evars))) + "]).ravel()"}""" % (
    )

    # never use real numpy
    return func_str.replace("np.", "jnp.").replace("jjnp.", "jnp.")


def compile_func_dist_str(distributions, decisions_outputs):
    """Compiles the transition functions for distributions.
    """

    if len(distributions) > 1:
        raise NotImplementedError(
            'More than one distribution is not yet implemented.')

    # already prepare for more than one distributions
    # each distribution gets an own string
    func_stst_dist_str_tpl = ()
    func_dist_str_tpl = ()

    for dist_name in distributions:

        dist = distributions[dist_name]
        exog = [v for v in dist if dist[v]['type'] == 'exogenous']
        endo = [v for v in dist if dist[v]['type'] == 'endogenous']

        if len(exog) > 1:
            raise NotImplementedError(
                'Endogenous distributions larger thank 1-D are not yet implemented.')

        func_stst_dist_str_tpl += f"""
            \n endog_inds, endog_probs = dists.interpolate_coord_robust({dist[endo[0]]['grid_variables']}, {endo[0]})
            \n {dist_name} = dists.stationary_distribution_forward_policy_1d(endog_inds, endog_probs, {dist[exog[0]]['grid_variables'][2]})
            """,

        func_dist_str_tpl += f"""
            \n endog_inds, endog_probs = dists.interpolate_coord_robust({dist[endo[0]]['grid_variables']}, {endo[0]})
            \n {dist_name} = {dist[exog[0]]['grid_variables'][2]}.T @ dists.stationary_distribution_forward_policy_1d({dist_name}, endog_inds, endog_probs)
            """,

    # join the tuples to one string that defines the final function
    func_stst_dist_str = f"""def func_stst_dist(decisions_outputs):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n {"".join(func_stst_dist_str_tpl)}
        \n return ({"".join(d + ', ' for d in distributions.keys())})"""

    func_dist_str = f"""def func_dist(dist, decisions_outputs):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n {"".join(func_stst_dist_str_tpl)}
        \n return ({"".join(d + ', ' for d in distributions.keys())})"""

    return func_stst_dist_str, func_dist_str
