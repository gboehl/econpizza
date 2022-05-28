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


def compile_backw_func_str(evars, par, shocks, inputs, outputs, calls, exog_grid_var_names):
    """Compile all information to a string that defines the backward function for 'decisions'.
    """

    func_str = f"""def func_backw(XLag, X, XPrime, XSS, VFPrime, shocks, pars):
            {compile_func_basics_str(evars, par, shocks)}
            \n ({"".join(v + ", " for v in inputs)}) = VFPrime
            \n %s
            \n return jnp.array(({"".join(v[:-5] + ", " for v in inputs)})), jnp.array(({", ".join(v for v in outputs)})), ({', '.join(v for v in exog_grid_var_names)})
            """ % '\n '.join(calls)

    return func_str


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

    return func_pre_stst_str


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
        exog = [v for v in dist if dist[v]['type'] in (
            'exogenous', 'custom_exogenous', 'time_varying_exogenous')]
        endo = [v for v in dist if dist[v]['type'] == 'endogenous']

        if len(exog) > 1:
            raise NotImplementedError(
                'Exogenous distributions with more than one dimension are not yet implemented.')

        func_stst_dist_str_tpl += f"\n endog_inds0, endog_probs0 = interp.interpolate_coord_robust({dist[endo[0]]['grid_variables']}, {endo[0]})",
        func_dist_str_tpl += f"\n endog_inds0, endog_probs0 = interp.interpolate_coord_robust({dist[endo[0]]['grid_variables']}, {endo[0]})",

        if len(endo) == 1:
            func_stst_dist_str_tpl += f"\n {dist_name}, {dist_name}_cnt = dists.stationary_distribution_forward_policy_1d(endog_inds0, endog_probs0, {dist[exog[0]]['grid_variables'][2]}, tol, maxit)",
            func_dist_str_tpl += f"\n {dist_name} = {dist[exog[0]]['grid_variables'][2]}.T @ dists.forward_policy_1d({dist_name}, endog_inds0, endog_probs0)",

        elif len(endo) == 2:
            func_stst_dist_str_tpl += f"""
                \n endog_inds1, endog_probs1 = interp.interpolate_coord_robust({dist[endo[1]]['grid_variables']}, {endo[1]})
                \n {dist_name}, {dist_name}_cnt = dists.stationary_distribution_forward_policy_2d(endog_inds0, endog_inds1, endog_probs0, endog_probs1, {dist[exog[0]]['grid_variables'][2]}, tol, maxit)
                """,
            func_dist_str_tpl += f"""
                \n endog_inds1, endog_probs1 = interp.interpolate_coord_robust({dist[endo[1]]['grid_variables']}, {endo[1]})
                \n forwarded_dist = dists.forward_policy_2d({dist_name}, endog_inds0, endog_inds1, endog_probs0, endog_probs1)
                \n {dist_name} = expect_transition({dist[exog[0]]['grid_variables'][2]}.T, forwarded_dist)
                """,

        else:
            raise NotImplementedError(
                'Endogenous distributions with more than two dimension are not yet implemented.')

    # join the tuples to strings that define the final functions
    func_stst_dist_str = f"""def func_stst_dist(decisions_outputs, tol, maxit):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n {"".join(func_stst_dist_str_tpl)}
        \n max_cnt = jnp.max({"".join(d + '_cnt, ' for d in distributions.keys())})
        \n return jnp.array(({"".join(d + ', ' for d in distributions.keys())})), max_cnt"""

    func_dist_str = f"""def func_dist(distributions, decisions_outputs):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n ({"".join(d+', ' for d in distributions)}) = distributions
        \n {"".join(func_dist_str_tpl)}
        \n return jnp.array(({"".join(d + ', ' for d in distributions.keys())}))"""

    return func_stst_dist_str, func_dist_str


def compile_eqn_func_str(evars, eqns, par, eqns_aux, shocks, distributions, decisions_outputs):
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
    func_str = f"""def func_eqns(XLag, X, XPrime, XSS, shocks, pars, distributions=[], decisions_outputs=[]):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(d+', ' for d in distributions)}) = distributions
        \n ({"".join(d+', ' for d in decisions_outputs)}) = decisions_outputs
        \n {eqns_aux_stack}
        \n {eqns_stack}
        \n {"return jnp.array([" + ", ".join("root_container"+str(i) for i in range(len(evars))) + "]).ravel()"}"""

    return func_str
