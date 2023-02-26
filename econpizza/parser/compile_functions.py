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

    if isinstance(calls, str):
        calls = calls.splitlines()

    func_str = f"""def func_backw(XLag, X, XPrime, XSS, VFPrime, shocks, pars):
            {compile_func_basics_str(evars, par, shocks)}
            \n ({"".join(v + ", " for v in inputs)}) = VFPrime
            \n %s
            \n return jnp.array(({"".join(v[:-5] + ", " for v in inputs)})), jnp.array(({", ".join(v for v in outputs)}))
            """ % '\n '.join(calls)

    return func_str


def compile_stst_func_str(evars, par, stst, init):
    """Compile all information from 'equations' section to a string that defines the function.
    """

    stst_str = '; '.join([f'{v} = {stst[v]}' for v in stst])

    # compile the final function string
    func_pre_stst_str = f"""def func_pre_stst(INTERNAL_init):
        \n ({"".join(v + ", " for v in init)}) = INTERNAL_init
        \n {stst_str}
        \n INTERNAL_vars = ({"".join(v + ", " for v in evars)})
        \n INTERNAL_par = ({"".join(p + ", " for p in par)})
        \n return jnp.array(INTERNAL_vars), jnp.array(INTERNAL_par)"""

    return func_pre_stst_str


def compile_forw_func_str(distributions, decisions_outputs):
    """Compiles the transition functions for distributions.
    """

    if len(distributions) > 1:
        raise NotImplementedError(
            'More than one distribution is not yet implemented.')

    # already prepare for more than one distributions
    # each distribution gets an own string
    func_forw_stst_str_tpl = ()
    func_forw_str_tpl = ()

    for dist_name in distributions:

        dist = distributions[dist_name]
        implemented_endo = ('exogenous_custom', 'exogenous_rouwenhorst')
        implemented_exo = ('endogenous_custom', 'endogenous_log')
        exog = [v for v in dist if dist[v]['type'] in implemented_endo]
        endo = [v for v in dist if dist[v]['type'] in implemented_exo]
        rest = [dist[v]['type'] for v in dist if dist[v]
                ['type'] not in implemented_endo + implemented_exo]

        if rest:
            raise NotImplementedError(
                f"Grid(s) of type {str(*rest)} not implemented.")

        if len(exog) > 1:
            raise NotImplementedError(
                'Exogenous distributions with more than one dimension are not yet implemented.')

        func_forw_str_tpl = f"\n endog_inds0, endog_probs0 = interp.interpolate_coord_robust({dist[endo[0]]['grid_name']}, {endo[0]})",

        if len(endo) == 1:
            func_forw_stst_str_tpl = func_forw_str_tpl + \
                (f"\n {dist_name}, {dist_name}_cnt = dists.stationary_distribution_forward_policy_1d(endog_inds0, endog_probs0, {dist[exog[0]]['transition_name']}, tol, maxit)",)
            func_forw_str_tpl += f"\n {dist_name} = {dist[exog[0]]['transition_name']}.T @ dists.forward_policy_1d({dist_name}, endog_inds0, endog_probs0)",

        elif len(endo) == 2:
            func_forw_str_tpl += f"\n endog_inds1, endog_probs1 = interp.interpolate_coord_robust({dist[endo[1]]['grid_name']}, {endo[1]})",
            func_forw_stst_str_tpl = func_forw_str_tpl + \
                (f"\n {dist_name}, {dist_name}_cnt = dists.stationary_distribution_forward_policy_2d(endog_inds0, endog_inds1, endog_probs0, endog_probs1, {dist[exog[0]]['transition_name']}, tol, maxit)",)
            func_forw_str_tpl += f"""
                \n forwarded_dist = dists.forward_policy_2d({dist_name}, endog_inds0, endog_inds1, endog_probs0, endog_probs1)
                \n {dist_name} = expect_transition({dist[exog[0]]['transition_name']}.T, forwarded_dist)
                """,

        else:
            raise NotImplementedError(
                'Endogenous distributions with more than two dimension are not yet implemented.')

    # join the tuples to strings that define the final functions
    func_forw_stst_str = f"""def func_forw_stst(decisions_outputs, tol, maxit):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n {"".join(func_forw_stst_str_tpl)}
        \n max_cnt = jnp.max({"".join(d + '_cnt, ' for d in distributions.keys())})
        \n return jnp.array(({"".join(d + ', ' for d in distributions.keys())})), max_cnt"""

    func_forw_str = f"""def func_forw(distributions, decisions_outputs):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n ({"".join(d+', ' for d in distributions)}) = distributions
        \n {"".join(func_forw_str_tpl)}
        \n return jnp.array(({"".join(d + ', ' for d in distributions.keys())}))"""

    return func_forw_stst_str, func_forw_str


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

    if isinstance(eqns_aux, str):
        eqns_aux = eqns_aux.splitlines()

    eqns_aux_stack = "\n ".join(eqns_aux) if eqns_aux else ""
    eqns_stack = "\n ".join(eqns)

    # compile the final function string
    func_str = f"""def func_eqns(XLag, X, XPrime, XSS, shocks, pars, distributions=[], decisions_outputs=[]):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(d+', ' for d in distributions)}) = distributions
        \n ({"".join(d+', ' for d in decisions_outputs)}) = decisions_outputs
        \n {eqns_aux_stack}
        \n {eqns_stack}
        \n {"return jnp.array([" + ", ".join("root_container"+str(i) for i in range(len(evars))) + "]).T.ravel()"}"""

    return func_str
