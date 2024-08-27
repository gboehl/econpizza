"""Functions that write other functions.
"""
import re
import jax
from .build_generic_functions import func_forw_generic, func_forw_stst_generic


def compile_func_basics_str(evars, par, shocks):

    func_str = f"""
 ({"".join(v + "Lag, " for v in evars)}) = XLag
 ({"".join(v + ", " for v in evars)}) = X
 ({"".join(v + "Prime, " for v in evars)}) = XPrime
 ({"".join(v + "SS, " for v in evars)}) = XSS
 ({"".join(p + ", " for p in par)}) = pars
 ({"".join(s + ", " for s in shocks)}) = shocks"""

    return func_str


def compile_backw_func_str(evars, par, shocks, inputs, outputs, calls):
    """Compile all information to a string that defines the backward function for 'decisions'.
    """

    if isinstance(calls, str):
        calls = calls.splitlines()

    func_str = f"""def func_backw(XLag, X, XPrime, XSS, WFPrime, shocks, pars):
            {compile_func_basics_str(evars, par, shocks)}
            \n ({"".join(v + ", " for v in inputs)}) = WFPrime
            \n %s
            \n return jnp.array(({"".join(v[:-5] + ", " for v in inputs)})), ({", ".join(v for v in outputs)})
            """ % '\n '.join(calls)

    return func_str


def get_forw_funcs(model):

    distributions = model['distributions']

    if len(distributions) > 1:
        raise NotImplementedError(
            'More than one distribution is not yet implemented.')

    # already prepare for more than one distributions
    for dist_name in distributions:

        dist = distributions[dist_name]

        # *_generic should be depreciated at some point
        implemented_endo = ('exogenous', 'exogenous_rouwenhorst', 'exogenous_generic', 'exogenous_custom')
        implemented_exo = ('endogenous', 'endogenous_log', 'endogenous_generic', 'endogenous_custom')
        exog = [v for v in dist if dist[v]['type'] in implemented_endo]
        endo = [v for v in dist if dist[v]['type'] in implemented_exo]
        other = [dist[v]['type'] for v in dist if dist[v]
                 ['type'] not in implemented_endo + implemented_exo]

        if len(exog) > 1:
            raise NotImplementedError(
                'Distributions with more than one exogenous variable are not yet implemented.')
        if len(endo) > 2:
            raise NotImplementedError(
                'Distributions with more than two endogenous variables are not yet implemented.')
        if other:
            raise NotImplementedError(
                f"Grid(s) of type {str(*other)} not implemented.")

        # for each object, check if it is provided in decisions_outputs
        try:
            transition = model['decisions']['outputs'].index(dist[exog[0]]['transition_name'])
        except ValueError:
            transition = model['context'][dist[exog[0]]['transition_name']]
        grids = []
        for i in endo:
            try:
                grids.append(model['decisions']['outputs'].index(dist[i]['grid_name']))
            except ValueError:
                grids.append(model['context'][dist[i]['grid_name']])
        indices = [model['decisions']['outputs'].index(i) for i in endo]

        model['context']['func_forw'] = jax.tree_util.Partial(
            func_forw_generic, grids=grids, transition=transition, indices=indices)
        model['context']['func_forw_stst'] = jax.tree_util.Partial(
            func_forw_stst_generic, grids=grids, transition=transition, indices=indices)

    return


def compile_eqn_func_str(evars, eqns, par, eqns_aux, shocks, distributions, decisions_outputs):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    # start compiling root_container
    for i, eqn in enumerate(eqns):
        eqsplit = re.split("(?<!=)=(?!=)", eqn)
        if len(eqsplit) == 1:
            eqns[i] = f"root_container{i} = {eqn}"
        elif len(eqsplit) == 2:
            eqns[i] = f"root_container{i} = {eqsplit[0]} - ({eqsplit[1]})"
        else:
            raise SyntaxError(f'More than one " = " in equation {i}: "{eqn}"')

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
