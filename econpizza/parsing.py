#!/bin/python
# -*- coding: utf-8 -*-
"""Functions for model parsing yaml -> working model instance. Involves a lot of dynamic function definition...
"""

import yaml
import re
import os
import tempfile
import jax
import jaxlib
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from jax.numpy import log, exp, sqrt, maximum, minimum
from grgrlib import load_as_module
from inspect import getmembers, isfunction
from jax.experimental.host_callback import id_print as jax_print
from .steady_state import solve_stst, solve_linear
from .utilities import grids, dists

jax.config.update("jax_enable_x64", True)
# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


# initialize model cache
cached_mdicts = ()
cached_models = ()


def eval_strs(vdict, pars=None, context=globals()):

    if vdict is None:
        return None

    if pars:
        exec(
            f"{', '.join(pars.keys())} = {', '.join(str(p) for p in pars.values())}", context)

    for v in vdict:
        exec(f'{v} = {vdict[v]}', context)
        if isinstance(vdict[v], str):
            vdict[v] = eval(f'{v}')

    return vdict


def load_functions_file(model, context):
    """Load the functions file as a module.
    """

    try:
        # prepare path
        if not os.path.isabs(model["functions_file"]):
            yaml_dir = os.path.dirname(model["path"])
            functions_file = os.path.join(yaml_dir, model["functions_file"])
        # load as a module
        context['module'] = load_as_module(functions_file)

        def func_or_compiled(func): return isinstance(
            func, jaxlib.xla_extension.CompiledFunction) or isfunction(func)
        for m in getmembers(module, func_or_compiled):
            exec(f'{m[0]} = module.{m[0]}', context)

    except KeyError:
        pass

    return


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
    return func_str.replace(" np.", " jnp.")


def compile_eqn_func_str(evars, eqns, par, eqns_aux, stst_eqns, shocks, distributions, decisions_outputs):
    """Compile all information from 'equations' section' to a string that defines the function.
    """

    # start compiling root_container
    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = "root_container%s = " % i + lhs + "- (" + rhs + ")"
        else:
            eqns[i] = "root_container%s = " % i + eqn

    # resolve all equations with the 'All' keyword
    for eqn in stst_eqns:
        if eqn.count("All") == 1:
            for vtype in ("", "SS", "Lag", "Prime"):
                stst_eqns.append(eqn.replace("All", vtype))
        elif eqn.count("All") > 1:
            raise NotImplementedError(
                "Multiple `All` in one equation are not implemented"
            )

    # compile the final function string
    func_str = f"""def func_raw(XLag, X, XPrime, XSS, shocks, pars, dists=[], decisions_outputs=[], stst=False, return_stst_vals=False):
        {compile_func_basics_str(evars, par, shocks)}
        \n ({"".join(d+', ' for d in distributions)}) = dists
        \n ({"".join(d+', ' for d in decisions_outputs)}) = decisions_outputs
        \n %s \n %s\n %s \n %s
        \n {"return np.array([" + ", ".join("root_container"+str(i) for i in range(len(evars))) + "])"}""" % (
        "if stst:\n  " + "\n  ".join(stst_eqns) if stst_eqns else "",
        "\n ".join(eqns_aux) if eqns_aux else "",
        "if return_stst_vals:\n  " +
        "return np.array([" + ", ".join(v for v in evars) + "])",
        "\n ".join(eqns),
    )

    # never use real numpy
    return func_str.replace(" np.", " jnp.")


def compile_dist_func_str(distributions, decisions_outputs):

    if len(distributions) > 1:
        raise NotImplementedError(
            'More than one distribution is not yet implemented.')

    stst_dist_func_str_tpl = ()

    for dist_name in distributions:

        dist = distributions[dist_name]
        exog = [v for v in dist if dist[v]['type'] == 'exogenous']
        endo = [v for v in dist if dist[v]['type'] == 'endogenous']

        if len(exog) > 1:
            raise NotImplementedError(
                'More than 1-D endogenous distributions are not yet implemented.')

        stst_dist_func_str_tpl += f"""
            \n endog_inds, endog_probs = dists.interpolate_coord_robust({dist[endo[0]]['grid_variables']}, {endo[0]})
            \n tmat_endo = dists.tmat_from_endo(endog_inds, endog_probs)
            \n tmat_exog = dists.tmat_from_exog({dist[exog[0]]['grid_variables'][2]}.T, endog_probs)
            \n tmat = tmat_exog @ tmat_endo
            \n {dist_name} = dists.stationary_distribution_iterative(tmat).reshape(endog_probs.shape)""",

    stst_dist_func_str = f"""def stst_dist_func(decisions_outputs):
        \n ({", ".join(decisions_outputs)},) = decisions_outputs
        \n {"".join(stst_dist_func_str_tpl)}
        \n return {", ".join(distributions.keys())}"""

    # TODO: also compile dynamic distributions func str

    return stst_dist_func_str


def compile_init_values(evars, initvals, stst):
    """Combine all available information in initial guesses.
    """

    # get inital values to test the function
    init = np.ones(len(evars)) * 1.1

    if initvals is not None:
        for v in initvals:
            try:
                init[evars.index(v)] = initvals[v]
            except ValueError:
                pass

    if stst:
        for v in stst:
            init[evars.index(v)] = stst[v]

    return init


def check_if_defined(evars, eqns):
    """Check if all variables are defined in period t.
    """

    for v in evars:
        v_in_eqns = [
            v in e.replace(v + "SS", "").replace(v + "Lag",
                                                 "").replace(v + "Prime", "")
            for e in eqns
        ]
        if not np.any(v_in_eqns):
            raise Exception(f"Variable `{v}` is not defined for time t.")
    return


def check_dublicates_and_determinancy(evars, eqns):
    """Check if there are dublicates and the numbers of eqns/vars match.
    """

    dubs = [x for i, x in enumerate(evars) if x in evars[:i]]
    dubmess = (
        ", variables list contains dublicate(s): %s" % ", ".join(
            dubs) if dubs else ""
    )

    sorted_evars = evars[:] = sorted(list(set(evars)), key=str.lower)

    if len(sorted_evars) != len(eqns):
        raise Exception(
            "Model has %s variables but %s equations%s."
            % (len(sorted_evars), len(eqns), dubmess)
        )
    elif dubs:
        print("(parse:) Warning%s" % dubmess)

    return sorted_evars


def check_func(func_raw, init, shocks, par):

    test = func_raw(
        init, init, init, init, np.zeros(
            len(shocks)), np.array(list(par.values())), [], []
    )
    if np.isnan(test).any():
        raise Exception("Initial values are NaN.")
    if np.isinf(test).any():
        raise Exception("Initial values are not finite.")

    return


def parse(mfile):
    """parse from yaml file"""

    f = open(mfile)
    mtxt = f.read()
    f.close()

    mtxt = mtxt.replace("^", "**")
    mtxt = re.sub(r"@ ?\n", " ", mtxt)
    # try to detect if `~` wants to be a `-`
    mtxt = mtxt.replace("\n ~ ", "\n - ")
    mtxt = mtxt.replace("\n  ~ ", "\n  - ")
    mtxt = mtxt.replace("   ~ ", "   - ")

    # get dict
    model = yaml.safe_load(mtxt)
    # create nice shortcuts
    model["pars"] = model["parameters"]
    model["vars"] = model["variables"]

    return model


def load(
    model,
    raise_errors=True,
    lti_max_iter=500,
    verbose=True,
    testing=False,
):
    """load model from dict or yaml file. Warning: contains filthy code (eg. globals, exec, ...)"""

    from .__init__ import PizzaModel

    global cached_mdicts, cached_models

    if isinstance(model, str):
        full_path = model
        model = parse(model)
        model['path'] = full_path

    model = PizzaModel(model)

    # check if model is already cached
    if model in cached_mdicts:
        model = cached_models[cached_mdicts.index(model)]
        model['context'] = globals()
        load_functions_file(model, model['context'])
        print("(parse:) Loading cached model.")
        return model

    mdict_raw = deepcopy(model)
    model['context'] = globals()

    # load file with additional functions as module (if it exists)
    load_functions_file(model, model['context'])

    defs = model.get("definitions")
    # never ever use real numpy
    if defs is not None:
        for d in defs:
            d = d.replace(" numpy ", " jax.numpy ")
            # execute these definitions globally (TODO: is this a good idea?)
            exec(d, model['context'])

    eqns = model["equations"].copy()

    # check if there are dublicate variables
    evars = check_dublicates_and_determinancy(model["variables"], eqns)
    # check if each variable is defined in time t (only defining xSS does not give a valid root)

    if model.get('distributions'):
        # create strings of the function that define the grids
        grid_strings = grids.create_grids(model['distributions'])

        # execute all of them
        for grid_str in grid_strings:
            exec(grid_str, model['context'])
    else:
        check_if_defined(evars, eqns)

    shocks = model.get("shocks") or ()
    par = eval_strs(model["parameters"])
    model["stst"] = stst = eval_strs(
        model["steady_state"].get("fixed_values"), pars=par)
    model["root_options"] = {}

    # collect number of foward and backward looking variables
    model["no_fwd"] = sum(
        var + "Prime" in "".join(model["equations"]) for var in evars)
    model["no_bwd"] = sum(var + "Lag" in "".join(model["equations"])
                          for var in evars)

    # collect initial guesses
    model["init"] = init = compile_init_values(
        evars, eval_strs(model["steady_state"].get("init_guesses")), stst)

    stst_eqns = model["steady_state"].get("equations") or []
    # add fixed values to the steady state equations
    if stst is not None:
        for key in stst:
            stst_eqns.append(key + "SS = " + str(stst[key]))

    # get function strings for decisions and distributions, if they exist
    if model.get('decisions'):
        decisions_outputs = model['decisions']['outputs']
        model["func_backw_str"] = compile_backw_func_str(
            evars, par, shocks, model['decisions']['inputs'], decisions_outputs, model['decisions']['calls'])
    else:
        decisions_outputs = []
    if model.get('distributions'):
        dist_names = list(model['distributions'].keys())
        model["dist_func_str"] = compile_dist_func_str(
            model['distributions'], decisions_outputs)
    else:
        dist_names = []

    # get a string that contains the function definition
    model["func_str"] = func_str = compile_eqn_func_str(evars, eqns, par, eqns_aux=model.get(
        'aux_equations'), stst_eqns=stst_eqns, shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpf.write(func_str)
    tmpf.close()

    # define the function
    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), model['context'])
    model["func_raw"] = func_raw

    # TODO: reactivate
    if not testing:
        # try if function works on initvals. If it does, jit-compile it and remove tempfile
        check_func(func_raw, init, shocks, par)
        model["func"] = jax.jit(func_raw, static_argnums=(8, 9))
        # model["func"] = jax.jit(func_raw, static_argnums=(7,))
        # TODO: stst_eqns must be executed _outside_ of eqn_func. Then the jacobian would be reusable

    # unlink the temporary file
    os.unlink(tmpf.name)

    if verbose:
        print("(parse:) Parsing done.")

    # add new model to cache
    cached_mdicts += (mdict_raw,)
    cached_models += (model,)

    return model
