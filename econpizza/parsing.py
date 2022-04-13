#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import re
import os
import tempfile
import numpy as np
from copy import deepcopy
from jax.numpy import log, exp, sqrt, maximum, minimum
from .steady_state import solve_stst, solve_linear
import jax
import jax.numpy as jnp
from grgrlib import load_as_module

jax.config.update("jax_enable_x64", True)
# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


# initialize model cache
cached_mdicts = ()
cached_models = ()


def eval_strs(vdict):

    if vdict is None:
        return None

    for v in vdict:
        if isinstance(vdict[v], str):
            vdict[v] = eval(vdict[v])

    return vdict


def load_functions_file(model):
    """Load the functions file as a module.
    """

    try:
        # prepare path
        if not os.path.isabs(model["functions_file"]):
            yaml_dir = os.path.dirname(model["path"])
            functions_file = os.path.join(yaml_dir, model["functions_file"])
        # load as a module
        return load_as_module(functions_file)
    except KeyError:
        pass


def compile_func_str(evars, eqns, par, eqns_aux, stst_eqns, shocks):
    """Compile all information to a string that defines the function.
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

    if not shocks:
        shock_str = ""
    else:
        shock_str = "(" + ", ".join(shocks) + ")" + " = shocks"

    # compile the final function string
    func_str = f"""def func_raw(XLag, X, XPrime, XSS, shocks, pars, stst=False, return_stst_vals=False):
        \n ({", ".join(v + "Lag" for v in evars)}) = XLag
        \n ({", ".join(evars)}) = X
        \n ({", ".join(v + "Prime" for v in evars)}) = XPrime
        \n ({", ".join(v + "SS" for v in evars)}) = XSS
        \n {shock_str}
        \n ({", ".join(par.keys())}) = pars
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


def compile_initial_values(evars, initvals, stst):
    """Combine all available information in initial guesses.
    """

    # get inital values to test the function
    init = np.ones(len(evars)) * 1.1

    if initvals is not None:
        for v in initvals:
            init[evars.index(v)] = initvals[v]

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
            raise Exception("Variable `%s` is not defined for time t." % v)
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
            len(shocks)), np.array(list(par.values()))
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
        model.funcs = load_functions_file(model)
        print("(parse:) Loading cached model.")
        return model

    mdict_raw = deepcopy(model)

    defs = model.get("definitions")
    # never ever use real numpy
    if defs is not None:
        for d in defs:
            d = d.replace(" numpy ", " jax.numpy ")
            # execute these definitions globally (TODO: is this a good idea?)
            exec(d, globals())

    eqns = model["equations"].copy()

    # check if there are dublicate variables
    evars = check_dublicates_and_determinancy(model["variables"], eqns)
    # check if each variable is defined in time t (only defining xSS does not give a valid root)
    check_if_defined(evars, eqns)

    shocks = model.get("shocks") or ()
    par = eval_strs(model["parameters"])
    model["stst"] = stst = eval_strs(model["steady_state"].get("fixed_values"))
    model["root_options"] = {}

    # collect number of foward and backward looking variables
    model["no_fwd"] = sum(
        var + "Prime" in "".join(model["equations"]) for var in evars)
    model["no_bwd"] = sum(var + "Lag" in "".join(model["equations"])
                          for var in evars)

    # collect initial guesses
    model["init"] = init = compile_initial_values(
        evars, eval_strs(model["steady_state"].get("init_guesses")), stst)

    stst_eqns = model["steady_state"].get("equations") or []
    # add fixed values to the steady state equations
    if stst is not None:
        for key in stst:
            stst_eqns.append(key + "SS = " + str(stst[key]))

    # get a string that contains the function definition
    model["func_str"] = func_str = compile_func_str(evars, eqns, par, eqns_aux=model.get(
        'aux_equations'), stst_eqns=stst_eqns, shocks=shocks)

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpf.write(func_str)
    tmpf.close()

    # define the function
    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), globals())
    model["func_raw"] = func_raw

    # TODO: reactivate
    if not testing:
        # try if function works on initvals. If it does, jit-compile it and remove tempfile
        check_func(func_raw, init, shocks, par)
        model["func"] = jax.jit(func_raw, static_argnums=(6, 7))

    # unlink the temporary file
    os.unlink(tmpf.name)

    if verbose:
        print("(parse:) Parsing done.")

    # add new model to cache
    cached_mdicts += (mdict_raw,)
    cached_models += (model,)

    # load file with additional functions as module (if it exists)
    model.funcs = load_functions_file(model)

    return model
