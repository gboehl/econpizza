#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import re
import os
import tempfile
import numpy as np
import cloudpickle as cpickle
from copy import deepcopy
from numpy import log, exp, sqrt
from numba import njit
from .steady_state import solve_stst, solve_linear

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


def load(model, raise_errors=True, use_ndifftools=True, lti_max_iter=500, verbose=True):
    """load model from dict or yaml file. Warning: contains filthy code (eg. globals, exec, ...)"""

    global cached_mdicts, cached_models

    if isinstance(model, str):
        model = parse(model)

    if model in cached_mdicts:
        model = cpickle.loads(cached_models[cached_mdicts.index(model)])
        print("(parse:) Loading cached model.")
        return model

    mdict_raw = deepcopy(model)

    defs = model.get("definitions")
    if defs is not None:
        for d in defs:
            exec(d, globals())

    evars = model["variables"]
    dubs = [x for i, x in enumerate(evars) if x in evars[:i]]
    dubmess = (
        ", variables list contains dublicate(s): %s" % ", ".join(dubs) if dubs else ""
    )

    evars = model["variables"][:] = sorted(list(set(evars)), key=str.lower)
    eqns = model["equations"].copy()

    if len(evars) != len(eqns):
        raise Exception(
            "Model has %s variables but %s equations%s."
            % (len(evars), len(eqns), dubmess)
        )
    elif dubs:
        print("(parse:) Warning%s" % dubmess)

    shocks = model.get("shocks") or ()
    par = eval_strs(model["parameters"])

    initvals = eval_strs(model["steady_state"].get("init_guesses"))
    stst = eval_strs(model["steady_state"].get("fixed_values"))
    model["stst"] = stst

    # collect number of foward and backward looking variables
    model["no_fwd"] = sum(var + "Prime" in "".join(model["equations"]) for var in evars)
    model["no_bwd"] = sum(var + "Lag" in "".join(model["equations"]) for var in evars)

    # check if each variable is defined in time t (only defining xSS does not give a valid root)
    for v in evars:
        v_in_eqns = [
            v in e.replace(v + "SS", "").replace(v + "Lag", "").replace(v + "Prime", "")
            for e in eqns
        ]
        if not np.any(v_in_eqns):
            raise Exception("Variable `%s` is not defined for time t." % v)

    # start compiling root_container
    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = "root_container[%s] = " % i + lhs + "- (" + rhs + ")"
        else:
            eqns[i] = "root_container[%s] = " % i + eqn

    eqns_aux = model.get("aux_equations")
    stst_eqns = model["steady_state"].get("equations") or []

    # resolve all equations with the 'All' keyword
    for eqn in stst_eqns:
        if eqn.count("All") == 1:
            for vtype in ("", "SS", "Lag", "Prime"):
                stst_eqns.append(eqn.replace("All", vtype))
        elif eqn.count("All") > 1:
            raise NotImplementedError(
                "Multiple `All` in one equation are not implemented"
            )

    # add fixed values to the steady state equations
    if stst is not None:
        for key in stst:
            stst_eqns.append(key + "SS = " + str(stst[key]))

    if not shocks:
        shock_str = ""
    elif len(shocks) > 1:
        shock_str = ", ".join(shocks) + " = shocks"
    else:
        shock_str = shocks[0] + " = shocks[0]"

    func_str = f"""def func_raw(XLag, X, XPrime, XSS, shocks, pars, stst=False, return_stst_vals=False):
        \n {", ".join(v + "Lag" for v in evars) + " = XLag"}
        \n {", ".join(evars) + " = X"}
        \n {", ".join(v + "Prime" for v in evars) + " = XPrime"}
        \n {", ".join(v + "SS" for v in evars) + " = XSS"}
        \n {shock_str}
        \n {", ".join(par.keys()) + " = pars"}
        \n root_container=np.empty({len(evars)})
        \n %s \n %s\n %s\n %s
        \n return root_container""" % (
        "if stst:\n  " + "\n  ".join(stst_eqns) if stst_eqns else "",
        "\n ".join(eqns_aux) if eqns_aux else "",
        "if return_stst_vals:\n  "
        + "return np.array(("
        + ", ".join(v for v in evars)
        + "))",
        "\n ".join(eqns),
    )

    # get inital values to test the function
    init = np.ones(len(evars)) * 1.1

    if isinstance(initvals, dict):
        for v in initvals:
            init[evars.index(v)] = initvals[v]

    if stst:
        for v in stst:
            init[evars.index(v)] = stst[v]

    model["init"] = init

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)

    tmpf.write(func_str)
    tmpf.close()

    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), globals())

    # try if function works on initvals. If it does, jit-compile it and remove tempfile
    test = func_raw(
        init, init, init, init, np.zeros(len(shocks)), np.array(list(par.values()))
    )
    if np.isnan(test).any():
        raise Exception("Initial values are NaN.")
    if np.isinf(test).any():
        raise Exception("Initial values are not finite.")

    func = njit(func_raw)
    os.unlink(tmpf.name)

    model["func"] = func
    model["func_str"] = func_str
    model["root_options"] = {}

    if verbose:
        print("(parse:) Parsing done.")

    solve_stst(model, raise_error=raise_errors, verbose=verbose)
    solve_linear(
        model,
        raise_error=raise_errors,
        use_ndifftools=use_ndifftools,
        lti_max_iter=lti_max_iter,
        verbose=verbose,
    )

    cached_mdicts += (mdict_raw,)
    cached_models += (cpickle.dumps(model, protocol=4),)

    return model
