#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import re
import numpy as np
from numpy import log, exp, sqrt
from numba import njit
from .steady_state import solve_stst, solve_linear


def parse(mfile, raise_errors=True, verbose=True):

    f = open(mfile)
    mtxt = f.read()
    f.close()

    mtxt = mtxt.replace("^", "**")
    mtxt = re.sub(r"@ ?\n", " ", mtxt)
    model = yaml.safe_load(mtxt)

    defs = model.get("definitions")
    if defs is not None:
        for d in defs:
            exec(d, globals())

    evars = model["variables"]
    shocks = model.get("shocks") or ()
    par = model["parameters"]
    eqns = model["equations"]

    if len(evars) != len(eqns):
        raise Exception(
            "Model has %s variables but %s equations." % (len(evars), len(eqns))
        )

    # collect number of foward and backward looking variables
    model["no_fwd"] = sum(var + "Prime" in "".join(model["equations"]) for var in evars)
    model["no_bwd"] = sum(var + "Lag" in "".join(model["equations"]) for var in evars)

    # start compiling F
    for i, eqn in enumerate(eqns):
        if "=" in eqn:
            lhs, rhs = eqn.split("=")
            eqns[i] = "F[%s] = " % i + lhs + "- (" + rhs + ")"
        else:
            eqns[i] = "F[%s] = " % i + eqn

    eqns_aux = model.get("aux_equations")

    if not shocks:
        shock_str = ""
    elif len(shocks) > 1:
        shock_str = ", ".join(shocks) + " = shocks"
    else:
        shock_str = shocks[0] + " = shocks[0]"

    func_str = """def func_raw(XLag, X, XPrime, XSS, shocks, pars):\n %s\n %s\n %s\n %s\n %s\n %s\n F=np.empty(%s)\n %s\n %s\n return F""" % (
        ", ".join(v + "Lag" for v in evars) + " = XLag",
        ", ".join(evars) + " = X",
        ", ".join(v + "Prime" for v in evars) + " = XPrime",
        ", ".join(v + "SS" for v in evars) + " = XSS",
        shock_str,
        ", ".join(par.keys()) + " = pars",
        str(len(evars)),
        "\n ".join(eqns_aux) if eqns_aux else "",
        "\n ".join(eqns),
    )

    try:
        exec(func_str, globals())
        func = njit(func_raw)
    except Exception as error:
        raise type(error)(
            str(error)
            + "\n\n This is the transition function as I tried to compile it:\n\n"
            + func_str
        )

    model["func"] = func
    model["func_str"] = func_str
    model["root_options"] = {}

    if verbose:
        print("Parsing done.")

    solve_stst(model, raise_error=raise_errors, verbose=verbose)
    solve_linear(model, raise_error=raise_errors, verbose=verbose)

    return model
