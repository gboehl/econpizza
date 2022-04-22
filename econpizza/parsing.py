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
import jax.numpy as jnp
from copy import deepcopy
from jax.numpy import log, exp, sqrt, maximum, minimum
from grgrlib import load_as_module
from inspect import getmembers, isfunction
from jax.experimental.host_callback import id_print as jax_print
from .steady_state import solve_stst, solve_linear
from .utilities import grids, dists
from .parser.compile_functions import *
from .parser.checks import *

jax.config.update("jax_enable_x64", True)
# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


# initialize model cache
cached_mdicts = ()
cached_models = ()


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


def compile_init_values(evars, decisions_inputs, initvals, stst):
    """Combine all available information in initial guesses.
    """

    # get inital values to test the function
    init = jnp.ones(len(evars)) * 1.1

    # structure: aggregate values first, then values of decisions functions
    if initvals is not None:
        for v in initvals:
            # assign aggregate values
            if v in evars:
                init = init.at[evars.index(v)].set(initvals[v])

    if stst:
        for v in stst:
            init = init.at[evars.index(v)].set(stst[v])

    return init


def define_function(func_str, context):

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpf.write(func_str)
    tmpf.close()

    # define the function
    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), context)

    return tmpf.name


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

    stst_eqns = model["steady_state"].get("equations") or []
    # add fixed values to the steady state equations
    if stst is not None:
        for key in stst:
            # setting ALL occurences of the variable should be fine since we are using pinv later
            stst_eqns.append(f"{key} = {stst[key]}")

    tmpf_names = ()

    # get function strings for decisions and distributions, if they exist
    if model.get('decisions'):
        decisions_outputs = model['decisions']['outputs']
        decisions_inputs = model['decisions']['inputs']
        model["func_backw_str"] = compile_backw_func_str(
            evars, par, shocks, model['decisions']['inputs'], decisions_outputs, model['decisions']['calls'])
        tmpf_names += define_function(model['func_backw_str'],
                                      model['context']),
    else:
        decisions_outputs = []
        decisions_inputs = []

    if model.get('distributions'):
        dist_names = list(model['distributions'].keys())
        model["func_stst_dist_str"], model["func_dist_str"] = compile_func_dist_str(
            model['distributions'], decisions_outputs)
        tmpf_names += define_function(
            model['func_stst_dist_str'], model['context']),
        tmpf_names += define_function(model['func_dist_str'],
                                      model['context']),

    else:
        dist_names = []

    # collect initial guesses
    model["init"] = compile_init_values(evars, decisions_inputs, eval_strs(
        model["steady_state"].get("init_guesses")), stst)

    # get strings that contains the function definitions
    model["func_pre_stst_str"] = compile_stst_func_str(
        evars, eqns, par, stst_eqns)
    model["func_str"] = compile_eqn_func_str(evars, eqns, par, eqns_aux=model.get(
        'aux_equations'), shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    tmpf_names += define_function(model["func_str"], model['context']),
    tmpf_names += define_function(model['func_pre_stst_str'],
                                  model['context']),

    # test if model works. Writing to tempfiles helps to get nice debug traces if not
    if not model.get('decisions'):
        # try if function works on initvals
        check_func(model['context']['func_raw'], model["init"], shocks, par)
        # TODO: also test other functions

    model['func'] = jax.jit(model['context']['func_raw'])
    # TODO: good idea to jit here and not later (or not at all)?

    # unlink the temporary files
    for tmpf in tmpf_names:
        os.unlink(tmpf)

    if verbose:
        print("(load:) Parsing done.")

    # add new model to cache
    cached_mdicts += (mdict_raw,)
    cached_models += (model,)

    return model
