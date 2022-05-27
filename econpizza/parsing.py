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
# from jax.numpy import log, exp, sqrt, maximum, minimum # maybe better to make this explicit
from grgrlib import load_as_module
from inspect import getmembers, isfunction
from jax.experimental.host_callback import id_print as jax_print
from .utilities import grids, dists, interp
from .parser.compile_functions import *
from .parser.checks import *

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
    model['path'] = mfile
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
        if isinstance(vdict[v], str):
            context[v] = eval(vdict[v])
            vdict[v] = eval(v, context)
        else:
            context[v] = vdict[v]

    return vdict


def parse_external_functions_file(model):
    """Parse the functions file.
    """

    try:
        # prepare path
        if not os.path.isabs(model["functions_file"]):
            yaml_dir = os.path.dirname(model["path"])
            model["functions_file"] = os.path.join(
                yaml_dir, model["functions_file"])

        # store content
        f = open(model["functions_file"])
        model['functions_file_plain'] = f.read()
        f.close()

    except KeyError:
        pass

    return


def load_external_functions_file(model, context):
    """Load the functions file as a module.
    """

    try:
        # load as a module
        context['module'] = load_as_module(model["functions_file"])

        def func_or_compiled(func): return isinstance(
            func, jaxlib.xla_extension.CompiledFunction) or isfunction(func)
        for m in getmembers(module, func_or_compiled):
            exec(f'{m[0]} = module.{m[0]}', context)

    except KeyError:
        pass

    return False


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


def get_exog_grid_var_names(distributions):
    # TODO: this will be important when implementing that grid parameters are endogenous variables
    # TODO: when activated, backward calls already return exogenous grid vars (exog_grid_var). They are not yet stacked, and not yet an input to forward calls

    exog_grid_var_names = ()

    if False:
        # if distributions:
        for dist_name in distributions:

            dist = distributions[dist_name]
            for v in dist:
                if dist[v]['type'] in ('exogenous', 'custom_exogenous'):
                    exog_grid_var_names += tuple(dist[v]['grid_variables'])

    return exog_grid_var_names


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

    # load file with additional functions as module (if it exists)
    parse_external_functions_file(model)
    mdict_raw = deepcopy(model)

    # check if model is already cached
    if model in cached_mdicts:
        model = cached_models[cached_mdicts.index(model)]
        model['context'] = globals()
        load_external_functions_file(model, model['context'])
        print("(parse:) Loading cached model.")
        return model

    model['context'] = globals()
    load_external_functions_file(model, model['context'])

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
    check_if_defined(evars, eqns, model.get('skip_check_if_defined'))

    # create fixed (time invariant) grids
    grids.create_grids(model.get('distributions'), model["context"])

    shocks = model.get("shocks") or ()
    par = eval_strs(model["parameters"])
    model["steady_state"]['fixed_evalued'] = stst = eval_strs(
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
    # initialize storage for all function strings
    model['func_strings'] = {}

    # TODO: currently disabled
    exog_grid_var_names = get_exog_grid_var_names(model.get('distributions'))

    # get function strings for decisions and distributions, if they exist
    if model.get('decisions'):
        decisions_outputs = model['decisions']['outputs']
        decisions_inputs = model['decisions']['inputs']
        model['func_strings']["func_backw"] = compile_backw_func_str(
            evars, par, shocks, decisions_inputs, decisions_outputs, model['decisions']['calls'], exog_grid_var_names)
        tmpf_names += define_function(model['func_strings']
                                      ['func_backw'], model['context']),
    else:
        decisions_outputs = []
        decisions_inputs = []

    if model.get('distributions'):
        dist_names = list(model['distributions'].keys())
        func_stst_dist_str, func_dist_str = compile_func_dist_str(
            model['distributions'], decisions_outputs)
        # store both strings
        model['func_strings']["func_stst_dist"] = func_stst_dist_str
        model['func_strings']["func_dist"] = func_dist_str
        # execute them
        tmpf_names += define_function(func_stst_dist_str, model['context']),
        tmpf_names += define_function(func_dist_str, model['context']),
    else:
        dist_names = []

    # collect initial guesses
    model["init"] = compile_init_values(evars, decisions_inputs, eval_strs(
        model["steady_state"].get("init_guesses")), stst)

    # get strings that contains the function definitions
    model['func_strings']["func_pre_stst"] = compile_stst_func_str(
        evars, eqns, par, stst_eqns)
    model['func_strings']["func_eqns"] = compile_eqn_func_str(evars, deepcopy(eqns), par, eqns_aux=model.get(
        'aux_equations'), shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    # test if model works. Writing to tempfiles helps to get nice debug traces if not
    tmpf_names += define_function(model['func_strings']
                                  ["func_eqns"], model['context']),
    tmpf_names += define_function(model['func_strings']
                                  ['func_pre_stst'], model['context']),

    # get the initial decision functions
    if model.get('decisions'):
        init_vf_list = [model['steady_state']['init_guesses'][dec_input]
                        for dec_input in model['decisions']['inputs']]  # let us for now assume that this must be present
        model['init_vf'] = jnp.array(init_vf_list)

        # check if initial decision functions and the distribution have same shapes
        dist_shape = tuple(
            [d['n'] for d in model['distributions'][dist_names[0]].values()])
        decisions_shape = model['init_vf'].shape
        if decisions_shape[-len(dist_shape):] != dist_shape:
            raise Exception(
                f"Initial decision and the distribution have different shapes: {decisions_shape}, {dist_shape}")

    # try if function works on initvals
    model['init_run'] = {}
    try:
        check_initial_values(model, shocks, par)
    except:
        if raise_errors:
            raise

    if verbose:
        print("(load:) Parsing done.")

    # add new model to cache
    cached_mdicts += (mdict_raw,)
    cached_models += (model,)

    return model
