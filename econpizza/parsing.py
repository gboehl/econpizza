#!/bin/python
# -*- coding: utf-8 -*-
"""Functions for model parsing yaml into a working model instance. Involves a lot of dynamic function definition...
"""

from copy import copy
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
    model["pars"] = model.get("parameters")
    model["vars"] = model["variables"]

    return model


def eval_strs(vdict, context={}):
    """Evaluate a dictionary of strings into a given context
    """

    if vdict is None:
        return None
    else:
        vdict = vdict.copy()

    for v in vdict:
        if isinstance(vdict[v], str):
            context[v] = eval(vdict[v], context)
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


def initialize_context():
    context = globals().copy()
    default_funcs = {'log': jnp.log,
                     'exp': jnp.exp,
                     'sqrt': jnp.sqrt,
                     'max': jnp.maximum,
                     'min': jnp.minimum}

    context.update(default_funcs)
    return context


def load_external_functions_file(model, context):
    """Load the functions file as a module.
    """

    try:
        # load as a module
        module = load_as_module(model["functions_file"])

        def func_or_compiled(func): return isinstance(
            func, jaxlib.xla_extension.CompiledFunction) or isfunction(func)
        for m in getmembers(module, func_or_compiled):
            context[m[0]] = m[1]

    except KeyError:
        pass

    return False


def compile_init_values(evars, pars, initvals, stst):
    """Combine all available information in initial guesses.
    """

    ufixed_vars = {v: 1.1 for v in evars if v not in stst}
    ufixed_pars = {v: 1.1 for v in pars if v not in stst}

    # get inital values to test the function
    init = {**ufixed_vars, **ufixed_pars}

    # structure: aggregate values first, then values of decisions functions
    if initvals is not None:
        for v in init:
            # assign aggregate values
            if v in initvals:
                init[v] = initvals[v]

    return init


def define_subdict_if_absent(parent, sub):
    try:
        return parent[sub]
    except KeyError:
        parent[sub] = {}
        return parent[sub]


def define_function(func_str, context):
    """Define functions from string. Writes the function into a temporary file in order to get meaningful debug traces.
    """

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpf.write(func_str)
    tmpf.close()

    # define the function
    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), context)

    return tmpf.name


def get_exog_grid_var_names(distributions):
    """WIP. So far unused.
    """
    # NOTE: this will be important when implementing that grid parameters are endogenous variables
    # NOTE: when activated, backward calls already return exogenous grid vars (exog_grid_var). They are not yet stacked, and not yet an input to forward calls

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
    verbose=True
):
    """Load model from dict or yaml file.

    Parameters
    ----------
    model : dict or string
        either a dictionary or the path to a yaml file to be parsed
    raise_errors : bool, optional
        whether to raise errors while checking. False will let the model fail siliently for debugging. Defaults to True
    verbose : bool, optional
        inform that parsing is done. Defaults to True

    Returns
    -------
    model : PizzaModel instance
        The parsed model
    """

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
        print("(parse:) Loading cached model.")
        return model

    model['context'] = initialize_context()
    load_external_functions_file(model, model['context'])

    defs = model.get("definitions")
    defs = '' if defs is None else defs
    defs = '\n'.join(defs) if isinstance(defs, list) else defs
    exec(defs, model['context'])

    eqns = model["equations"].copy()

    # check if there are dublicate variables
    evars = check_dublicates_and_determinancy(model["variables"], eqns)
    # check if each variable is defined in time t (only defining xSS does not give a valid root)
    check_if_defined(evars, eqns, model.get('skip_check_if_defined'))

    # create fixed (time invariant) grids
    grids.create_grids(model.get('distributions'), model["context"])

    shocks = model.get("shocks") or ()
    _ = define_subdict_if_absent(model, "steady_state")
    fixed_values = define_subdict_if_absent(
        model["steady_state"], "fixed_values")
    stst = eval_strs(fixed_values, context=model['context'])
    model["steady_state"]['fixed_evalued'] = stst

    par = define_subdict_if_absent(model, "parameters")
    if isinstance(par, dict):
        raise TypeError(f'parameters must be a list and not {type(par)}.')
    model["parameters"] = par
    model["root_options"] = {}

    # collect number of foward and backward looking variables
    model["no_fwd"] = sum(
        var + "Prime" in "".join(model["equations"]) for var in evars)
    model["no_bwd"] = sum(var + "Lag" in "".join(model["equations"])
                          for var in evars)

    tmpf_names = ()
    # initialize storage for all function strings
    model['func_strings'] = {}

    # NOTE: currently disabled
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
    init_guesses = eval_strs(model["steady_state"].get(
        "init_guesses"), context=model['context'])
    model["init"] = init = compile_init_values(evars, par, init_guesses, stst)

    # get strings that contains the function definitions
    model['func_strings']["func_pre_stst"] = compile_stst_func_str(
        evars, par, stst, init)
    model['func_strings']["func_eqns"] = compile_eqn_func_str(evars, deepcopy(eqns), par, eqns_aux=model.get(
        'aux_equations'), shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    # test if model works. Writing to tempfiles helps to get nice debug traces if not
    tmpf_names += define_function(model['func_strings']
                                  ["func_eqns"], model['context']),
    tmpf_names += define_function(model['func_strings']
                                  ['func_pre_stst'], model['context']),

    # get the initial decision functions
    if model.get('decisions'):
        init_vf_list = [init_guesses[dec_input]
                        for dec_input in model['decisions']['inputs']]  # let us for now assume that this must be present
        model['init_vf'] = jnp.array(init_vf_list)

        # check if initial decision functions and the distribution have same shapes
        check_shapes(model['distributions'], model['init_vf'], dist_names)

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
