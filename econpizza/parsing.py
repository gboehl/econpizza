# -*- coding: utf-8 -*-

"""Functions for model parsing yaml into a working model instance. Involves a lot of dynamic function definition...
"""

from copy import copy
import yaml
import re
import os
import sys
import tempfile
import jax
import jaxlib
import jax.numpy as jnp
import importlib.util as iu
from copy import deepcopy, copy
from inspect import getmembers, isfunction
from jax.experimental.host_callback import id_print as jax_print
from .utilities import grids, dists, interp
from .parser.compile_functions import *
from .parser.checks import *

# initialize model cache
cached_mdicts = ()
cached_models = ()


def _load_as_module(path, add_to_path=True):
    """load a file as a module
    """

    if add_to_path:
        directory = os.path.dirname(path)
        sys.path.append(directory)

    modname = os.path.splitext(os.path.basename(path))[0]
    spec = iu.spec_from_file_location(modname, path)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def parse(mfile):
    """Parse model dictionary from yaml file. This can be desirable if values should be exchanged before loading the model.

    Parameters
    ----------
    mfile : string
        path to a yaml file to be parsed

    Returns
    -------
    mdict : dict
        the parsed yaml as a dictionary
    """

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
    mdict = yaml.safe_load(mtxt)
    # create nice shortcuts
    mdict['path'] = mfile
    mdict["pars"] = mdict.get("parameters")
    mdict["vars"] = mdict["variables"]

    return mdict


def _eval_strs(vdict, context={}):
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


def _parse_external_functions_file(model):
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


def _initialize_context():
    context = globals().copy()
    default_funcs = {'log': jnp.log,
                     'exp': jnp.exp,
                     'sqrt': jnp.sqrt,
                     'max': jnp.maximum,
                     'min': jnp.minimum}

    context.update(default_funcs)
    return context


def _initialize_cache():
    cache = {}
    cache['steady_state'] = ()
    cache['func_pre_stst'] = {}
    cache['steady_state_dicts'] = ()
    return cache


def _load_external_functions_file(model, context):
    """Load the functions file as a module.
    """

    try:
        # load as a module
        module = _load_as_module(model["functions_file"])

        def func_or_compiled(func): return isinstance(
            func, jaxlib.xla_extension.CompiledFunction) or isfunction(func)
        for m in getmembers(module, func_or_compiled):
            context[m[0]] = m[1]

    except KeyError:
        pass

    return False


def _compile_init_values(evars, pars, initvals, stst):
    """Combine all available information in initial guesses.
    """

    ufixed_vars = {v: .95 for v in evars if v not in stst}
    ufixed_pars = {v: .95 for v in pars if v not in stst}

    # get inital values to test the function
    init = {**ufixed_vars, **ufixed_pars}

    # structure: aggregate values first, then values of decisions functions
    if initvals is not None:
        for v in init:
            # assign aggregate values
            if v in initvals:
                init[v] = initvals[v]

    return init


def _define_subdict_if_absent(parent, sub):
    try:
        return parent[sub]
    except KeyError:
        parent[sub] = {}
        return parent[sub]


def _define_function(func_str, context):
    """Define functions from string. Writes the function into a temporary file in order to get meaningful debug traces.
    """

    # use a termporary file to get nice debug traces if things go wrong
    tmpf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpf.write(func_str)
    tmpf.close()

    # define the function
    exec(compile(open(tmpf.name).read(), tmpf.name, "exec"), context)

    return tmpf.name


def load_model(model):

    global cached_mdicts, cached_models

    # load file with additional functions as module (if it exists)
    _parse_external_functions_file(model)
    mdict_raw = deepcopy(model)

    # check if model is already cached
    if model in cached_mdicts:
        model = cached_models[cached_mdicts.index(model)]
        return model, False

    model['context'] = _initialize_context()
    model['cache'] = _initialize_cache()
    _load_external_functions_file(model, model['context'])

    defs = model.get("definitions")
    defs = '' if defs is None else defs
    defs = '\n'.join(defs) if isinstance(defs, list) else defs
    exec(defs, model['context'])

    eqns = model["equations"].copy()

    # check if there are dublicate variables
    check_dublicates(model["variables"])
    check_dublicates(model["parameters"])
    evars = check_determinancy(model["variables"], eqns)
    # check if each variable is defined in time t (only defining xSS does not give a valid root)
    check_if_defined(evars, eqns, model.get('skip_check_if_defined'))

    # create fixed (time invariant) grids
    grids.create_grids(model.get('distributions'), model["context"])

    shocks = model.get("shocks") or ()
    _ = _define_subdict_if_absent(model, "steady_state")

    par = _define_subdict_if_absent(model, "parameters")
    if isinstance(par, dict):
        raise TypeError(f'parameters must be a list and not {type(par)}.')
    model["parameters"] = par

    # initialize storage for all function strings
    model['func_strings'] = {}

    # get function strings for decisions and distributions, if they exist
    if model.get('decisions'):
        decisions_outputs = model['decisions']['outputs']
        decisions_inputs = model['decisions']['inputs']
        model['func_strings']["func_backw"] = compile_backw_func_str(
            evars, par, shocks, decisions_inputs, decisions_outputs, model['decisions']['calls'])
        _define_function(model['func_strings']
                         ['func_backw'], model['context'])
    else:
        decisions_outputs = []
        decisions_inputs = []

    if model.get('distributions'):
        # get names of distributions and the forward functions
        dist_names = list(model['distributions'].keys())
        get_forw_funcs(model)
    else:
        dist_names = []

    # get strings that contains the function definitions
    model['func_strings']["func_eqns"] = compile_eqn_func_str(evars, deepcopy(eqns), par, eqns_aux=model.get(
        'aux_equations'), shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    # test if model works. Writing to tempfiles helps to get nice debug traces if not
    _define_function(model['func_strings']["func_eqns"], model['context'])
    # add new model to cache
    cached_mdicts += (deepcopy(mdict_raw),)
    cached_models += (model,)

    return model, True


def load_stst(model, is_fresh, raise_errors, verbose):

    # check if model is already cached
    stst_dict = deepcopy(model['steady_state'])
    cache = model['cache']
    if stst_dict in model['cache']['steady_state_dicts']:
        model['steady_state'] = cache['steady_state'][cache['steady_state_dicts'].index(
            stst_dict)]
        model['context']['func_pre_stst'] = model['steady_state']['func_pre_stst']
        if verbose:
            print("(load:) Loading cached model.")
        return model

    shocks = model.get("shocks") or ()
    par = model["parameters"]
    evars = model["variables"]
    dist_names = list(model['distributions'].keys()
                      ) if model.get('distributions') else []

    # collect fixed values
    fixed_values = _define_subdict_if_absent(
        model["steady_state"], "fixed_values")
    fixed_evaluated = _eval_strs(fixed_values, context=model['context'])

    # collect initial guesses
    init_guesses = _eval_strs(model["steady_state"].get(
        "init_guesses"), context=model['context'])
    model["steady_state"]["init"] = init = _compile_init_values(
        evars, par, init_guesses, fixed_evaluated)

    # ensure ordering and lenght of fixed values is fine
    fixed_values_names = tuple(
        sorted([k for k in fixed_evaluated if k in par or k in evars]))
    model["steady_state"]['fixed_evalued'] = {
        k: fixed_evaluated[k] for k in fixed_values_names}

    # check, then get strings that contains the definition of func_pre_stst and compile
    try:
        model['context']['func_pre_stst'] = model['cache']['func_pre_stst'][fixed_values_names]
    except KeyError:
        _define_function(compile_stst_func_str(
            evars, par, fixed_values_names, init), model['context'])
        model['cache']['func_pre_stst'][fixed_values_names] = model['context']['func_pre_stst']
    model['steady_state']['func_pre_stst'] = model['context']['func_pre_stst']

    # get the initial decision functions
    if model.get('decisions'):
        init_vf_list = [init_guesses[dec_input]
                        for dec_input in model['decisions']['inputs']]  # for now assume that this must be present
        model["steady_state"]['init_vf'] = jnp.array(init_vf_list)

        # check if initial decision functions and the distribution have same shapes
        check_shapes(model['distributions'],
                     model["steady_state"]['init_vf'], dist_names)

    # try if function works on initvals
    try:
        check_initial_values(model, shocks, par)
    except:
        if raise_errors:
            raise

    model['cache']['steady_state'] += model['steady_state'],
    model['cache']['steady_state_dicts'] += stst_dict,
    model['cache']['reset_stst'] = True

    if verbose and is_fresh:
        print("(load:) Parsing done.")
    elif verbose and not is_fresh:
        print("(load:) Loading cached model with changed fixed or initial values.")

    return model


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
    model : PizzaModel
        The parsed model
    """

    from .__init__ import PizzaModel

    # parse if this is a path to yaml file
    if isinstance(model, str):
        full_path = model
        model = parse(model)
        model['path'] = full_path
    # make it a model
    model = PizzaModel(model)

    # loading model and steady state are separated for efficient caching
    model_stst = model.pop('steady_state')
    # compile or check if model without steady state section is cached
    model, is_fresh = load_model(model)
    # add steady state section again
    model['steady_state'] = model_stst
    # compile check if model with steady state section is cached
    return load_stst(model, is_fresh, raise_errors, verbose)
