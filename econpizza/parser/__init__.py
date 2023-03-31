# -*- coding: utf-8 -*-

"""Functions for model parsing yaml into a working model instance. Involves a lot of dynamic function definition...
"""

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
from .write_dynamic_functions import *
from .checks import *
from ..utilities import grids, dists, interp

# initialize model cache
cached_mdicts = ()
cached_models = ()


def d2jnp(x): return jnp.array(list(x.values()))


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
    mdict["vars"] = mdict["variables"]
    # load file with additional functions as module (if it exists)
    _parse_external_functions_file(mdict)

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
    cache['steady_state_keys'] = ()
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


def _get_pre_stst_mapping(init_vals, fixed_values, evars, par_names):
    """Define the mapping from init & fixed vals to model variables & parameters
    """
    x2var = jnp.zeros((len(evars), len(init_vals)))
    x2par = jnp.zeros((len(par_names), len(init_vals)))
    fixed2var = jnp.zeros((len(evars), len(fixed_values)))
    fixed2par = jnp.zeros((len(par_names), len(fixed_values)))

    for i, v in enumerate(init_vals):
        if v in evars:
            x2var = x2var.at[evars.index(v), i].set(1)
        else:
            x2par = x2par.at[par_names.index(v), i].set(1)
    for i, v in enumerate(fixed_values):
        if v in evars:
            fixed2var = fixed2var.at[evars.index(v), i].set(1)
        else:
            fixed2par = fixed2par.at[par_names.index(v), i].set(1)

    return x2var, x2par, fixed2var, fixed2par


def compile_stst_inputs(model):

    par_names = model["parameters"]
    evars = model["variables"]
    context = model["context"]
    # remove old values from model context
    [context.pop(key) for key in par_names if key in context]
    [context.pop(key) for key in evars if key in context]

    # collect fixed values and ensure ordering and lenght is fine
    fixed_values = _define_subdict_if_absent(
        model["steady_state"], "fixed_values")
    fixed_values_evaluated = _eval_strs(fixed_values, context=context)
    fixed_values_names = tuple(
        sorted([k for k in fixed_values_evaluated if k in par_names or k in evars]))
    fixed_evaluated = {
        k: fixed_values_evaluated[k] for k in fixed_values_names}

    # collect initial guesses
    init_guesses = _eval_strs(model["steady_state"].get(
        "init_guesses"), context=context)
    init_vals = _compile_init_values(
        evars, par_names, init_guesses, fixed_evaluated)

    # get the initial decision functions
    if model.get('distributions'):
        dist_names = list(model['distributions'].keys())
        # for now assume that this must be present
        init_wf = jnp.array([init_guesses[dec_input]
                            for dec_input in model['decisions']['inputs']])
        # allow `n` in distributions to be a variable from the context (so far only for one distribution)
        for d in model['distributions'][dist_names[0]].values():
            if isinstance(d['n'], str):
                d['n'] = eval(d['n'], context)
        # check if initial decision functions and the distribution have same shapes
        check_shapes(model['distributions'], init_wf, dist_names)
    else:
        init_wf = jnp.array(None)

    # define func_pre_stst
    mapping = _get_pre_stst_mapping(
        init_vals, fixed_evaluated, evars, par_names)

    return init_vals, fixed_evaluated, init_wf, mapping


def load(
    model,
    raise_errors=True,
    verbose=True
):
    """Load a model from a dictionary or a YAML file.

    Parameters
    ----------
    model : dict or string
        either a dictionary or the path to a YAML file to be parsed
    raise_errors : bool, optional
        whether to raise errors while checking. False will let the model fail siliently for debugging. Defaults to True
    verbose : bool, optional
        inform that parsing is done. Defaults to True

    Returns
    -------
    model : PizzaModel
        The parsed model
    """

    global cached_mdicts, cached_models
    from ..__init__ import PizzaModel

    # parse if this is a path to yaml file
    if isinstance(model, str):
        full_path = model
        model = parse(model)
        model['path'] = full_path

    # define the model dictionary as key for cached models
    mdict = deepcopy(model)
    stst_subdict = mdict.pop('steady_state') if 'steady_state' in mdict else {}
    # check if model is already cached
    if mdict in cached_mdicts:
        model = copy(cached_models[cached_mdicts.index(mdict)])
        # always use the fresh current stst sec from yaml
        model['steady_state'] = stst_subdict
        if verbose:
            print("(load:) Loading cached model.")
        return model

    # make it a model
    model = PizzaModel(model)
    # initialize objects
    model['context'] = _initialize_context()
    model['cache'] = _initialize_cache()
    _load_external_functions_file(model, model['context'])

    # compile definitions
    defs = model.get("definitions")
    defs = '' if defs is None else defs
    defs = '\n'.join(defs) if isinstance(defs, list) else defs
    exec(defs, model['context'])
    # get aggregate equations
    eqns = model["equations"].copy()

    # check if there are dublicate variables
    check_dublicates(model["variables"])
    check_dublicates(model.get("parameters"))
    evars = check_determinancy(model["variables"], eqns)
    # check if each variable is defined in time t (only defining xSS does not give a valid root)
    check_if_defined(evars, eqns, model.get('decisions'),
                     model.get('skip_check_if_defined'))

    # create fixed (time invariant) grids
    grids.create_grids(model.get('distributions'), model["context"], verbose)
    shocks = model.get("shocks") or ()

    # initialize storages
    _ = _define_subdict_if_absent(model, "func_strings")
    _ = _define_subdict_if_absent(model, "steady_state")
    par_names = _define_subdict_if_absent(model, "parameters")
    if isinstance(par_names, dict):
        raise TypeError(
            f'parameters must be a list and not {type(par_names)}.')

    # get function strings for decisions and distributions, if they exist
    if model.get('decisions'):
        decisions_outputs = model['decisions']['outputs']
        decisions_inputs = model['decisions']['inputs']
        model['func_strings']["func_backw"] = compile_backw_func_str(
            evars, par_names, shocks, decisions_inputs, decisions_outputs, model['decisions']['calls'])
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
    model['func_strings']["func_eqns"] = compile_eqn_func_str(evars, deepcopy(eqns), par_names, eqns_aux=model.get(
        'aux_equations'), shocks=shocks, distributions=dist_names, decisions_outputs=decisions_outputs)

    # writing to tempfiles helps to get nice debug traces if the model does not work
    _define_function(model['func_strings']["func_eqns"], model['context'])
    # compile fixed and initial values
    stst_inputs = compile_stst_inputs(model)
    # try if function works on initvals
    try:
        check_initial_values(model, shocks, *stst_inputs)
    except:
        if raise_errors:
            raise
    # add new model to cache
    cached_mdicts += (mdict,)
    cached_models += (model,)

    if verbose:
        print("(load:) Parsing done.")

    return model
