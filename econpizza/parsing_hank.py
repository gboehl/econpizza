#!/bin/python
# -*- coding: utf-8 -*-

import os
from .parsing import parse
from grgrlib import load_as_module


def load(model):
    """load model from dict or yaml file. Warning: contains filthy code (eg. globals, exec, ...)"""

    from .__init__ import PizzaModel

    if isinstance(model, str):
        # store path before it gets overwritten
        full_path = model

        # let model be a model instance
        model = PizzaModel(parse(model))
        # store path in dict
        model['path'] = full_path

    # load file with additional functions as module (if it exists)
    try:
        # prepare path
        if not os.path.isabs(model["functions_file"]):
            yaml_dir = os.path.dirname(full_path)
            model['functions_file'] = os.path.join(
                yaml_dir, model["functions_file"])
        # load as a module
        model.funcs = load_as_module(model['functions_file'])
    except KeyError:
        pass

    return model


def create_grids(distributions):
    """Get the strings of functions that define the grids.
    """

    grid_strings = ()

    for dist_name, dist in distributions.items():
        for grid_name, g in dist.items():

            if g['type'] == 'exogenous':
                # skip this only if none of the parameters is given
                # in this case the grid must be defined in some stage in the yaml
                if not all([i not in g for i in ['rho', 'sigma', 'n']]):
                    grid_strings += f"{g['grid_variables']} = grids.markov_rouwenhorst(rho={g['rho']}, sigma={g['sigma']}, N={g['n']})",

            elif g['type'] == 'endogenous':
                # as above
                if not all([i not in g for i in ['min', 'max', 'n']]):
                    grid_strings += f"{g['grid_variables']} = grids.log_grid(amin={g['min']}, amax={g['max']}, n={g['n']})",

    return grid_strings
