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
