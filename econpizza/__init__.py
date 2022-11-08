#!/bin/python
# -*- coding: utf-8 -*-

import logging
import jax
import os
from .parsing import parse, load
from .steady_state import solve_stst
from .solvers.shooting import find_path_shooting
from .solvers.stacking import find_path_stacking
from .solvers.solve_linear import find_path_linear
from .solvers.solve_linear_state_space import *

__version__ = '0.2.4'

jax.config.update("jax_enable_x64", True)
# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


class PizzaModel(dict):
    def __init__(self, *args, **kwargs):
        super(PizzaModel, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_het_vars(self, xst):
        """Get all disaggregated variables for a given trajectory of aggregate variables.

        Parameters
        ----------

        self : PizzaModel
            the model instance
        xst : array
            a trajectory of aggregate variables

        Returns
        -------
        rdict : dict
            a dictionary of the outputs of the decision and distributions stage
        """

        stacked_func = self['context']['stacked_func_dist']
        decisions_outputs = self['decisions']['outputs']
        dist_names = list(self['distributions'].keys())

        het_vars = stacked_func(xst[1:-1].ravel(), True)

        rdict = {oput: het_vars[0][i]
                 for i, oput in enumerate(decisions_outputs)}
        rdict.update({oput: het_vars[1][i]
                     for i, oput in enumerate(dist_names)})

        return rdict


PizzaModel.solve_stst = solve_stst
PizzaModel.solve_linear_state_space = solve_linear_state_space
PizzaModel.find_path = find_path_stacking
PizzaModel.find_path_stacking = find_path_stacking
PizzaModel.find_path_linear = find_path_linear
PizzaModel.find_path_linear_state_space = find_path_linear_state_space
PizzaModel.find_path_shooting = find_path_shooting

logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_nk = os.path.join(pth, "examples", "nk.yaml")
example_bh = os.path.join(pth, "examples", "bh.yaml")
example_dsge = os.path.join(pth, "examples", "med_scale_nk.yaml")
example_hank = os.path.join(pth, "examples", "hank.yaml")
example_hank_labor = os.path.join(pth, "examples", "hank_labor.yaml")
example_hank2 = os.path.join(pth, "examples", "hank2.yaml")
