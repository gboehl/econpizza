#!/bin/python
# -*- coding: utf-8 -*-

import logging
import jax
import os
import numpy as np
from .shooting import find_pizza
from .stacking import find_stack
from .parsing import parse, load
from .steady_state import solve_stst
from .solve_linear import solve_linear

jax.config.update("jax_enable_x64", True)
# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


class PizzaModel(dict):
    def __init__(self, *args, **kwargs):
        super(PizzaModel, self).__init__(*args, **kwargs)
        self.__dict__ = self


PizzaModel.find_stack = find_stack
PizzaModel.find_path = find_pizza
PizzaModel.solve_stst = solve_stst
PizzaModel.solve_linear = solve_linear

find_path_stacked = find_stack
find_path = find_pizza

np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_nk = os.path.join(pth, "examples", "nk.yaml")
example_bh = os.path.join(pth, "examples", "bh.yaml")
example_dsge = os.path.join(pth, "examples", "med_scale_nk.yaml")
example_hank = os.path.join(pth, "examples", "hank.yaml")
example_hank_labor = os.path.join(pth, "examples", "hank_labor.yaml")
example_hank2 = os.path.join(pth, "examples", "hank2.yaml")
