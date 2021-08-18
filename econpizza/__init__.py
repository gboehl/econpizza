#!/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
from .shooting import find_pizza
from .stacking import find_stack
from .parsing import parse, load
from .steady_state import solve_stst, solve_linear

find_path_stacked = find_stack
find_path = find_pizza

np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_nk = os.path.join(pth, "examples", "nk.yaml")
example_bh = os.path.join(pth, "examples", "bh.yaml")
