#!/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
from .shooting import parse, find_path, solve_stst

np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.INFO)

pth = os.path.dirname(__file__)

example_nk = os.path.join(pth, "examples", "nk.yaml")
example_bh = os.path.join(pth, "examples", "bh.yaml")
