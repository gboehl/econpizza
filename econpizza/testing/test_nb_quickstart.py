# -*- coding: utf-8 -*-

import os
import sys
import jax.numpy as jnp
from testbook import testbook
# autopep8: off
sys.path.insert(0, os.path.abspath("."))
import econpizza as ep
# autopep8: on

filepath = os.path.dirname(__file__)

ipynb_path = os.path.join(filepath, '..', '..', 'docs',
                          'tutorial', 'quickstart.ipynb')


@testbook(ipynb_path, execute=True)
def test_this(tb):
    assert True
