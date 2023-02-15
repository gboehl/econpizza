# -*- coding: utf-8 -*-

from . import examples
from .solvers.steady_state import solve_stst
from .solvers.stacking import find_path_stacking
from .solvers.solve_linear import find_path_linear
from .solvers.solve_linear_state_space import solve_linear_state_space, find_path_linear_state_space
from .solvers.shooting import find_path_shooting
from .parsing import parse, load

import logging
import os
import jax
import jax.numpy as jnp

# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


__version__ = '0.4.4'

jax.config.update("jax_enable_x64", True)


class PizzaModel(dict):
    def __init__(self, *args, **kwargs):
        super(PizzaModel, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_distributions(self, xst, shock=None):
        """Get all disaggregated variables for a given trajectory of aggregate variables.

        Parameters
        ----------

        self : PizzaModel
            the model instance
        xst : array
            a _full_ trajectory of aggregate variables

        Returns
        -------
        rdict : dict
            a dictionary of the distributions
        """

        shocks = self.get("shocks") or ()
        dist_names = list(self['distributions'].keys())
        decisions_outputs = self['decisions']['outputs']
        x = xst[1:-1].flatten()
        x0 = xst[0]

        # deal with shocks if any
        shock_series = jnp.zeros((len(xst)-2, len(shocks)))
        if shock is not None:
            shock_series = shock_series.at[0,
                                           shocks.index(shock[0])].set(shock[1])

        # get functions and execute
        backwards_sweep = self['context']['backwards_sweep']
        forwards_sweep = self['context']['forwards_sweep']
        decisions_output_storage = backwards_sweep(x, x0, shock_series.T)
        dists_storage = forwards_sweep(decisions_output_storage)

        # store this
        rdict = {oput: decisions_output_storage[i]
                 for i, oput in enumerate(decisions_outputs)}
        rdict.update({oput: dists_storage[i]
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
