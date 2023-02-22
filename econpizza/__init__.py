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

    solve_stst = solve_stst
    find_path = find_path_stacking

    def get_distributions(model, init_state, init_dist=None, shock=None):
        """Get all disaggregated variables for a given trajectory of aggregate variables.

        Parameters
        ----------

        model : PizzaModel
            the model instance
        init_state : array
            a _full_ trajectory of aggregate variables
        init_dist : array, optional
            the initial distribution. Defaults to the steady state distribution
        shock : array, optional
            shock in period 0 as in `(shock_name_as_str, shock_size)`. Defaults to no shock

        Returns
        -------
        rdict : dict
            a dictionary of the distributions
        """

        dist0 = jnp.array(init_dist) if init_dist is not None else jnp.array(
            model['steady_state'].get('distributions'))
        shocks = model.get("shocks") or ()
        dist_names = list(model['distributions'].keys())
        decisions_outputs = model['decisions']['outputs']
        x = init_state[1:-1].flatten()
        x0 = init_state[0]

        # deal with shocks if any
        shock_series = jnp.zeros((len(init_state)-2, len(shocks)))
        if shock is not None:
            shock_series = shock_series.at[0,
                                           shocks.index(shock[0])].set(shock[1])

        # get functions and execute
        backwards_sweep = model['context']['backwards_sweep']
        forwards_sweep = model['context']['forwards_sweep']
        decisions_output_storage = backwards_sweep(x, x0, shock_series.T)
        dists_storage = forwards_sweep(decisions_output_storage, dist0)

        # store this
        rdict = {oput: decisions_output_storage[i]
                 for i, oput in enumerate(decisions_outputs)}
        rdict.update({oput: dists_storage[i]
                     for i, oput in enumerate(dist_names)})

        return rdict

    find_path_linear = find_path_linear
    find_path_shooting = find_path_shooting
    find_path_linear_state_space = find_path_linear_state_space
    solve_linear_state_space = solve_linear_state_space


logging.basicConfig(level=logging.WARNING)
