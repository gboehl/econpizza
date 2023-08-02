# -*- coding: utf-8 -*-

import logging
import os
import jax
import jax.numpy as jnp
from copy import deepcopy

from . import examples
from .__version__ import __version__
from .solvers.steady_state import solve_stst
from .solvers.stacking import find_path_stacking
from .solvers.solve_linear import find_path_linear
from .solvers.solve_linear_state_space import solve_linear_state_space, find_path_linear_state_space
from .solvers.shooting import find_path_shooting
from .parser import parse, load


# set number of cores for XLA
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

jax.config.update("jax_enable_x64", True)

# create local alias
copy = deepcopy


class PizzaModel(dict):
    """Base class for models. Contains all necessary methods and informations.
    """

    def __init__(self, mdict, *args, **kwargs):
        # do not overwrite original input
        mdict = deepcopy(mdict)
        super(PizzaModel, self).__init__(mdict, *args, **kwargs)
        self.__dict__ = self

    def get_distributions(self, trajectory, init_dist=None, shock=None, pars=None):
        """Get all disaggregated variables for a given trajectory of aggregate variables.

        Note that the output objects do, other than the result from `find_path` with stacking, not include the time-T objects and that the given distribution is as from the beginning of each period.

        Parameters
        ----------

        trajectory : array
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
            self['steady_state'].get('distributions'))
        pars = jnp.array(list(self['pars'].values())
                         ) if pars is None else pars
        shocks = self.get("shocks") or ()
        dist_names = list(self['distributions'].keys())
        decisions_outputs = self['decisions']['outputs']
        x = trajectory[1:-1].flatten()
        x0 = trajectory[0]

        # deal with shocks if any
        shock_series = jnp.zeros((len(trajectory)-2, len(shocks)))
        if shock is not None:
            shock_series = shock_series.at[0,
                                           shocks.index(shock[0])].set(shock[1])

        # get functions and execute
        backwards_sweep = self['context']['backwards_sweep']
        forwards_sweep = self['context']['forwards_sweep']
        decisions_output_storage = backwards_sweep(x, x0, shock_series.T, pars)
        dists_storage = forwards_sweep(decisions_output_storage, dist0)

        # store this
        rdict = {oput: decisions_output_storage[i]
                 for i, oput in enumerate(decisions_outputs)}
        rdict.update({oput: dists_storage[i]
                     for i, oput in enumerate(dist_names)})

        return rdict

    solve_stst = solve_stst
    find_path = find_path_stacking
    find_path_linear = find_path_linear
    find_path_shooting = find_path_shooting
    find_path_linear_state_space = find_path_linear_state_space
    solve_linear_state_space = solve_linear_state_space


logging.basicConfig(level=logging.WARNING)
