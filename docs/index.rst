
Econpizza is a framework to solve and simulate nonlinear perfect foresight models, with or without heterogeneous agents.
A parser allows to express economic models in a simple, high-level fashion as yaml-files.
Additionally, generic and robust routines for steady state search are provided.

The baseline solver is a Newton-based stacking method in the spirit of Boucekkine (1995), Juillard (1996) and others. Hence, the method is similar to the solver in dynare, but faster and more robust due to the use of automatic differentiation and sparse jacobians. Even perfect-foresight IRFs for large-scale nonlinear models with, e.g., occassionally binding constraints can be computed in less than a second.

The package makes heavy use of `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `Jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

Econpizza can solve nonlinear HANK models. The approach to deal with the distribution is inspired by the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_). Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.

There is a `model parser <https://econpizza.readthedocs.io/en/latest/quickstart.html#the-yaml-file>`_ to allow for the simple and generic specification of models (with or without heterogeneity).

.. toctree::
   :maxdepth: 2

   readme
   quickstart
   tutorial.ipynb
   method.ipynb
   modules
   indices
