econpizza
=========

**Solve nonlinear heterogeneous agent models using tools from machine learning**

.. image:: https://github.com/dfm/emcee/workflows/Tests/badge.svg
    :target: https://github.com/gboehl/econpizza/actions?query=workflow%3ATests
.. image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest
.. image:: https://github.com/gboehl/pydsge/workflows/Continuous%20Integration%20Workflow/badge.svg?branch=master
    :target: https://github.com/gboehl/econpizza/actions
.. image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza

Econpizza is a framework to solve and simulate nonlinear perfect foresight models, with or without heterogeneous agents.
A parser allows to express economic models in a simple, high-level fashion as yaml-files.
Additionally, generic and robust routines for steady state search are provided.
Heavy lifting is done using
`automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `Jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

.. toctree::
   :maxdepth: 2

   readme
   quickstart
   tutorial.ipynb
   method.ipynb
   modules
   indices
