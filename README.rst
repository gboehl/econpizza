econpizza
=========

**Solve nonlinear heterogeneous agent models using automatic differentiation**

.. image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. image:: https://github.com/dfm/emcee/workflows/Tests/badge.svg
    :target: https://github.com/gboehl/econpizza/actions
.. image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest
.. image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza

**Econpizza** is a framework to solve and simulate *fully nonlinear* perfect foresight models, with or without heterogeneous agents.
The package implements the solution method proposed in `Robust Nonlinear Transition Dynamics in HANK <https://gregorboehl.com/live/hank_speed_boehl.pdf>`_ *(Gregor Boehl, 2023)*.
It allows to specify and solve nonlinear macroeconomic models quickly in a simple, high-level fashion.
Generic and robust routines for steady state search are provided.

The package can solve nonlinear models with heterogeneous agents, such as HANK models with one or two assets and portfolio choice. Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.
The method extends the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_) to fully nonlinear heterogeneous agent models models by iteratively using `Jacobian-vector producs <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions>`_ to approximate the solution to the linear system of equations associated with each Newton iteration. This not only allows to study the dynamics of aggregate variables, but also the complete nonlinear transition dynamics of the cross-sectional distribution of assets and disaggregated objects.

To solve models with representative agents a shooting methods similar to Laffargue (1990), Boucekkine (1995) and Juillard (1996) is implemented. It is faster and more reliable than the extended path method in dynare due to the use of automatic differentiation for the efficient jacobian decompositions during each Newton-step. Nonlinear perfect-foresight transition dynamics can - even for large-scale nonlinear models with several occassionally binding constraints - be computed in less than a second.

The package builds heavily on `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.


Documentation
-------------

Guides and tutorials are provided on ReadTheDocs:

 * `Installation <https://econpizza.readthedocs.io/en/stable/guide/installation.html>`_
 * `User guide <https://econpizza.readthedocs.io/en/stable/index.html>`_
 * `Quickstart tutorial <https://econpizza.readthedocs.io/en/stable/tutorial/quickstart.html>`_

Citation
--------
.. code-block:: bibtex

    @Misc{boehl2022pizza,
    title         = {Robust Nonlinear Transition Dynamics in HANK},
    author        = {Boehl, Gregor},
    howpublished  = {\url{https://gregorboehl.com/live/hank_speed_boehl.pdf}},
    year = {2023}
    }
