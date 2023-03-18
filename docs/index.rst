Overview: **Econpizza**
=======================

**Econpizza** is a framework to solve and simulate *fully nonlinear* perfect foresight models, with or without heterogeneous agents.
The package implements the solution method proposed in `HANK on Speed: Robust Nonlinear Solutions using Automatic Differentiation <https://gregorboehl.com/live/hank_speed_boehl.pdf>`_ *(Gregor Boehl, 2023)*.
It allows to specify and solve economic models in a simple, high-level fashion.

The package builds heavily on `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.
A central philosophy is to consequently separate the low-level routines for *model solution* (which is what happens under the hood) from
*model specification* (via a ``yaml`` file) and the
high-level interface for *model simulation and analysis* (what the user does with the model).

The package can solve nonlinear models with heterogeneous households or firms with one or two assets and portfolio choice. Steady state and nonlinear impulse responses (including, e.g., the ZLB) can typically be found within a few seconds.
It not only allows to study the dynamics of aggregate variables, but also the complete nonlinear transition dynamics of the cross-sectional distribution of assets and disaggregated objects. Routines for models with a representative agents are also provided. These are faster and more reliable than the extended path method in dynare due to the use of automatic differentiation for the efficient jacobian decompositions during each Newton-step. Nonlinear perfect-foresight transition dynamics can - even for large-scale nonlinear models with several occassionally binding constraints - be computed in less than a second.

References
----------

Please cite with

.. code-block::

    @Misc{boehl2022pizza,
    title         = {HANK on Speed: Robust Nonlinear Solutions using Automatic Differentiation},
    author        = {Boehl, Gregor},
    howpublished  = {\url{https://gregorboehl.com/live/hank_speed_boehl.pdf}},
    year = {2023}
    }
