econpizza
=========
.. |badge0| image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. |badge1| image:: https://github.com/gboehl/econpizza/actions/workflows/continuous-integration.yml/badge.svg
    :target: https://github.com/gboehl/econpizza/actions
.. |badge2| image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest
.. |badge3| image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza

|badge0| |badge1| |badge2| |badge3|

**Solve nonlinear heterogeneous agent models using automatic differentiation**

Econpizza is a framework to solve and simulate **fully nonlinear** perfect foresight models, with or without heterogeneous agents.
The package implements the solution method proposed in `HANK on Speed: Robust Nonlinear Solutions using Automatic Differentiation <https://gregorboehl.com/live/hank_speed_boehl.pdf>`_ *(Gregor Boehl, 2023, SSRN No. 4433585)*.
It allows to specify and solve nonlinear macroeconomic models quickly in a simple, high-level fashion and provides generic and robust routines for steady state search.

The package can solve nonlinear models with heterogeneous agents, such as HANK models with one or two assets and portfolio choice. Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.
The method extends the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_) to fully nonlinear heterogeneous agent models models by iteratively using `Jacobian-vector producs <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions>`_ to approximate the solution to the linear system of equations associated with each Newton iteration. This not only allows to study the dynamics of aggregate variables, but also the complete nonlinear transition dynamics of the cross-sectional distribution of assets and disaggregated objects.

To solve models with representative agents a shooting methods similar to Laffargue (1990), Boucekkine (1995) and Juillard (1996) is implemented. It is faster and more reliable than the extended path method in dynare due to the use of automatic differentiation for the efficient jacobian decompositions during each Newton-step. Nonlinear perfect-foresight transition dynamics can - even for large-scale nonlinear models with several occassionally binding constraints - be computed in less than a second.

The package builds heavily on `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.
There is a `presentation <https://gregorboehl.com/revealjs/adh/index.html>`_ on how this works behind the szenes.


Documentation
-------------

▶ `User guide <https://econpizza.readthedocs.io/en/stable/index.html>`_
▶ `Quickstart tutorial <https://econpizza.readthedocs.io/en/stable/tutorial/quickstart.html>`_

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as typing

.. code-block:: bash

   pip install econpizza

in your terminal or Anaconda Prompt.

Citation
--------
.. code-block:: bibtex

    @article{boehl2023goodpizza,
        title       = {HANK on Speed: Robust Nonlinear Solutions using Automatic Differentiation},
        author      = {Boehl, Gregor},
        journal     = {Available at SSRN 4433585},
        year        = {2023}
    }
