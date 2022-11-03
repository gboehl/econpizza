econpizza
=========

**Solve nonlinear heterogeneous agent models using machine learning techniques**

.. image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. image:: https://github.com/dfm/emcee/workflows/Tests/badge.svg
    :target: https://github.com/gboehl/econpizza/actions
.. image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest
.. image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza

Econpizza is a framework to solve and simulate nonlinear perfect foresight models, with or without heterogeneous agents.
A parser allows to express economic models in a simple, high-level fashion as yaml-files.
Generic and robust routines for steady state search are provided.

The baseline solver is a Newton-based stacking method in the spirit of Boucekkine (1995), Juillard (1996) and others. Hence, the method is similar to the solver in dynare, but faster and more robust due to the use of automatic differentiation and sparse jacobians. Even perfect-foresight IRFs for large-scale nonlinear models with, e.g., occassionally binding constraints can be computed in less than a second.

The package makes heavy use of `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

Econpizza can solve nonlinear models with heterogeneous agents, including HANK models. The approach to deal with heterogeneity is inspired by the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_). Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.

The philosophy behind this package is to consequently separate **model specification** (via a ``yaml`` file), a high-level interface for **model simulation and analysis**, and the low-level routines for **model solution** (which is what happens under the hood).

A `model parser <https://econpizza.readthedocs.io/en/latest/quickstart.html#the-yaml-file>`_ allows for the simple and generic specification of models (with or without heterogeneity) in ``yaml`` format.


Documentation
-------------

The documentation and some **tutorials** can be found `here <https://econpizza.readthedocs.io/en/latest/quickstart.html>`_.


Installation
------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as typing

.. code-block:: bash

   pip install econpizza

in your terminal or Anaconda Prompt. Alternatively, the most recent version from GitHub with some experimental features can be installed via

.. code-block:: bash

   pip install git+https://github.com/gboehl/grgrlib
   pip install git+https://github.com/gboehl/econpizza

Note that the latter requires `git <https://www.activestate.com/resources/quick-reads/pip-install-git/#:~:text=To%20install%20Git%20for%20Windows,installer%20and%20follow%20the%20steps.>`_ to be installed.

Installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^
Econpizza needs **JAX** to be installed. This is not a problem for MacOS and Linux, but the time for JAX to fully support Windows has not yet come. Fortunately, there is help out there (see `this link <https://github.com/cloudhan/jax-windows-builder>`_ for the somewhat cryptic original reference). To install JAX, run

.. code-block:: bash

    pip install "jax[cpu]===0.3.20" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

*prior* to installing Econpizza. Econpizza should then run just fine (`proof <https://github.com/gboehl/econpizza/actions/runs/2579662335>`_).

In case you run into an error with `ptxas` (like `in this case <https://github.com/tensorflow/models/issues/7640>`_), a workaround is to disable CUDA by running the following **before** importing econpizza or JAX:

.. code-block:: python

    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


References
----------

**econpizza** is developed by Gregor Boehl to simulate nonlinear perfect foresight models. Please cite it with

.. code-block::

    @Misc{boehl2022pizza,
    title         = {Econpizza: solving nonlinear heterogeneous agents models using machine learning techniques},
    author        = {Boehl, Gregor},
    howpublished  = {\url{https://econpizza.readthedocs.io/_/downloads/en/latest/pdf/}},
    year = {2022}
    }

For the Boehl-Hommes method: Boehl and Hommes (2021). `Rational vs. Irrational Beliefs in a Complex World <https://gregorboehl.com/live/rational_chaos_bh.pdf>`_. *IMFS Working papers*


.. code-block::

    @techreport{boehl2021rational,
    title         = {Rational vs. Irrational Beliefs in a Complex World},
    author        = {Boehl, Gregor and Hommes, Cars},
    year          = 2021,
    institution   = {IMFS Working Paper Series}
    }


I appreciate citations for **econpizza** because it helps me to find out how people have been using the package and it motivates further work.
