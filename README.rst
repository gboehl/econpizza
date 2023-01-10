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

Econpizza is a framework to solve and simulate *fully nonlinear* perfect foresight models, with or without heterogeneous agents.
A parser allows to express economic models in a simple, high-level fashion as yaml-files.
Generic and robust routines for steady state search are provided.

The baseline method for representative agent models builds on the shooting methods of, e.g., Boucekkine (1995) and Juillard (1996). It is faster and more reliable than the nonlinear solver in dynare due to the use of a Newton method in combination with automatic differentiation and efficient jacobian decompositions. Nonlinear perfect-foresight transition dynamics can - even for large-scale nonlinear models with several occassionally binding constraints - be computed in less than a second.

The package can solve nonlinear models with heterogeneous agents, such as HANK models with portfolio choice. Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.
The approach to deal with heterogeneity extends the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_) to fully nonlinear models by iteratively using `jacobian-vector producs <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions>`_ to construct the inverse jacobian during each Newton iteration. This not only allows to study the dynamics of aggregate variables, but also the complete nonlinear transition dynamics of the distribution of assets across agents.

The package builds heavily on `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

A central philosophy of this package is to consequently separate the low-level routines for *model solution* (which is what happens under the hood) from
*model specification* (via a ``yaml`` file) and the
high-level interface for *model simulation and analysis* (what the user does with the model).

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
