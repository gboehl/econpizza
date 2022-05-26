
econpizza
=========

**Simulate nonlinear perfect foresight models, with or without heterogeneous agents**

.. image:: https://github.com/dfm/emcee/workflows/Tests/badge.svg
    :target: https://github.com/gboehl/econpizza/actions?query=workflow%3ATests
.. image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza
.. image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest    

The baseline mechanism is a Newton-based stacking method in the spirit of Boucekkine (1995), Juillard (1996) and others. Hence, the method is similar to the solver in dynare, but faster and more robust due to the use of automatic differentiation and sparse jacobians. Even perfect-foresight IRFs for large-scale nonlinear models with, e.g., occassionally binding constraints can be computed in less than a second. 

The package makes heavy use of `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

The package allows to solve nonlinear HANK models. The approach to deal with the distribution is inspired by the `Sequence-Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ method (`Auclert et al., 2022, ECMA <https://doi.org/10.3982/ECTA17434>`_). Steady state and nonlinear impulse responses (including, e.g., the ELB) can typically be found within a few seconds.

Installation
-------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as:

.. code-block:: bash

   pip install econpizza
  
Alternatively, the most recent version from GitHub with some experimental features can be installed via

.. code-block:: bash

   pip install git+https://github.com/gboehl/grgrlib
   pip install git+https://github.com/gboehl/econpizza

Note that the latter requires `git <https://www.activestate.com/resources/quick-reads/pip-install-git/#:~:text=To%20install%20Git%20for%20Windows,installer%20and%20follow%20the%20steps.>`_ to be installed.

Documentation
-------------

The documentation can be found `here <https://econpizza.readthedocs.io/en/latest/tutorial.html>`_.

Citation
--------

**econpizza** is developed by Gregor Boehl to simulate nonlinear perfect foresight models. Please cite it with

.. code-block::

    @Misc{boehl2022pizza,
    title         = {Econpizza: Solve all sorts of nonlinear perfect foresight models},
    author        = {Boehl, Gregor},
    howpublished  = {\url{https://github.com/gboehl/econpizza}},
    year = {2022}
    }

For the Boehl-Hommes method:

.. code-block::

    @techreport{boehl2021rational,
    title         = {Rational vs. Irrational Beliefs in a Complex World},
    author        = {Boehl, Gregor and Hommes, Cars},
    year          = 2021,
    institution   = {IMFS Working Paper Series}
    }


I appreciate citations for **econpizza** because it helps me to find out how people have been using the package and it motivates further work.


References
----------

Boehl, Gregor and Hommes, Cars (2021). `Rational vs. Irrational Beliefs in a Complex World <https://gregorboehl.com/live/rational_chaos_bh.pdf>`_. *IMFS Working papers*
