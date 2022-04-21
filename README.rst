
econpizza
=========

**Simulate nonlinear perfect foresight models in Python using AD**

.. image:: https://github.com/dfm/emcee/workflows/Tests/badge.svg
    :target: https://github.com/gboehl/econpizza/actions?query=workflow%3ATests
.. image:: https://badge.fury.io/py/econpizza.svg
    :target: https://badge.fury.io/py/econpizza
.. image:: https://img.shields.io/badge/GitHub-gboehl%2Feconpizza-blue.svg?style=flat
    :target: https://github.com/gboehl/econpizza
.. image:: https://readthedocs.org/projects/econpizza/badge/?version=latest
    :target: http://econpizza.readthedocs.io/en/latest/?badge=latest    

The baseline mechanism is a Newton-based stacking method in the spirit of Boucekkine (1995), Juillard (1996) and others. It is hence similar to the solver in dynare, but faster and more robust due to the use of automatic differentiation and sparse jacobians. Even IRFs for large-scale models with occassionally binding constraints can be computed in less than a second.

The package makes heavy use of `automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ via `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_!

Installation
-------------

It's as simple as:

.. code-block:: bash

   pip install econpizza

Documentation
-------------

There is some `documentation <https://econpizza.readthedocs.io/en/latest/tutorial.html>`_ out there.

Citation
--------

**econpizza** is developed by Gregor Boehl to simulate nonlinear perfect foresight models. Please cite it with

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
