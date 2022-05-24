Getting started
===============

This package contains two methods. *Stacking*, the main method, is a generic nonlinear solver that should work on all sorts of problems. *Shooting* is the method of Boehl & Hommes (2021), which is useful for models with nonlinear, chaotic dynamics.

Quickstart
----------

An small-scale nonlinear New Keynesian model with ZLB is provided `as an example <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. Here is how to simulate it and plot some nonlinear impulse responses:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import econpizza as ep
    from econpizza import example_nk


    # use the NK model again
    mod = ep.load(example_nk)
    _ = mod.solve_stst()
    _ = mod.solve_linear()

    # increase the discount factor by .02 (this is _not_ percentage deviation!)
    shk = ('e_beta', .02)

    # use the stacking method. As above, you could also feed in the initial state instead
    x, x_lin, flag = mod.find_stack(shock=shk)

    # plotting. x_lin is the linearized first-order solution
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.plot(x_lin[:,i])
        plt.title(v)


The impulse responses are the usual dynamics of a nonlinear DSGE.

The folder `yaml files <https://github.com/gboehl/econpizza/tree/master/econpizza/examples>`_ also contains a medium scale New Keynesian DSGE model as an example file (``med_scale_nk.yaml``, see `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/med_scale_nk.yaml>`_). It can be imported with:

.. code-block:: python

    from econpizza import example_dsge

    mod = ep.load(example_dsge)


.. include:: the_yaml.rst
.. include:: boehl_hommes.rst
