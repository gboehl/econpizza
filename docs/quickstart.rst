
Getting started
===============

This package contains two methods. *Stacking*, the main method, is a generic nonlinear solver that should work on all sorts of problems. *Shooting* is the method of Boehl & Hommes (2021), which is useful for models with nonlinear, chaotic dynamics.

Quickstart
----------

An small-scale nonlinear New Keynesian model with ZLB is provided `as an example <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. Here is how to simulate it and plot some nonlinear impulse responses:

.. code-block:: python

    import matplotlib.pyplot as plt
    import econpizza as ep
    from econpizza import example_nk


    # use the NK model again
    mod = ep.load(example_nk)
    _ = mod.solve_stst()

    # increase the discount factor by .02 (this is _not_ percentage deviation!)
    shk = ('e_beta', .02)

    # use the stacking method. As above, you could also feed in the initial state instead
    x, flag = mod.find_path(shock=shk)

    # plotting. x_lin is the linearized first-order solution
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)


The impulse responses are the usual dynamics of a nonlinear DSGE.

Alternatively, you can specify the initial conditions instead of a shock vector (some models do not feature shocks):

.. code-block:: python

    # get the steady state as initial condion
    x0 = mod['stst'].copy()
    # and multiply the initial beta by 1.01
    x0['beta'] *= 1.01

    # solving...
    x, flag = mod.find_path(x0=x0.values())

    # plotting...
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

You can also have a look at the linearized solution. This is pretty much shat the guys from the
`Sequence Space Jacobian <https://github.com/shade-econ/sequence-jacobian>`_ are doing.

.. code-block:: python

    # solve for the linear impulse response
    x_lin, flag = mod.find_path_linear(x0=x0.values())

    # and compare linear and nonlinear IRFs
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:30,i])
        plt.plot(x_lin[:30,i])
        plt.title(v)

The resulting plots will look somewhat similar to this:

.. image:: https://github.com/gboehl/econpizza/blob/master/docs/lin_and_nlin.png?raw=true
  :width: 600
  :alt: Linear vs. nonlinear IRFs

The folder `yaml files <https://github.com/gboehl/econpizza/tree/master/econpizza/examples>`_ also contains a medium scale New Keynesian DSGE model as an example file (``med_scale_nk.yaml``, see `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/med_scale_nk.yaml>`_). It can be imported with:

.. code-block:: python

    from econpizza import example_dsge

    mod = ep.load(example_dsge)



.. include:: the_yaml.rst
.. include:: boehl_hommes.rst
