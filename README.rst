
econpizza
=========

Contains simple tools to simulate perfect foresight models. The method is similar to the one introduced in Boehl & Hommes (2021), where we use it to solve for chaotic asset price dynamics. It can be understood as a policy function iteration where the initial state is the only fixed grid point and all other grid points are chosen endogenously (as in EGM) to map the expected trajectory. The main advantage (in terms of robustness) over Fair-Taylor comes from exploiting fact that any determined perfect forsight model must be a contraction mapping.

The code is in alpha state and provided for reasons of collaboration, replicability and code sharing in the spirit of open science. You are welcome to get in touch if you are interested working with the package.


Documentation
-------

There is no formal documentation (yet). Check
`this documentation <https://pydsge.readthedocs.io/en/latest/installation_guide.html>`_ for an easy going installation (using `econpizza` instead of `pydsge`). The package optionally depends on the `grgrlib` package to check for determinancy, so maybe also install it when you are at it.

An small-scale nonlinear New Keynesian model with ZLB is provided `as an example <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. Here is how to simulate it and plot some nonlinear impulse responses:


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from econpizza import *

    # load the example. The steady state is automatically solved for
    # example_nk is nothing else but the path to the yaml, hence you could also use `filename = 'path_to/model.yaml'`
    mod = parse(example_nk)

    # get the steady state as an initial state
    state = mod['stst'].copy()
    # increase the discount factor by one percent
    state['beta'] *= 1.02

    # simulate the model
    x, flag = find_path(mod, state.values())

    # plotting
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

The impulse responses are the usual dynamics of a nonlinear DSGE.

The `yaml files <https://github.com/gboehl/econpizza/tree/master/econpizza/examples>`_ follow a simple structure:

1. define all variables and shocks
2. provide the nonlinear equations. Note that the dash at the beginning of each line is *not* a minus!
3. provide the parameters and values.
4. optionally provide some steady state values and/or values for initial guesses
5. optionally provide some auxilliary equations that are not directly part of the nonlinear system (see the `yaml for the BH model <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/bh.yaml>`_)

Lets go for a second, numerically more challenging example: the chaotic rational expectations model of Boehl & Hommes (2021)

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from econpizza import *

    # parse the yaml
    mod = parse(example_bh)

    # choose an interesting initial state
    state = np.zeros(len(mod['variables']))
    state[:-1] = [.1, .2, 0.]

    # solve and simulate. The lower eps is not actually necessary
    x, flag = find_path(mod, state, T=1000, max_horizon=1000, tol=1e-8)

    # plotting
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

This will give you:

.. image:: docs/p_and_n.png
  :width: 400
  :alt: Dynamics of prices and fractions

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


We appreciate citations for **econpizza** because it helps us to find out how people have been using the package and it motivates further work.


References
----------

Boehl, Gregor and Hommes, Cars (2021). `Rational vs. Irrational Beliefs in a Complex World <https://gregorboehl.com/live/rational_chaos_bh.pdf>`_. *IMFS Working papers*
