
econpizza
=========

Contains simple tools to simulate perfect foresight models. The method is similar to the one introduced in Boehl & Hommes (2021), where we use it to solve for chaotic asset price dynamics. It can be understood as a policy function iteration where the initial state is the only fixed grid point and all other grid points are chosen endogenously (as in EGM) to map the expected trajectory. The main advantage (in terms of robustness) over Fair-Taylor comes from exploiting fact that any determined perfect forsight model must be a contraction mapping.  

The code is in alpha state and provided for reasons of collaboration, replicability and code sharing in the spirit of open science. You are welcome to get in touch if you are interested working with the package.


Documentation
-------

There is no formal documentation (yet). An example small-scale New Keynesian model is provided `as an example <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. Here is how to simulate and plot some nonlinear impulse responses:


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from econpizza import * 
    
    # load the example. The steady state is automatically solved for
    mod = parse(example)

    # get the steady state as an initial state
    state = mod['stst'].copy()
    # increase the discount factor by one percent
    state['beta'] *= 1.01

    # simulate the model
    x, flag = find_path(mod, state.values())

    # plotting
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

The `nk.yaml <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_ file follows a simple structure:

1. define all variables and shocks
2. provide the nonlinear equations. Note that the dash at the beginning of each line is *not* a minus! 
3. provide the parameters and values. 
4. optionally provide some steady state values and/or values for initial guesses


Citation
--------

**econpizza** is developed by Gregor Boehl to simulate nonlinear perfect foresight models. Please cite it with

.. code-block::

    @Software{boehl2021,
      Title  = {econpizza -- A package to simulate nonlinear perfect foresight models},
      Author = {Gregor Boehl},
      Year   = {2021},
      Url    = {https://github.com/gboehl/econpizza},
    }

We appreciate citations for **econpizza** because it helps us to find out how people have been using the package and it motivates further work.


References
----------

Boehl, Gregor and Hommes, Cars (2021). `Rational vs. Irrational Beliefs in a Complex World <https://gregorboehl.com/live/rational_chaos_bh.pdf>`_. *Unpublished Manuscript*
