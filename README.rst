
econpizza
=========

Contains simple tools to simulate perfect foresight models. The method is similar to the one introduced in Boehl & Hommes (2021), where we use it to solve for chaotic asset price dynamics.

The code is in alpha state and provided for reasons of collaboration, replicability and code sharing in the spirit of open science. You are very welcome to get in touch if you are interested working with the package.



Documentation
-------

There is some no formal documentation (yet). An example small-scale New Keynesian model is provided `as an example <https://pydsge.readthedocs.io/en/latest/getting_started.html>`_.


.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt
    from econpizza import * 

    mod = parse(example)

    state = mod['stst'].copy()
    state['beta'] *= 1.01

    x, flag = find_path(mod, state.values())

    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

The `nk.yaml` file follows a simple structure. First, define all variables and shocks. Second, provide the nonlinear equations. Note that the dash is *not* a minus! Then provide the parameters. Lastly, you can optionally provide some steady state values and/or values for initial guesses.


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
