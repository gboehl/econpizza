
Boehl-Hommes method
-------------------

The package also contains an alternative "shooting" method much aligned to the one introduced in Boehl & Hommes (2021). In the original paper we use this method to solve for chaotic asset price dynamics. The method can be understood as a policy function iteration where the initial state is the only fixed grid point and all other grid points are chosen endogenously (as in a "reverse" EGM) to map the expected trajectory.

The main advantage (in terms of robustness) over the stacking method comes from exploiting the property that most determined perfect foresight models are a contraction mapping both, forward and backwards. The model is given by

.. code-block::

    f(x_{t-1}, x_t, x_{t+1}) = 0.

We iterate on the expected trajectory itself instead of the policy function. We hence require

.. code-block::

   d f(x_{t-1}, x_t, x_{t+1} ) < d x_{t-1},
   d f(x_{t-1}, x_t, x_{t+1} ) < d x_{t+1}.

This is also the weakness of the method: not every DSGE model (that is determined in the Blanchard-Kahn sense) is such backward-and-forward contraction. In most cases the algorithm converges anyways, but convergence is not guaranteed.

The following example shows how to use the shooting method on the simple New Keynesian model.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import econpizza as ep
    from econpizza import example_nk

    # load the example.
    # example_nk is nothing else but the path to the yaml, hence you could also use `filename = 'path_to/model.yaml'`
    mod = ep.load(example_nk)
    # solve for the steady state
    _ = mod.solve_stst()

    # get the steady state as an initial state
    state = mod['stst'].copy()
    # increase the discount factor by one percent
    state['beta'] *= 1.02

    # simulate the model
    x, _, flag = mod.find_path_shooting(state.values())

    # plotting
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

Lets go for a second, numerically more challenging example: the chaotic rational expectations model of Boehl & Hommes (2021)

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import econpizza as ep
    from econpizza import example_bh

    # parse the yaml
    mod = ep.load(example_bh, raise_errors=False)
    _ = mod.solve_stst()

    # choose an interesting initial state
    state = np.zeros(len(mod['variables']))
    state[:-1] = [.1, .2, 0.]

    # solve and simulate
    x, _, flag = ep.find_path_shooting(mod, state, T=500, max_horizon=1000, tol=1e-5)

    # plotting
    for i,v in enumerate(mod['variables']):

        plt.figure()
        plt.plot(x[:,i])
        plt.title(v)

This will give you boom-bust cycles in asset pricing dynamics:

.. image:: https://github.com/gboehl/econpizza/blob/master/docs/p_and_n.png?raw=true
  :width: 800
  :alt: Dynamics of prices and fractions
