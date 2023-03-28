Nonlinear simulations
=====================

The main functionality of nonlinear simulations is provided by the functions :meth:`econpizza.PizzaModel.find_path`. The main arguments are either ``shock`` or ``init_state``, which allows to specify an economic shock as a tuple of the shock name (as specified in ``shocks`` in the YAML) and the size of the shock, or a vector of initial states, respectively.

.. note::

   All numerical methods are subject to numerical errors. To reduce these, you can decrease the numerical tolerance ``tol``. However, this should not be below the tolerance level for the steady state search, or below machine precision.

If the model has heterogeneous agents, the routine will automatically compute the steady state sequence space Jacobian. This can be skipped using the ``skip_jacobian`` flag.

.. autofunction:: econpizza.PizzaModel.find_path

The function :meth:`econpizza.PizzaModel.get_distributions` allows to retrieve the sequence of distributions and decision variables. To that end it requires the shocks and initial distribution together with the trajectory of aggregated variables as input.

.. autofunction:: econpizza.PizzaModel.get_distributions
