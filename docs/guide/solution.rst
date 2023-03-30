Nonlinear simulations
=====================

The main functionality of nonlinear simulations is provided by the functions :meth:`econpizza.PizzaModel.find_path`. The main arguments are either ``shock`` or ``init_state``, which allows to specify an economic shock as a tuple of the shock name (as specified in ``shocks`` in the YAML) and the size of the shock, or a vector of initial states, respectively.

.. note::

   All numerical methods are subject to numerical errors. To reduce these, you can decrease the numerical tolerance ``tol``. However, this should not be below the tolerance level for the steady state search, or below machine precision.

.. hint::

   As stated in the paper, a sufficient condition for convergence of the solution routine is that the generalized eigenvalues of the sequence space Jacobian and its steady-state pendant are all positive (which is prohibitory expensive to check). If the method does not converge, the ``use_solid_solver=True`` flag can be used to check if the model solves when using a conventional Newton method with the true Jacobian.

.. autofunction:: econpizza.PizzaModel.find_path

If the model has heterogeneous agents, the routine will automatically compute the steady state sequence space Jacobian. This can be skipped using the ``skip_jacobian`` flag.

The function :meth:`econpizza.PizzaModel.get_distributions` allows to retrieve the sequence of distributions and decision variables. To that end it requires the shocks and initial distribution together with the trajectory of aggregated variables as input.

.. autofunction:: econpizza.PizzaModel.get_distributions
