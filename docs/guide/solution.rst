Nonlinear simulations
=====================

The main functionality of nonlinear simulations is provided by the function :meth:`econpizza.PizzaModel.find_path`
The main arguments are either ``shock`` or ``init_state``, which allows to specify an economic shock as a tuple of the shock name (as specified in ``shocks`` in the YAML) and the size of the shock, or a vector of initial states, respectively.
The function :meth:`econpizza.PizzaModel.get_distributions` allows to retrieve the full nonlinear sequence of the distribution.

.. note::

   All numerical methods are subject to numerical errors. To reduce these, you can decrease the numerical tolerance ``tol``. However, this should not be below the tolerance level used for the steady state search.

.. hint::

   A sufficient condition for convergence of the solution routine is that the `generalized eigenvalues <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Generalized_eigenvalue_problem>`_ of the sequence space Jacobian and its steady-state pendant are all positive. [#f1]_ If the procedure does not converge, the ``use_solid_solver=True`` flag can be used to check if the model solves when using a conventional Newton method with the true Jacobian (this may take quite a while).

.. autofunction:: econpizza.PizzaModel.find_path

If the model has heterogeneous agents, the routine will automatically compute the steady state sequence space Jacobian. This can be skipped using the ``skip_jacobian`` flag.

Any additional argument will be passed on to the specific Newton method. For models with heterogeneous agents this is :meth:`econpizza.utilities.newton.newton_for_jvp`:

.. autofunction:: econpizza.utilities.newton.newton_for_jvp

For models with representative agents, the Newton method is :meth:`econpizza.utilities.newton.newton_for_banded_jac`:

.. autofunction:: econpizza.utilities.newton.newton_for_banded_jac

If ``use_solid_solver`` is set to `True`, the Newton method `newton_jax_jit <https://grgrjax.readthedocs.io/en/latest/#grgrjax.newton_jax_jit>`_ from the `grgrjax <https://grgrjax.readthedocs.io>`_ package is used.

The function :meth:`econpizza.PizzaModel.get_distributions` allows to retrieve the sequence of distributions and decision variables. To that end it requires the shocks and initial distribution together with the trajectory of aggregated variables as input.

.. autofunction:: econpizza.PizzaModel.get_distributions

.. rubric:: Footnotes

.. [#f1] Unfortunately, this is prohibitory expensive to check as it would require to calculate the full sequence space Jacobian and its eigenvalues.
