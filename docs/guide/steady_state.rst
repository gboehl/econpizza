The steady state
================

The steady state search can be evoked by calling the function :meth:`econpizza.PizzaModel.solve_stst` documented below. The function collects all available information from ``steady_state`` key of the YAML and attempts to find a set of variables and parameters that satisfies the aggregate equations using the routine outlined in the paper.

Upon failure, the function tries to be as informative as possible. If the search is not successful, a possible path to find the error is to set the function's keyword argument ``raise_errors`` to ``False``. The function then raises a warning instead of failing with an exception, and returns a dictionary containing the results from the root finding routine, such as, e.g. the last Jacobian matrix.

.. note::

   A classic complaint is "**Function returns NaNs**" or "**The Jacobian contains NaNs**". This is usually either due infinite/non-existent derivatives or invalid values (e.g. `sqrt(-1)` or `1/0`) along the way. While the package tries to provide more information about where the error occurred, a good idea is to follow JAX's hints on `how to debug NaNs <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#debugging-nans>`_.

.. tip::

   * A common gotcha for heterogeneous agent models is that the distribution contains negative values. The routine will be informative about that. This is usually due to rather excessive interpolation outside the grid and can often be fixed by using a grid with larger minimum/maximum values.
   * To avoid NaNs, ensure that parameter values produce valid results on an unbounded domain. E.g., instead of declaring a parameter `nu` when calling `sqrt(nu)`, declare a parameter `nu_log` and define `nu = exp(nu_log)` in the `aux_equations` block. This ensures that `nu` is always positive.
   * A known problem is that for steady state values set to zero (either as fixed value or initial guess), the Jacobian matrix may contain NaNs. This is due to how JAX internally calculates Jacobians via Jacobian-vector-products. It then often helps to set these values to `1e-8` instead of `0`.
   * The steady state procedure is based on the `pseudoinverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`_ of the Jacobian. If the procedure fails, it will try to tell you the rank of the Jacobian and the number of degrees of freedom. More degrees of freedom than the Jacobian rank normally implies that you should fix more steady state values and vice versa.
   * If the desired precision is not reached, the error message will tell you in which equation the maximum error did arise. You can use the ``equations`` key to get the list of equations (as strings), e.g. ``print(model['equations'][17])`` to get the equation with index 17.

.. autofunction:: econpizza.PizzaModel.solve_stst
