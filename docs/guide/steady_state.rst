The Steady State
================

The steady state search can be evoked by calling :meth:`econpizza.PizzaModel.solve_stst` documented below. The function collects all available information from the YAML and attempts to find a set of variables and parameters that satisfy the aggregate equations given the fixed values using the routine outlined in the original paper.

The function tries to be as informative as possible. If the search is not successful, one possible way of debugging is to set the keyword argument ``raise_errors`` to ``False``. The function then only raises a warning and returns a dictionary containing the results from the root finding process, such as, e.g. the last Jacobian matrix.

.. note::

   A classic complaint is that "**The Jacobian contains NaNs**". This is usually due to numerical errors somewhere along the way. While the package tries to provide more information about where the error occurred, a good alternative starting point is to look at the Jacobian from the last Newton iteration.

.. note::

   A common gotcha for heterogeneous agent models is that the distribution contains negative values. The procedure will be informative about that. This is usually due to too much interpolation outside the grid and can often be fixed by using a grid with larger maximum values.


.. autofunction:: econpizza.PizzaModel.solve_stst
