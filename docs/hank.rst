
HANK
----

There is **experimental** support for heterogenous agent models. So far only one distribution is implemented.

The provided example is the same model as used by Auclert et al., 2022, which is documented `in a notebook here <https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/hank.ipynb>`_ and given in the appendix of their paper. 

There are some deviations:

 * a monetary policy rule with interest rate inertia
 * the zero lower bound on nominal interest rates
 * the NK-Phillips Curve is the conventional nonlinear Phillips Curve as derived from Rothemberg pricing
Detailscan be found in the section on the `yaml` file `right below <https://econpizza.readthedocs.io/en/latest/tutorial.html#the-yaml-file>`_.

The following code block simulates the example model:

.. code-block:: python

    from grgrlib import pplot
    import econpizza as ep
    from econpizza import example_hank

    # parse the example hank model from *.yaml
    mod_dict = ep.parse(example_hank)
    # compile the model
    mod = ep.load(mod_dict)
    # solve the steady state
    _ = mod.solve_stst()

    # this is a dict containing the steady state values
    x0 = mod['stst'].copy()
    # setting a shock on the discount factor
    x0['beta'] = 0.99

    # use the adjusted steady state as initial value
    xst, _, flags = mod.find_stack(x0.values(), horizon=100, tol=1e-8)

    # plot the dynamic responses using the pplot function from grgrlib
    pplot(xst[:30], labels=mod['variables'])


Further details on the implementation of heterogeneous agent models are given `in the technical section <https://econpizza.readthedocs.io/en/latest/method.html>`_.

