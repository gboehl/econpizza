
HANK
----

There is **experimental** support for heterogenous agent models. So far only one distribution is implemented.

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

Further details on the implementation and the yaml file can be found ...
