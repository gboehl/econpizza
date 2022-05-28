#!/bin/python
# -*- coding: utf-8 -*-

from econpizza.__init__ import *
import econpizza as ep


def test_bh():

    mod = load(example_bh, raise_errors=False)
    _ = mod.solve_stst()

    state = np.zeros(len(mod["variables"]))
    state[:-1] = [0.1, 0.2, 0.0]

    x, _, flag = find_path(
        mod, state, T=50, max_horizon=500, tol=1e-8, verbose=2)

    assert flag == 0
    assert np.allclose(x[9], np.array(
        [0.22287535, 0.25053816, 0.24429734, 0.23821162]))


def test_nk():

    mod = load(example_nk)
    _ = mod.solve_stst()

    state = mod["stst"].copy()
    state["beta"] *= 1.02

    x, _, flag = find_path(mod, state.values(), T=10,
                           max_horizon=10, verbose=2)

    assert flag == 0
    assert np.allclose(
        x[9], np.array([1.00608913, 0.32912145, 6.50140449, 1.00499411,
                       1.00266364, 1.00266364, 0.83199537, 0.32912146])
    )


def test_stacked():

    mod = load(example_nk)
    _ = mod.solve_stst()

    shk = ("e_beta", 0.02)

    x, _, flag = find_path_stacked(mod, shock=shk)

    assert flag == 0
    assert np.allclose(
        x[9],
        np.array([1.00703268, 0.32763172, 6.50140449, 1.00377031,
                 1., 0.99306852, 0.82840695, 0.32765387])
    )


def test_hank():

    mod_dict = ep.parse(example_hank)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst(tol_newton=1e-4, tol_forwards=1e-4)

    x0 = mod['stst'].copy()
    x0['beta'] = 0.99  # setting a shock on the discount factor

    x, _, flag = mod.find_stack(x0.values(), horizon=50, tol=1e-4)

    assert not flag
    assert np.allclose(
        x[9], np.array(
            [5.60257058,  0.99016942,  0.99957714,  0.1660532,
             1.00090995,  0.99473296,  0.99682864,  1.,
             0.9915595,  1.002, -0.01776778,  0.42670988,
             0.28734992,  0.78473366,  0.83276591,  0.99957714,
             1.00090995,  1.]
        )
    )
