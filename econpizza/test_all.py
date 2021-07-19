#!/bin/python
# -*- coding: utf-8 -*-

from .__init__ import *


def test_bh():

    mod = load(example_bh)

    state = np.zeros(len(mod["variables"]))
    state[:-1] = [0.1, 0.2, 0.0]

    x, _, flag = find_path(mod, state, T=1000, max_horizon=1000, tol=1e-8, verbose=2)

    assert flag == 0
    assert np.allclose(x[9], np.array([0.22287535, 0.25053816, 0.24429734, 0.23821162]))


def test_nk():

    mod = load(example_nk)

    state = mod["stst"].copy()
    state["beta"] *= 1.02

    x, _, flag = find_path(mod, state.values(), verbose=2)

    assert flag == 0
    assert np.allclose(
        x[9],
        np.array(
            [
                1.00608913,
                3.08899444,
                1.00390525,
                1.0,
                0.99659073,
                0.83039318,
                3.08915869,
            ]
        ),
    )


def test_stacked():

    mod = load(example_nk)

    shk = ("e_beta", 0.02)

    x, _, flag = find_path_stacked(mod, shock=shk)

    assert flag == 0
    assert np.allclose(
        x[9],
        np.array(
            [1.00703268, 3.08098172, 1.00377032, 1.0, 0.99306854, 0.82840692, 3.08119]
        ),
    )
