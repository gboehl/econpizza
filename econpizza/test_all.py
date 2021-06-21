#!/bin/python
# -*- coding: utf-8 -*-

from .__init__ import *


def test_bh():

    mod = parse(example_bh)

    state = np.zeros(len(mod["variables"]))
    state[:-1] = [0.1, 0.2, 0.0]

    x, _, flag = find_path(mod, state, T=1000, max_horizon=1000, tol=1e-8, verbose=2)

    assert flag == 0
    assert np.allclose(x[9], np.array([0.12557463, 0.12244423, 0.11939178, 0.22274411]))


def test_nk():

    mod = parse(example_nk)

    state = mod["stst"].copy()
    state["beta"] *= 1.02

    x, _, flag = find_path(mod, state.values(), verbose=2)

    assert flag == 0
    assert np.allclose(
        x[9],
        np.array(
            [
                3.08915869,
                3.08899444,
                1.00390525,
                1.0,
                0.99659073,
                1.00608913,
                0.83039318,
            ]
        ),
    )
