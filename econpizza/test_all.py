#!/bin/python
# -*- coding: utf-8 -*-

import os
from econpizza.__init__ import *
import econpizza as ep

filepath = os.path.dirname(__file__)


def test_bh(create=False):

    mod = load(example_bh, raise_errors=False)
    _ = mod.solve_stst()

    state = np.zeros(len(mod["variables"]))
    state[:-1] = [0.1, 0.2, 0.0]

    x, _, flag = find_path(
        mod, state, T=50, max_horizon=500, tol=1e-8, verbose=2)

    path = os.path.join(filepath, "test_storage", "bh.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


def test_nk(create=False):

    mod = load(example_nk)
    _ = mod.solve_stst()

    state = mod["stst"].copy()
    state["beta"] *= 1.02

    x, _, flag = find_path(mod, state.values(), T=10,
                           max_horizon=10, verbose=2)

    path = os.path.join(filepath, "test_storage", "nk.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


def test_stacked(create=False):

    mod = load(example_nk)
    _ = mod.solve_stst()

    shk = ("e_beta", 0.02)

    x, _, flag = find_path_stacked(mod, shock=shk)

    path = os.path.join(filepath, "test_storage", "stacked.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


def test_hank(create=False):

    mod_dict = ep.parse(example_hank)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst()

    x0 = mod['stst'].copy()
    x0['beta'] = 0.99  # setting a shock on the discount factor

    x, _, flag = mod.find_stack(x0.values(), horizon=100)

    path = os.path.join(filepath, "test_storage", "hank.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


def test_hank_labor(create=False):

    mod_dict = ep.parse(example_hank_labor)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst()

    x0 = mod['stst'].copy()
    x0['beta'] = 0.99  # setting a shock on the discount factor

    x, _, flag = mod.find_stack(x0.values(), horizon=100)

    path = os.path.join(filepath, "test_storage", "hank_labor.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


def test_hank2(create=False):

    mod_dict = ep.parse(example_hank2)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst()

    x0 = mod['stst'].copy()
    x0['beta'] = 0.99  # setting a shock on the discount factor

    x, _, flag = mod.find_stack(x0.values(), horizon=100)

    path = os.path.join(filepath, "test_storage", "hank2.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)


test_bh()
test_nk()
test_stacked()
test_hank()
test_hank_labor()
test_hank2()
