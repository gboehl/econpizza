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

    x, flag = find_path_shooting(
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

    x, flag = find_path_shooting(mod, state.values(), T=10,
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

    x, flag = find_path_stacking(mod, shock=shk)

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
    _ = mod.solve_stst(tol=1e-4)

    x0 = mod['stst'].copy()
    x0['beta'] *= 1.01  # setting a shock on the discount factor

    x, flag = mod.find_path(x0.values(), horizon=10)
    x_lin, _ = mod.find_path_linear(x0.values(), horizon=10)
    het_vars = mod.get_het_vars(x)
    dist = het_vars['dist']

    path_x = os.path.join(filepath, "test_storage", "hank.npy")
    path_x_lin = os.path.join(filepath, "test_storage", "hank_lin.npy")
    path_dist = os.path.join(filepath, "test_storage", "hank_dist.npy")

    if create:
        np.save(path_x, x)
        np.save(path_x_lin, x_lin)
        np.save(path_dist, dist)
        print(f'Test file updated at {path_x},{path_x_lin} and {path_dist}')
    else:
        test_x = np.load(path_x)
        test_x_lin = np.load(path_x_lin)
        test_dist = np.load(path_dist)

        assert flag == 0
        assert np.allclose(x, test_x)
        assert np.allclose(x_lin, test_x_lin)
        assert np.allclose(dist, test_dist)


def test_hank_labor(create=False):

    mod_dict = ep.parse(example_hank_labor)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst(tol=1e-6)

    x0 = mod['stst'].copy()
    x0['beta'] *= 1.01  # setting a shock on the discount factor

    x, flag = mod.find_path(x0.values(), horizon=10, use_jacrev=False)

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
    _ = mod.solve_stst(tol=1e-6)

    x0 = mod['stst'].copy()
    x0['beta'] *= 1.01  # setting a shock on the discount factor

    x, flag = mod.find_path(x0.values(), horizon=10)

    path = os.path.join(filepath, "test_storage", "hank2.npy")

    if create:
        np.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = np.load(path)

        assert flag == 0
        assert np.allclose(x, test_x)
