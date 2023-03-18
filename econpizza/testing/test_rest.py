# -*- coding: utf-8 -*-

import os
import sys
import jax.numpy as jnp
# autopep8: off
sys.path.insert(0, os.path.abspath("."))
import econpizza as ep
# autopep8: on

filepath = os.path.dirname(__file__)


def test_bh(create=False):

    mod = ep.load(ep.examples.bh, raise_errors=False)
    _ = mod.solve_stst()

    state = jnp.zeros(len(mod["variables"]))
    state = state.at[:-1].set([0.1, 0.2, 0.0])

    x, flag = ep.find_path_shooting(
        mod, state, horizon=50, max_horizon=500, tol=1e-8, verbose=2)

    path = os.path.join(filepath, "cache", "bh.npy")

    assert flag == 0
    if create:
        jnp.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = jnp.load(path)
        assert jnp.allclose(x, test_x)


def test_hank_labor(create=False):

    mod_dict = ep.parse(ep.examples.hank_labor)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst(tol=1e-8)

    shocks = ('e_beta', .005)

    x, flag = mod.find_path(shocks, horizon=50)

    path = os.path.join(filepath, "cache", "hank_labor.npy")

    assert flag == 0
    if create:
        jnp.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = jnp.load(path)
        assert jnp.allclose(x, test_x)


def test_solid(create=False):

    mod_dict = ep.parse(ep.examples.hank)
    mod = ep.load(mod_dict)
    _ = mod.solve_stst(tol=1e-8)

    shocks = ('e_beta', .005)

    x, flag = mod.find_path(shocks, use_solid_solver=True,
                            horizon=20, chunk_size=90)

    path = os.path.join(filepath, "cache", "hank_solid.npy")

    assert flag == 0
    if create:
        jnp.save(path, x)
        print(f'Test file updated at {path}')
    else:
        test_x = jnp.load(path)
        assert jnp.allclose(x, test_x)


def create_all():

    test_bh(create=True)
    test_hank_labor(create=True)
    test_solid(create=True)
