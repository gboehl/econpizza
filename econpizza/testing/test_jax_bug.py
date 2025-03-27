# -*- coding: utf-8 -*-

import econpizza as ep # pizza

def test_jax_bug():

    example_hank2 = ep.examples.hank2
    hank2_dict = ep.parse(example_hank2)
    hank2 = ep.load(hank2_dict)
    _ = hank2.solve_stst()

    assert ep.jnp.isclose(hank2.stst['y'], 1.681181863723405)
