#!/bin/python
# -*- coding: utf-8 -*-

import jax
import numpy as np


def find_path_linear(model, shock, T, x, use_linear_guess):
    """Solves the expected trajectory given the linear model.
    """

    if model.get("lin_pol") is not None:

        stst = np.array(list(model["stst"].values()))
        sel = np.isclose(stst, 0)

        shocks = model.get("shocks") or ()
        tshock = np.zeros(len(shocks))
        if shock is not None:
            tshock[shocks.index(shock[0])] = shock[1]

        x_lin = np.empty_like(x)
        x_lin[0] = x[0]
        x_lin[0][~sel] = (x[0][~sel] / stst[~sel] - 1)

        for t in range(T):
            x_lin[t + 1] = model["lin_pol"][0] @ x_lin[t]

            if not t:
                x_lin[t + 1] += model["lin_pol"][1] @ tshock

        x_lin[:, ~sel] = ((1 + x_lin) * stst)[:, ~sel]

        if use_linear_guess:
            return x_lin.copy(), x_lin
        else:
            return x, x_lin
    else:
        return x, None
