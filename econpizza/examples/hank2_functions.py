# -*- coding: utf-8 -*-
"""functions for the two-asset HANK model. Heavily inspired by https://github.com/shade-econ/sequence-jacobian/#sequence-space-jacobian
"""

import jax.numpy as jnp
from econpizza.utilities.interp import interpolate_coord, apply_coord, interpolate, lhs_equals_rhs_interpolate
from grgrjax import jax_print


def egm_init_Wa(b_grid, a_grid, z_grid, sigma_c):
    Wa = (0.6 + 1.1 * b_grid[:, None] +
          a_grid) ** (-sigma_c) * jnp.ones((z_grid.shape[0], 1, 1))
    return Wa


def egm_init_Wb(b_grid, a_grid, z_grid, sigma_c):
    Wb = (0.5 + b_grid[:, None] + 1.2 *
          a_grid) ** (-sigma_c) * jnp.ones((z_grid.shape[0], 1, 1))
    return Wb


def egm_step(Wa_p, Wb_p, a_grid, b_grid, z_grid, e_grid, kappa_grid, beta, sigma_c, rb, ra, chi0, chi1, chi2, Psi1):

    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Wb_p
    Wa = beta * Wa_p
    W_ratio = Wa / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===
    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1+Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_unc = apply_coord(i, pi, a_grid)
    c_endo_unc = apply_coord(i, pi, Wb) ** (-1/sigma_c)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===
    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + a_endo_unc + addouter(-z_grid, b_grid, -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (interpolate_coord and apply_coord work on last axis, so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    a_unc = apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = apply_coord(i, pi, b_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===
    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + kappa_grid[None, :, None])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_con = apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + kappa_grid[None, :, None]) ** (-1/sigma_c)
                  * apply_coord(i, pi, Wb[:, 0:1, :]) ** (-1/sigma_c))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===
    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + a_endo_con
              + addouter(-z_grid, jnp.full(len(kappa_grid),
                         b_grid[0]), -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = interpolate(b_endo.swapaxes(1, 2), b_grid,
                        a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===
    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    b = jnp.where(b_unc <= b_grid[0], b_grid[0], b_unc)
    a = jnp.where(b_unc <= b_grid[0], a_con, a_unc)

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c = addouter(z_grid, (1 + rb) * b_grid, (1 + ra) * a_grid) - Psi - a - b

    uc = c ** (-sigma_c)
    uce = e_grid[:, None, None] * uc

    # update derivatives of value function using envelope conditions
    Wa = (1 + ra - Psi2) * uc
    Wb = (1 + rb) * uc

    return Wa, Wb, a, b, c, uce


def adjustment_costs(a, a_grid, ra, chi0, chi1, chi2):
    chi = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)[0]
    return chi


def marginal_cost_grid(a_grid, ra, chi0, chi1, chi2):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, None],
                             a_grid[None, :], ra, chi0, chi1, chi2)[1]
    return Psi1


def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """Adjustment cost Psi(ap, a) and its derivatives with respect to
    first argument (ap) and second argument (a)"""
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = jnp.abs(a_change)
    sign_change = jnp.sign(a_change)

    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)

    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
    return Psi, Psi1, Psi2


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, None, None] + b[:, None] + a


def income(e_grid, tax, w, N, transfers=0.):
    z_grid = (1 - tax) * w * N * e_grid + transfers
    return z_grid
