#!/bin/python
# -*- coding: utf-8 -*-

import jax
import scipy.sparse as ssp
import scipy as sp


def newton(func, jac, init, maxit, tol, sparse, verbose):

    res = {}
    cnt = 0
    xi = init.copy()

    while True:
        cnt += 1
        xold = xi.copy()
        if sparse:
            xi -= ssp.linalg.spsolve(ssp.csr_matrix(jac(xi)), func(xi))
        else:
            xi -= sp.linalg.solve(jac(xi), func(xi))
        eps = jax.numpy.abs(xi - xold).max()

        if verbose:
            print(f'    Iteration {cnt:3d} | max error {eps:.2e}')

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < 1e-8:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jax.numpy.isnan(eps):
            raise Exception('Newton method returned `NaN`')

    res['x'], res['fun'], res['niter'] = xi, func(xi), cnt

    return res
