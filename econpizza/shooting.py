#!/bin/python
# -*- coding: utf-8 -*-

import yaml
import re
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from numba import njit


def parse(mfile):

    f = open(mfile)
    mtxt = f.read()
    f.close()

    mtxt = mtxt.replace('^', '**')
    mtxt = mtxt.replace(';', '')
    mtxt = re.sub(r"@ ?\n", " ", mtxt)
    model = yaml.safe_load(mtxt)

    evars = model['variables']
    shocks = model.get('shocks') or ()
    par = model['parameters']
    stst = model.get('steady_state').get('fixed_values')
    eqns = ('F[%s] = %s' % (i, e) for i, e in enumerate(model['equations']))
    eqns_aux = model.get('aux_equations')

    for k in stst:
        if isinstance(stst[k], str):
            stst[k] = eval(stst[k])

    model['stst'] = stst

    if not shocks:
        shock_str = ''
    elif len(shocks) > 1:
        shock_str = ', '.join(shocks)+' = shocks'
    else:
        shock_str = shocks[0] + ' = shocks[0]'

    func_str = '''def func_raw(XLag, X, XPrime, XSS, shocks, pars):\n %s\n %s\n %s\n %s\n %s\n %s\n F=np.empty(%s)\n %s\n %s\n return F''' % (
        ', '.join(v + 'Lag' for v in evars)+' = XLag',
        ', '.join(evars)+' = X',
        ', '.join(v + 'Prime' for v in evars)+' = XPrime',
        ', '.join(v + 'SS' for v in evars)+' = XSS',
        shock_str,
        ', '.join(par.keys())+' = pars',
        str(len(evars)),
        '\n '.join(eqns_aux) if eqns_aux else '',
        '\n '.join(eqns))

    try:
        exec(func_str, globals())
        func = njit(func_raw)
    except Exception as e:
        print(func_str)
        raise e

    model['func'] = func
    model['func_str'] = func_str
    solve_stst(model)

    return model


def solve_stst(model):

    evars = model['variables']
    func = model['func']
    par = model['parameters']
    inits = model['steady_state'].get('init_guesses')
    stst = model.get('stst')
    shocks = model.get('shocks') or ()

    def func_stst(x):

        xss = ()
        for i, v in enumerate(evars):
            if v in stst:
                xss += stst[v],
            else:
                xss += x[i],

        XSS = np.array(x)
        trueXSS = np.array(xss)

        return func(XSS, XSS, XSS, trueXSS, np.zeros(len(shocks)), np.array(list(par.values())))

    init = ()
    for v in evars:

        ininit = False
        if isinstance(inits, dict):
            if v in inits.keys():
                ininit = True

        if v in stst.keys():
            init += stst[v],
        elif ininit:
            init += inits[v],
        else:
            init += 1.,

    res = so.root(func_stst, init)

    if not res['success'] or np.any(np.abs(func_stst(res['x'])) > 1e-8):
        raise Exception('Steady state not found')

    rdict = dict(zip(evars, res['x']))
    model['stst'] = rdict
    model['stst_vals'] = np.array(list(rdict.values()))
    return rdict


def solve_current(model, XLag, XPrime):

    func = model['func']
    par = model['parameters']
    stst = model.get('stst')
    shocks = model.get('shocks') or ()

    def func_current(x): return func(XLag, x, XPrime, np.array(
        list(stst.values())), np.zeros(len(shocks)), np.array(list(par.values())))
    res = so.root(func_current, XPrime, options=model['root_options'])

    if not res['success']:
        raise Exception('Current state not found')

    err = np.max(np.abs(func_current(res['x'])))
    if err > 1e-8:
        print("Maximum error exceeds tolerance with %s." % err)

    return res['x']


def find_path(model, x0, n=30, max_horizon=100, max_iter=200, eps=1e-16, root_options=None, verbose=True):

    stst = list(model['stst'].values())
    evars = model['variables']
    model['root_options'] = root_options

    x = np.ones((n+max_horizon, len(evars)))*np.array(stst)
    x[0] = list(x0)

    flag = np.zeros(3)

    for i in range(n):

        cond = False
        cnt = 2

        while True:

            x_old = x[1].copy()
            imax = min(cnt, max_horizon)

            for t in range(imax):
                x[t+1] = solve_current(model, x[t], x[t+2])

            if np.abs(x_old - x[1]).max() < eps and cnt > 2:
                break

            if cnt == max_iter:
                flag[0] = 1
                break

            if np.any(np.isnan(x)):
                flag[1] = 1
                break

            if np.any(np.isinf(x)):
                flag[2] = 1
                break

            cnt += 1

    fin_flag = 0
    mess = ''

    if flag[0]:
        fin_flag += 1
        mess += ', max_iter reached'

    if flag[1]:
        fin_flag += 2
        mess += ', contains NaNs'

    if flag[2]:
        fin_flag += 4
        mess += ', contains infs'

    if verbose:
        print('Pizza done%s.' % mess)

    return x[:n], fin_flag
