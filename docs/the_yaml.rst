
The *.yaml-file
-------------------

All relevant information is supplied via the yaml file. For general information about the YAML markup language and its syntax see `Wikipedia <https://en.wikipedia.org/wiki/YAML>`_. The yaml files follow a simple structure:

1. define all variables and shocks
2. provide the nonlinear equations. Note that each line starts with a `~`.
3. provide the parameters and values.
4. optionally provide some steady state values and/or values for initial guesses
5. optionally provide some auxilliary equations that are not directly part of the nonlinear system (see the `yaml for the BH model <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/bh.yaml>`_)

Let us have a look of the yaml of a hank model we will discuss in `the tutorial <https://econpizza.readthedocs.io/en/latest/tutorial.html>`_. The file can be found `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank.yaml>`_. The first line reads:

.. code-block::

    functions_file: '../examples/hank_functions.py'

The relative path to a functions-file, which may provide additional functions. In this example, the file defines the functions ``transfers``, ``wages``, ``hh``, ``labor_supply`` and ``hh_init``.

.. code-block::

    # these are available during all three stages (decisions, distributions, equations)
    definitions:
        - from numpy import log, maximum
        - from jax.experimental.host_callback import id_print as jax_print

General definitions and imports. These are available during all three stages (decisions, distributions, equations).

.. code-block::

    variables: [Div, Y, Yprod, w, pi, Rn, Rs, R, Rstar, Tax, Z, beta, vphi, C, L, B, Top10C, Top10A]

All the *aggregate* variables that are being tracked on a global level. If a variable is not listed here, you will not be able to recover it later. Since these are aggregate variables, they have dimensionality one.

.. code-block::

    distributions:
      dist: # a distribution named 'dist'
        # ordering matters. The ordering here is corresponds to the ordering of the axis of the distribution
        skills: # dim0
          type: exogenous
          grid_variables: [skills_grid, skills_stationary, skills_transition] # returns skills_grid, skills_stationary, skills_transition
          rho: 0.966
          sigma: 0.5
          n: 4
        a: # dim1
          type: endogenous
          grid_variables: a_grid # a variable named a_grid will be made available during decisions calls and distributions calls
          min: 0.0
          max: 100
          n: 40


The distributions block. Defines a distribution (here ``dist``) and all its dimensions. The information provided here will later be used to construct the distribution-forward-functions. If this is not supplied, Pizza assumes that you are providing a representative agent model.

.. code-block::

    decisions: # stage one
      inputs: [VaPrime] # additional to all aggregated variables defined in 'variables'
      calls:
        # these are executed subsequently, starting with the latest in time T. Each call takes the previous outputs as given
        ~ T = transfers(e_stationary, Div, Tax, e_grid)
        ~ we = wages(w, e_grid)
        ~ VaPrimeExp = e_tmat @ VaPrime
        ~ Va, a, c, n = hh(VaPrimeExp, a_grid, we, T, r-1, beta, eis, frisch, vphi)
        ~ ne = labor_supply(n, e_grid)
      outputs: [a,c,ne] # those are the ones stored for the following stages

The decisions block. Only relevant for heterogeneous agents models. It is important to correctly specify the dynamic inputs (here: marginals of the value function) and outputs, i.e. those variables that are needed as inputs for the distribution stage. Note that calls are evaluated one after another.

.. code-block::

    # stage three (optional): aux_equations
    # these can contain misc definitions that are available during the final stage. 
    # outputs from decisions, the grid variables, the distributions and 
    # aggregate variables from 'variables' (including those with "Prime", "Lag",...) are included by default
    # from here on the objects are _sequences_ with shapes either (1, horizon) or (n1, n2, horizon). Last dimension is always the time dimension
    aux_equations:
        ~ A = np.sum(D*a, axis=(0,1)) # note that we are summing over the first two dimensions e and a, but not the time dimension (dimension 2)
        ~ NE = np.sum(D*ne, axis=(0,1))
        ~ aggr_c = np.sum(D*c, axis=(0,1))
        # calculate consumption share of top-10% cumsumers
        ~ c_flat = c.reshape(-1,c.shape[-1]) # consumption flattend for each t
        ~ dist_sorted_c = jnp.take_along_axis(D.reshape(-1,c.shape[-1]), jnp.argsort(c_flat, axis=0), axis=0) # distribution sorted after consumption level, flattend for each t
        ~ top10c = jnp.where(jnp.cumsum(dist_sorted_c, axis=0) > .9, c_flat, 0.).sum(0)/c_flat.sum(axis=0) # must use `where` for jax. All sums must be taken over the non-time axis
        # calculate wealth share of top-10% wealth holders
        ~ a_flat = a.reshape(-1,a.shape[-1]) # assets flattend for each t
        ~ dist_sorted_a = jnp.take_along_axis(D.reshape(-1,a.shape[-1]), jnp.argsort(a_flat, axis=0), axis=0) # as above ...
        ~ top10a = jnp.where(jnp.cumsum(dist_sorted_a, axis=0) > .9, a_flat, 0.).sum(0)/a_flat.sum(axis=0)

Auxiliary equations. These are executed before the ``equations`` block, and can be used for all sorts of definitions that you may not want to keep track of. For heterogeneous agents models, this is a good place to do aggregation. Auxiliary equations are also executed subsequently.

.. code-block::

    equations: # final stage
        ~ C = aggr_c # definition
        ~ Top10C = top10c # definition
        ~ Top10A = top10a # definition
        ~ L = Yprod / Z # production function
        ~ Div = - w * L + (1 - psi*(pi/piSS - 1)**2/2)*Yprod # dividends
        ~ Y = (1 - psi*(pi/piSS - 1)**2/2)*Yprod # "effective" output
        ~ C = Y # market clearing
        ~ psi*(pi/piSS - 1)*pi/piSS = (1-theta) + theta*w + psi*betaPrime*C/CPrime*(piPrime/piSS - 1)*piPrime/piSS*YprodPrime/Yprod # NKPC
        ~ R = RsLag/pi # real rate ex-post
        ~ Rs = (Rstar*((pi/piSS)**phi_pi)*((Y/YLag)**phi_y))**(1-rho)*RsLag**rho # MP rule on shadow nominal rate
        ~ Rn = maximum(1, Rs) # ZLB
        ~ Tax = (R-1) * B # balanced budget
        ~ B = A # bond market clearing
        ~ NE = L # labor market clearing
        ~ beta = betaSS*(betaLag/betaSS)**rho_beta # exogenous beta
        ~ Rstar = RstarSS*(RstarLag/RstarSS)**rho_rstar # exogenous rstar
        ~ Z = ZSS*(ZLag/ZSS)**rho_Z # exogenous technology
        ~ vphi = vphiSS # actually a parameter

Equations. The central part of the yaml. Here you define the model equations, which will then be parsed such that each row must hold. Use ``xPrime`` for variable `x` in `t+1` and ``xLag`` for `t-1`. Access steady-state values with ``xSS``. You could specify a representative agent model with just stating the equations block (additional to variables). Importantly, ``equations`` are *not* executed subsequently but simultaneously!

.. code-block::

    parameters:
        eis: 0.5
        frisch: 0.5
        rho_e: 0.966
        sd_e: 0.5
        mu: 1.2
        theta: 6.
        psi: 96
        phi_pi: 2
        phi_y: 1.5
        rho: .8
        rho_beta: .8
        rho_rstar: .8
        rho_Z: .8

Define the model parameters. Note that for parameters that need to be fitted, it is better to define a variable instead (such as ``vphi`` above).

.. code-block::

    steady_state:
        fixed_values:
            Y: 1.0
            Z: 1.0
            pi: 1.0
            rstar: 1.005
            B: 5.6
            L: 1.0

        init_guesses:
            beta: 0.98
            vphi: 0.8
            w: 1/1.2
            Div: 1 - 1/1.2
            Tax: 0.028
            r: 1.005
            we: wages(w, e_grid)
            T: transfers(e_stationary, Div, Tax, e_grid)
            VaPrime: hh_init(a_grid, we, r, eis, T)[1]

The steady state block. ``fixed_values`` are those steady state values that are fixed ex-ante. ``init_guesses`` are initial guesses for steady state finding. Note that for heterogeneous agents models it is required that the initial value of inputs to the decisions-stage are given (here ``VaPrime``).
