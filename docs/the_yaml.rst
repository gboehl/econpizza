
The *.yaml-file
-------------------

All relevant information is supplied via the yaml file. Let us have a look of the yaml of a hank model. For more general information about the YAML markup language and syntax, see 
`Wikipedia <https://en.wikipedia.org/wiki/YAML>`_.

.. code-block::

    functions_file: '../examples/hank_functions.py'

The relative path to a functions-file, which may provide additional functions. Here the functions ``transfers``, ``wages``, ``hh``, ``labor_supply`` and ``hh_init`` are defined in this file.

.. code-block::

    # these are available during all three stages (decisions, distributions, equations)
    definitions:
        - from numpy import log
        - from jax.experimental.host_callback import id_print as jax_print

General definitions and imports. These are available during all three stages (decisions, distributions, equations).

.. code-block::

    variables: [Div, Y, Yprod, w, pi, rn, r, rstar, Tax, Z, beta, vphi, C, L, B] 

All the *aggregate* variables that are being tracked on a global level. If a variable is not listed here, you will not be able to recover it later. Since these are aggregate variables, they have dimensionality one.

.. code-block::

    distributions:
      D: # a distribution named 'D'
        # ordering matters because it is the same ordering as the axis of all distribution variables
        e: # dim0
          type: exogenous
          grid_variables: [e_grid, e_stationary, e_tmat] # returns e_grid, e_stationary, e_tmat
          rho: 0.966
          sigma: 0.5
          # alternatively: if the previous 3 are not defined here, it is assumed that the grid_variables are available during the distribution stage (as an output of 'decisions')
          n: 4
        a: # dim1
          type: endogenous
          grid_variables: a_grid # a variable named a_grid will be made available during decisions calls and distributions calls
          min: 0.0
          max: 150
          n: 10

The distributions block. Defines a distribution (here ``D``) and all its dimensions. The information provided here will later be used to construct the distribution-forward-functions. If this is not supplied, Pizza assumes that you are providing a representative agent model.

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
        ~ Caggr = np.sum(D*c, axis=(0,1)) 

Auxiliary equations. These are executed before the ``equations`` block, and can be used for all sorts of definitions that you may not want to keep track of. For heterogeneous agents models, this is a good place to do aggregation. Auxiliary equations are also executed subsequently.

.. code-block::

    equations: # final stage
        ~ L = Yprod / Z
        ~ Div = - w * L + (1 - psi*(pi/piSS - 1)**2/2)*Yprod
        ~ C = (1 - psi*(pi/piSS - 1)**2/2)*Yprod
        ~ C = Y
        ~ C = Caggr
        ~ psi*(pi/piSS - 1)*pi/piSS = (1-theta) + theta*w + psi*betaPrime*C/CPrime*(piPrime/piSS - 1)*piPrime/piSS*YprodPrime/Yprod
        ~ r = rnLag/pi
        ~ rn = (rstar*((pi/piSS)**phi_pi)*((Y/YLag)**phi_y))**(1-rho)*rnLag**rho
        ~ Tax = (r-1) * B
        ~ B = A
        ~ NE = L
        ~ vphi = vphiSS # actually a parameter
        ~ beta = betaSS*(betaLag/betaSS)**rho_beta
        ~ rstar = rstarSS*(rstarLag/rstarSS)**rho_rstar
        ~ Z = ZSS*(ZLag/ZSS)**rho_Z

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