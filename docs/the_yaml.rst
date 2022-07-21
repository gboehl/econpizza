
The *.yaml-file
---------------

All relevant information is supplied via the yaml file. For general information about the YAML markup language and its syntax see `Wikipedia <https://en.wikipedia.org/wiki/YAML>`_. The yaml files follow a simple structure:

1. define all variables and shocks
2. provide the nonlinear equations. Note that each equation starts with a `~`.
3. define the parameters
4. define the values of the parameters in the `steady_state` section
5. optionally provide some steady state values and/or values for initial guesses
6. optionally provide auxilliary equations that are not directly part of the nonlinear system (see the `yaml for the BH model <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/bh.yaml>`_)

I will first briefly discuss the yaml of the small scale representative agents model `above <https://econpizza.readthedocs.io/en/latest/quickstart.html#quickstart>`_ and then turn to more complex HANK model.

Representative agent models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file for the small scale NK model can be found `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. The first block (``variables`` and ``shocks``) is self explanatory:

.. code-block::

    variables: [y, c, pi, r, rn, beta, w, chi]
    shocks: [e_beta]

Note that it is not necessary to define shocks. You can also simply set the initial values of any (exogenous) state. Note that shocks are not yet implemented for heterogeneous agent models.

.. code-block::

    parameters: [ theta, psi, phi_pi, phi_y, rho, h, eta, rho_beta, chi ]

Use the ``parameters`` block to define any *parameters*. Parameters are treated the same as variables, but they are time invariant. During steady state search they are treated exactly equally. For this reason their values are provided in the `steady_state` block.

.. code-block::

    definitions: |
        from jax.numpy import log, maximum

The second block (``definitions``) defines general definitions and imports, which are available at all time.

.. code-block::

    equations:
        ~ w = chi*(c - h*cLag)*y**eta  # labor supply
        ~ 1 = r*betaPrime*(c - h*cLag)/(cPrime - h*c)/piPrime  # euler equation
        ~ psi*(pi/piSS - 1)*pi/piSS = (1-theta) + theta*w + psi*betaPrime*(c-h*cLag)/(cPrime-h*c)*(piPrime/piSS - 1)*piPrime/piSS*yPrime/y  # Phillips curve
        ~ c = (1-psi*(pi/piSS - 1)**2/2)*y  # market clearing
        ~ rn = (rSS*((pi/piSS)**phi_pi)*((y/yLag)**phi_y))**(1-rho)*rnLag**rho  # monetary policy rule
        ~ r = maximum(1, rn)  # zero lower bound on nominal rates
        ~ log(beta) = (1-rho_beta)*log(betaSS) + rho_beta*log(betaLag) + e_beta  # exogenous discount factor shock

``equations``. The most central part of the yaml. Here you define the model equations, which will then be parsed such that each equation prefixed by a `~` must hold. Use ``xPrime`` for variable `x` in `t+1` and ``xLag`` for `t-1`. Access steady-state values with ``xSS``. You could specify a representative agent model with just stating the equations block (additional to variables). Importantly, ``equations`` are *not* executed subsequently but simultaneously!
Note that you need one equation for each variable defined in ``variables``.

.. code-block::

    steady_state:
        fixed_values:
            # parameters
            theta: 6.  # demand elasticity
            psi: 96  # price adjustment costs
            phi_pi: 4  # monetary policy rule coefficient #1
            phi_y: 1.5  # monetary policy rule coefficient #2
            rho: .8  # interest rate smoothing
            h: .44  # habit formation
            eta: .33  # inverse Frisch elasticity
            rho_beta: .9  # autocorrelation of discount factor shock

            # steady state values
            beta: 0.9984
            y: .33
            pi: 1.02^.25

        init_guesses: # the default initial guess is always 1.1
            chi: 6

Finally, the ``steady_state`` block allows to fix parameters and, if desired, some steady state values, and provide initial guesses for others. Note that the default initial guess for any variable/parameter not specified here will be ``1.1``.


Heterogeneous agent models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us have a look of the yaml of a hank model we will discuss in `the tutorial <https://econpizza.readthedocs.io/en/latest/tutorial.html>`_. The file can be found `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank.yaml>`_. The first line reads:

.. code-block::

    functions_file: '../examples/hank_functions.py'

The relative path to a functions-file, which may provide additional functions. In this example, the file defines the functions ``transfers``, ``wages``, ``hh``, ``labor_supply`` and ``hh_init``.

.. code-block::

    # these are available during all three stages (decisions, distributions, equations)
    definitions: |
        from jax.numpy import log, maximum
        from jax.experimental.host_callback import id_print as jax_print

General definitions and imports (as above). These are available during all three stages (decisions, distributions, equations).

.. code-block::

    variables: [Div, Y, Yprod, w, pi, Rn, Rs, R, Rstar, Tax, Z, beta, vphi, C, L, B, Top10C, Top10A]

All the *aggregate* variables that are being tracked on a global level. If a variable is not listed here, you will not be able to recover it later. Since these are aggregate variables, they have dimensionality one.

.. code-block::

    parameters: [ eis, frisch, theta, psi, phi_pi, phi_y, rho, rho_beta, rho_rstar, rho_Z ]

Define the model parameters, as above.

.. code-block::

    distributions:
      dist: # the name of the first distribution
        # ordering matters. The ordering here is corresponds to the shape of the axis of the distribution
        skills: # first dimension
          type: exogenous
          grid_variables: [skills_grid, skills_stationary, skills_transition] # returns skills_grid, skills_stationary, skills_transition
          rho: 0.966
          sigma: 0.6
          n: 4
        a: # second dimension
          type: endogenous
          grid_variables: a_grid # a variable named a_grid will be made available during decisions calls and distributions calls
          min: 0.0
          max: 50
          n: 40

The distributions block. Defines a distribution (here ``dist``) and all its dimensions. The information provided here will later be used to construct the distribution-forward-functions. If this is not supplied, Pizza assumes that you are providing a representative agent model.

.. code-block::

    decisions: # stage one: iterating the decisions function backwards
      inputs: [VaPrime] # additional to all aggregated variables defined in 'variables'
      calls: |
        # these are executed subsequently, starting with the last in time T and then iterating forwards
        # Each call takes the previous outputs as given
        T = transfers(skills_stationary, Div, Tax, skills_grid)
        VaPrimeExp = skills_transition @ VaPrime
        Va, a, c = hh(VaPrimeExp, a_grid, skills_grid, w, n, T, R, beta, eis, frisch)
      # the 'outputs' values are stored for the following stages
      # NOTE: each output must have the same shape as the distribution (4,40)
      outputs: [a,c]


The decisions block. Only relevant for heterogeneous agents models. It is important to correctly specify the dynamic inputs (here: marginals of the value function) and outputs, i.e. those variables that are needed as inputs for the distribution stage. Note that calls are evaluated one after another.

.. code-block::

    # stage three (optional): aux_equations
    aux_equations: |
        A = jnp.sum(dist*a, axis=(0,1)) # note that we are summing over the first two dimensions e and a, but not the time dimension (dimension 2)
        aggr_c = jnp.sum(dist*c, axis=(0,1))
        # `dist` here corresponds to the dist from the *previous* period.


        # calculate consumption share of top-10% cumsumers
        c_flat = c.reshape(-1,c.shape[-1]) # consumption flattend for each t
        dist_sorted_c = jnp.take_along_axis(dist.reshape(-1,c.shape[-1]), jnp.argsort(c_flat, axis=0), axis=0) # distribution sorted after consumption level, flattend for each t
        top10c = jnp.where(jnp.cumsum(dist_sorted_c, axis=0) > .9, c_flat, 0.).sum(0)/c_flat.sum(axis=0) # must use `where` for jax. All sums must be taken over the non-time axis

        # calculate wealth share of top-10% wealth holders
        a_flat = a.reshape(-1,a.shape[-1]) # assets flattend for each t
        dist_sorted_a = jnp.take_along_axis(dist.reshape(-1,a.shape[-1]), jnp.argsort(a_flat, axis=0), axis=0) # as above
        top10a = jnp.where(jnp.cumsum(dist_sorted_a, axis=0) > .9, a_flat, 0.).sum(0)/a_flat.sum(axis=0)

Auxiliary equations. This again works exactly as for the representative agent model. These are executed before the ``equations`` block, and can be used for all sorts of definitions that you may not want to keep track of. For heterogeneous agents models, this is a good place to do aggregation. Auxiliary equations are also executed subsequently.

The distribution (``dist``) corresponds to the distribution **at the beginning of the period**, i.e. the distribution from last period. This is because the outputs of the decisions stage correspond to the asset holdings (on grid) at the beginning of the period, while the distribution calculated *from* the decision outputs holds for the next period.

.. code-block::

    equations: # final stage
        # definitions
        ~ C = aggr_c
        ~ Top10C = top10c
        ~ Top10A = top10a

        # firms
        ~ n = Yprod / Z # production function
        ~ Div = - w * n + (1 - psi*(pi/piSS - 1)**2/2)*Yprod # dividends
        ~ Y = (1 - psi*(pi/piSS - 1)**2/2)*Yprod # "effective" output
        ~ psi*(pi/piSS - 1)*pi/piSS = (1-theta) + theta*w + psi*piPrime/Rn*(piPrime/piSS - 1)*piPrime/piSS*YprodPrime/Yprod # NKPC

        # government
        ~ R = RsLag/pi # real rate ex-post
        ~ Rs = (Rstar*((pi/piSS)**phi_pi)*((Y/YLag)**phi_y))**(1-rho)*RsLag**rho # MP rule on shadow nominal rate
        ~ Rn = maximum(1, Rs) # ZLB
        ~ Tax = (R-1) * BLag # balanced budget

        # clearings
        ~ C = Y # market clearing
        ~ B = A # bond market clearing
        ~ w**frisch = n # labor market clearing

        # exogenous
        ~ beta = betaSS*(betaLag/betaSS)**rho_beta # exogenous beta
        ~ Rstar = RstarSS*(RstarLag/RstarSS)**rho_rstar # exogenous rstar
        ~ Z = ZSS*(ZLag/ZSS)**rho_Z # exogenous technology

Equations. This also works exactly as for representative agents models.

.. code-block::

    steady_state:
        fixed_values:
            # parameters:
            eis: 0.5
            frisch: 0.5
            theta: 6.
            psi: 96
            phi_pi: 1.5
            phi_y: .25
            rho: .8
            rho_beta: .9
            rho_rstar: .9
            rho_Z: .8

            # steady state
            Y: 1.0
            pi: 1.0
            beta: 0.97
            B: 5.6
            w: (theta-1)/theta
            n: w**frisch

        init_guesses:
            Rstar: 1.002
            Div: 1 - w
            Tax: 0.028
            R: Rstar
            VaPrime: hh_init(a_grid, skills_stationary)

The steady state block. ``fixed_values`` are those steady state values that are fixed ex-ante. ``init_guesses`` are initial guesses for steady state finding. Values are defined from the top to the bottom, so it is possible to use recursive definitions, such as `n: w**frisch`.

Note that for heterogeneous agents models it is required that the initial value of inputs to the decisions-stage are given (here ``VaPrime``).
