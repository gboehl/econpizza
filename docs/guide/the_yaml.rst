
Specifying models
-----------------

Models are specified in a YAML file, which uses the YAML markup language.
The YAML format is widely used due to its intuitive handling,
for example for configuration files or in applications where data is being stored or transmitted which should be human readable.
For general information about the format and its syntax see `Wikipedia <https://en.wikipedia.org/wiki/YAML>`_.

The YAML file contains all relevant information from model equations, variable declarations and steady state values.
Models specified as a YAML files can be parsed into a :class:`econpizza.PizzaModel`
using :py:meth:`econpizza.parse` or :py:meth:`econpizza.load`.
An instance of :class:`econpizza.PizzaModel` holds all the relevant information and functionality of the model.

The YAML file
^^^^^^^^^^^^^

The YAML files follow a simple structure:

1. list all variables, parameters and shocks
2. provide the nonlinear equations. Note that each equation starts with a ``~``.
3. define the values of the parameters and fixed steady state values in the ``steady_state`` section
4. optionally provide auxiliary equations that are not directly part of the nonlinear system
5. optionally provide initial guesses for all other steady state values and parameters

I will first briefly discuss `the YAML <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yml>`_ of the small scale *representative* agents NK model `from the quickstart tutorial <../tutorial/quickstart.ipynb>`_ and then turn to a more complex HANK model.
A collection of examples is provided `with the package <https://github.com/gboehl/econpizza/tree/master/econpizza/examples>`_.

YAML: representative agent models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GitHub version of the YAML file for the small scale NK model can be found `here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yml>`_. The first block (``variables`` and ``shocks``) is self explanatory:

.. code-block:: yaml

    variables: [y, c, pi, r, rn, beta, w, chi]
    shocks: [e_beta]

Note that it is not necessary to define shocks. You can also simply set the initial values of any (exogenous) state.

.. code-block:: yaml

    parameters: [ theta, psi, phi_pi, phi_y, rho, h, eta, rho_beta, chi ]

Use the ``parameters`` block to define any *parameters*. Parameters are treated the same as variables, but they are time invariant. During steady state search they are treated exactly equally. For this reason their values are provided in the ``steady_state`` block.

.. code-block:: yaml

    definitions: |
        from jax.numpy import log, maximum

The second block (``definitions``) defines general definitions and imports, which are available at all stages.

.. code-block:: yaml

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

.. code-block:: yaml

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


YAML: heterogeneous agent models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us have a look of the YAML of a hank model we will discuss in `the tutorial <../tutorial/hank1.ipynb>`_.
The GitHub version of the file (`link <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank_with_comments.yml>`_) also contains exhaustive additional comments. The first line reads:

.. code-block:: yaml

    functions_file: 'hank_functions.py'

The relative path to a functions-file, which may provide additional functions.
The GitHub version of the functions file for this model `can be found here <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank_functions.py>`_.
In this example, the file defines the functions ``transfers``, ``wages``, ``hh``, ``labor_supply`` and ``hh_init``.

.. code-block:: yaml

    definitions: |
        from jax.numpy import log, maximum
        from econpizza.tools import percentile, jax_print

General definitions and imports (as above). These are available during all three stages (decisions, distributions, equations).
We will use the ``percentile`` function to get some distributional statistics. ``jax_print`` is a JAX-jit-able print function that can be used during ``call`` stages for debugging.

.. code-block:: yaml

    variables: [ div, y, y_prod, w, pi, R, Rn, Rr, Rstar, tax, z, beta, C, n, B, Top10C, Top10A ]

All the *aggregate* variables that are being tracked on a global level. If a variable is not listed here, you will not be able to recover it later. Since these are aggregate variables, they have dimensionality one.

.. code-block:: yaml

    parameters: [ sigma_c, sigma_l, theta, psi, phi_pi, phi_y, rho, rho_beta, rho_r, rho_z ]
    shocks: [ e_beta, e_rstar, e_z ]

Define the model parameters and shocks, as above.


.. code-block:: yaml

    distributions:
      # the name of the first distribution
      dist:
        # ordering matters. The ordering here is corresponds to the shape of the axis of the distribution
        # the naming of the dimensions (skills, a) is arbitrary
        skills:
          # first dimension
          type: exogenous_rouwenhorst
          rho: 0.966
          sigma: 0.6
          n: 4
        a:
          # second dimension. Endogenous distribution objects require inputs from the decisions stage. An object named 'a' assumes that the decisions stage returns a variable named 'a'
          type: endogenous_log
          min: 0.0
          max: 50
          n: 50

The distributions block. Defines a distribution (here ``dist``) and all its dimensions. The information provided here is used to construct the distribution-forward-functions. If this is not supplied, econpizza assumes that you are providing a representative agent model.

Exogenous grids are grids for idiosyncratic shocks. A grid type "exogenous_rouwenhorst" requires the parameters ``rho``, ``sigma`` and ``n``. Alternatively, a grid type "exogenous_generic" only needs ``n`` and expects the grid variable and the transition matrix to be defined somewhere else.

Endogenous grids are grids for idiosyncratic state variables. A grid type "exogenous_log" requires the parameters ``min``, ``max`` and ``n``. Based on these, a log grid will be created. Alternatively, a grid type "endogenous_generic" only needs ``n`` and expects the grid variable to be defined somewhere else.

.. code-block:: yaml

    decisions:
      # define the multidimensional input "WaPrime", in addition to all aggregated variables (defined in 'variables')
      inputs: [WaPrime]
      # calls executed during the decisions stage
      calls: |
        # these functions are defined in functions_file
        tfs = transfers(skills_stationary, div, tax, skills_grid)
        WaPrimeExp = skills_transition @ WaPrime
        Wa, a, c = egm_step(WaPrimeExp, a_grid, skills_grid, w, n, tfs, Rr, beta, sigma_c, sigma_l)
      # the 'outputs' values are stored for the following stages
      outputs: [a,c]

The decisions block. Only relevant for heterogeneous agents models. It is important to correctly specify the dynamic inputs (here: marginals of the value function) and outputs, i.e. those variables that are needed as inputs for the distribution stage. Note that calls are evaluated one after another.

.. code-block:: py

    aux_equations: |
        # `dist` here corresponds to the dist *at the beginning of the period*
        aggr_a = jnp.sum(dist*a, axis=(0,1))
        aggr_c = jnp.sum(dist*c, axis=(0,1))
        # calculate consumption and wealth share of top-10%
        top10c = 1 - percentile(c, dist, .9)
        top10a = 1 - percentile(a, dist, .9)

Auxiliary equations. This again works exactly as for the representative agent model. These are executed before the ``equations`` block, and can be used for all sorts of definitions that you may not want to keep track of. For heterogeneous agents models, this is a good place to do aggregation. Auxiliary equations are also executed subsequently.

The distribution (``dist``) corresponds to the distribution **at the beginning of the period**, i.e. the distribution from last period. This is because the outputs of the decisions stage correspond to the asset holdings (on grid) at the beginning of the period, while the distribution calculated *from* the decision outputs holds for the next period.

.. code-block:: yaml

    # final/main stage: aggregate equations
    equations:
        # definitions
        ~ C = aggr_c
        ~ Top10C = top10c
        ~ Top10A = top10a

        # firms
        ~ n = y_prod/z # production function
        ~ div = -w*n + (1 - psi*(pi/piSS - 1)**2/2)*y_prod # dividends
        ~ y = (1 - psi*(pi/piSS - 1)**2/2)*y_prod # "effective" output
        ~ psi*(pi/piSS - 1)*pi/piSS = (1-theta) + theta*w + psi*piPrime/R*(piPrime/piSS - 1)*piPrime/piSS*y_prodPrime/y_prod # NKPC

        # government
        ~ tax = (Rr-1)*BLag # balanced budget
        ~ Rr = RLag/pi # real ex-post bond return
        ~ Rn = (Rstar*((pi/piSS)**phi_pi)*((y/yLag)**phi_y))**(1-rho)*RnLag**rho # MP rule on shadow nominal rate
        ~ R = maximum(1, Rn) # ZLB

        # clearings
        ~ C = y # market clearing
        ~ B = aggr_a # bond market clearing
        ~ n**sigma_l = w # labor market clearing

        # exogenous
        ~ beta = betaSS*(betaLag/betaSS)**rho_beta*exp(e_beta) # exogenous beta
        ~ Rstar = RstarSS*(RstarLag/RstarSS)**rho_r*exp(e_rstar) # exogenous rstar
        ~ z = zSS*(zLag/zSS)**rho_z*exp(e_z) # exogenous technology

Equations. This also works exactly as for representative agents models.

.. code-block:: yaml

    steady_state:
        fixed_values:
            # parameters:
            sigma_c: 2 # intertemporal elasticity of substitution
            sigma_l: 2 # inverse Frisch elasticity of labour supply
            theta: 6. # elasticity of substitution
            psi: 60. # parameter on the costs of price adjustment
            phi_pi: 1.5 # Taylor rule coefficient on inflation
            phi_y: 0.1 # Taylor rule coefficient on output
            rho: 0.8 # persistence in (notional) nominal interest rate
            rho_beta: 0.9 # persistence of discount factor shock
            rho_r: 0.9 # persistence of MP shock
            rho_z: 0.9 # persistence of technology shocks

            # steady state
            y: 1.0 # effective output
            y_prod: 1.0 # output
            C: 1.0 # consumption
            pi: 1.0 # inflation
            beta: 0.98 # discount factor
            B: 5.6 # bond supply
            # definitions can be recursive: theta is defined above
            w: (theta-1)/theta # wages
            n: w**(1/sigma_l) # labor supply
            div: 1 - w*n # dividends
            z: y/n # technology

        init_guesses:
            Rstar: 1.002 # steady state target rage
            Rr: Rstar # steady state real rage
            Rn: Rstar # steady state notional rage
            R: Rstar # steady state nominal rage
            tax: 0.028
            WaPrime: egm_init(a_grid, skills_stationary)

The steady state block. ``fixed_values`` are those steady state values that are fixed ex-ante. ``init_guesses`` are initial guesses for steady state finding. Values are defined from the top to the bottom, so it is possible to use recursive definitions, such as ``n: w**frisch``.

Note that for heterogeneous agents models it is required that the initial value of inputs to the decisions-stage are given (here ``WaPrime``).

.. note::

   `Econpizza` is written in `JAX <https://jax.readthedocs.io>`_, which is a machine learning framework for Python developed by Google. JAX provides automatic differentiation and just-in-time compilation ("jitting"), which makes the package fast and robust.
   However, running jitted JAX code brings along a few limitations. Check the `common gotchas in JAX <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_ for details.


Model parsing
^^^^^^^^^^^^^

Models specified as a YAML files can be parsed and loaded using :py:meth:`econpizza.parse` and :py:meth:`econpizza.load`.

.. autofunction:: econpizza.parse

This returns a dictionary containing all the informations provided in the YAML file. Parsing before loading allows to change some features of the model manually. The dictionary can then be forwarded to :py:meth:`econpizza.load`:

.. autofunction:: econpizza.load

If desired, :meth:`econpizza.load` can also parse the YAML-file directly. The function then returns an instance of
:class:`econpizza.PizzaModel`, which holds all the relevant information and functionality of the model:

.. autoclass:: econpizza.PizzaModel
