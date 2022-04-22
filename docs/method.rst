Some theory
===========

An small-scale nonlinear New Keynesian model with ZLB is provided `as an example <https://github.com/gboehl/econpizza/blob/master/econpizza/examples/nk.yaml>`_. Here is how to simulate it and plot some nonlinear impulse responses:

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}


or:

.. math::

    v_t =& v(\tilde{x}, v_{t+1})\\
    d_t =& d(\tilde{x}, \tilde{v}, d_{t-1})\\
    0 =& f(\tilde{x}, \tilde{d}, \tilde{v})
