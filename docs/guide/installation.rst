Installation
============

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as typing

.. code-block:: bash

   pip install econpizza

in your terminal or Anaconda Prompt.

On Windows
----------

Econpizza needs the Python package `JAX <https://jax.readthedocs.io>`_ to be installed. This is not a problem for MacOS and Linux (the package will be automatically installed when installing Econpizza), but the time for JAX to fully support Windows has not yet come. Fortunately, there is help out there (see `this link <https://github.com/cloudhan/jax-windows-builder>`_ for the somewhat cryptic original reference). To install JAX, run

.. code-block:: bash

    pip install "jax[cpu]===0.3.25" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

*prior* to installing Econpizza. Econpizza should then run just fine (`proof <https://github.com/gboehl/econpizza/actions/runs/2579662335>`_).

In case you run into an error with `ptxas` (like `in this case <https://github.com/tensorflow/models/issues/7640>`_), a workaround is to disable CUDA by adding the following line **before** importing econpizza or JAX:

.. code-block:: python

    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
