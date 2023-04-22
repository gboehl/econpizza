Installation
============

On Linux & MacOS
----------------

Installing the `repository version <https://pypi.org/project/econpizza/>`_ from PyPi is as simple as typing

.. code-block:: bash

   pip install econpizza

in your terminal or Anaconda Prompt.

On Windows
----------

Econpizza depends on the Python package `JAX <https://jax.readthedocs.io>`_. This is not a problem for MacOS and Linux (the package will be automatically installed when installing Econpizza), prebuild binaries do not yet exist for Windows. Fortunately, some unofficial binaries exist (see `this link <https://github.com/cloudhan/jax-windows-builder>`_).
One way to install JAX on Windows is thus to run

.. code-block:: bash

    pip install "jax[cpu]===0.3.25" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

in your terminal or Anaconda Prompt, *prior* to installing Econpizza. Then continue as for Linux and Mac. Econpizza then just runns fine (`proof <https://github.com/gboehl/econpizza/actions/runs/2579662335>`_). Alternatively to using the unofficial binaries, you can consult the `JAX installation guide <https://github.com/google/jax#installation>`_.

In case you run into an error with `ptxas` (like `in this case <https://github.com/tensorflow/models/issues/7640>`_), a workaround is to disable CUDA by running the following line **before** importing econpizza or JAX in your python code:

.. code-block:: python

    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
