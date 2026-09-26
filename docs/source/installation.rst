Installation
============

The runtime requires Python 3.8 or newer, NumPy, and PyTorch 2.0 or newer.
A GPU is optional. Dependency resolution selects versions compatible with your
Python version; documentation tools have their own requirements.

Install the package
-------------------

.. code-block:: bash

   python -m pip install sparse-kappa

Install this checkout in an isolated environment
------------------------------------------------

From the repository root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

On Windows, activate with ``.venv\Scripts\activate``. Editable installation
ensures Python imports the checkout you are changing. Run scripts using the same
interpreter used to install the package.

Development and documentation dependencies
------------------------------------------

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests -q
   python -m sphinx -W --keep-going -b html docs/source /tmp/sparse-kappa-docs

SciPy is optional for users; it is included in the development extra because
some tests/benchmarks use it. Sphinx, Furo, and MyST are documentation tools and
are not required for numerical estimation or GNN prediction.

Verify the installation
-----------------------

.. code-block:: python

   import numpy as np
   import torch
   import sparse_kappa
   from sparse_kappa import cond_estimate

   print("sparse-kappa:", sparse_kappa.__version__)
   print("PyTorch:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   print("kappa_2 (expected 10):", cond_estimate(np.diag([1., 2., 10.])))

CPU and CUDA
------------

Backend constructors choose CUDA when available and CPU otherwise. Existing
PyTorch tensors retain their device; keep all tensors in an operation on the
same device. For GNN workflows, ``TrainingConfig(device="cpu")`` and
``TrainingConfig(device="cuda")`` explicitly select execution. Start on CPU
with the examples in :doc:`quickstart`, then validate the CUDA environment for
your machine. Installing a CUDA wheel on a runner does not provide a GPU.

Read :doc:`performance` before processing large matrices: current numerical
backend inputs are converted to dense storage.
