Architecture and reliability
============================

Module responsibilities
-----------------------

* ``sparse_kappa.py`` validates numerical inputs, selects estimators, and
  provides the function/class API.
* ``norm1/`` and ``norm2/`` implement estimator algorithms; ``solvers.py``
  supplies inverse actions and factorization caching.
* ``backend/`` is the PyTorch compatibility boundary. Its dense storage and
  solver behavior are documented in :doc:`performance`.
* ``gnn/data.py`` owns graph and dataset containers, ``gnn/features.py`` builds
  graph tensors, ``gnn/models.py`` owns neural message passing, and
  ``gnn/training.py`` owns configuration, training, inference, and checkpoints.
* ``examples/`` holds executable tutorials. Documentation includes these files
  directly, so code shown to users comes from the same source as the scripts.
* ``tests/`` protects numerical contracts, sparse feature extraction, edge
  ablations, and checkpoint compatibility.

Configuration and extension boundaries
--------------------------------------

``pyproject.toml`` is the authoritative package metadata/dependency definition;
``setup.py`` is a compatibility shim. Documentation tooling is an optional
extra. ``docs/source`` is the Sphinx source used by Read the Docs.

Numerical methods enter through the method maps in ``ConditionNumberEstimator``.
New methods should return the established result fields and document stopping
criteria and solver assumptions. Backend improvements must preserve supported
input forms, dtype/device behavior, and public conversion/solver contracts.

GNN extension points are callable feature extraction, PyTorch models, optimizer
and scheduler factories, loss functions, and validation callbacks. Custom
components must be supplied explicitly when reloading their checkpoints;
the library does not guess how to reconstruct arbitrary user code.

Reliability guarantees and boundaries
-------------------------------------

Numerical APIs reject empty, rectangular, and non-finite matrices. GNN
configuration rejects invalid training controls, data rejects invalid targets,
and training checks non-finite losses. Built-in GNN feature extraction rejects
complex data rather than silently discarding imaginary parts. Models preserve
edge mode and extractor settings across checkpoint round trips.

Checkpoint writes use a temporary file beside the destination and atomic
replacement, preventing a failed serialization from destroying an existing
checkpoint. Loading uses restricted ``weights_only=True`` deserialization.
Checkpoints retain inference configuration and loss history, but do not include
optimizer state, data splits, or a full reproducibility record.

These checks do not certify estimator accuracy, detect every singular matrix,
or guarantee recovery after arbitrary hardware/filesystem failure. Sparse
storage at graph construction also does not imply sparse numerical backend
operations. GPU validation and distribution-shift evaluation remain separate
from the CPU regression suite.

Development validation
----------------------

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests -q
   python examples/toy_models.py
   python examples/api_workflows.py
   python examples/gnn_edge_features.py --epochs 2
   python examples/gnn_edge_features.py --epochs 2 --no-edge-features
   python -m sphinx -W --keep-going -b html docs/source /tmp/sparse-kappa-docs

CI runs CPU tests, executes tutorials, and builds the documentation. Installing
CUDA on a standard hosted CPU runner is not a GPU test; real GPU checks require
hardware-backed runners and representative matrices.
