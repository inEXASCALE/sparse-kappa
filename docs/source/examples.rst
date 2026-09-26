Runnable examples
=================

Every script below defines its imports, matrices, and parameters. There are no
undefined ``A``, ``A_test``, or training labels. Install the checkout first:

.. code-block:: bash

   python -m pip install -e .

Toy models with reference answers
---------------------------------

``examples/toy_models.py`` covers five nonsingular families and one singular
case. It compares both norms with ``torch.linalg.cond`` on these small matrices.
Dense reference calculations are suitable for verification, not large-scale
production workloads. Numerical estimates need not be identical to the reference.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Matrix
     - Reference kappa_1
     - Reference kappa_2
   * - Identity (4 by 4)
     - 1
     - 1
   * - Diagonal [1, 2, 4, 10]
     - 10
     - 10
   * - 1D Poisson (8 by 8)
     - 40
     - Approximately 32.1634
   * - Upper triangular [[1,2,0],[0,3,1],[0,0,4]]
     - 5
     - Approximately 5.31406
   * - Diagonal [1, 1e-4, 1e-8]
     - 1e8
     - 1e8
   * - Diagonal [1, 0]
     - Not used in the solve-based demo
     - Infinity

.. literalinclude:: ../../examples/toy_models.py
   :language: python
   :caption: examples/toy_models.py

Supported inputs, diagnostics, and method comparison
----------------------------------------------------

NumPy arrays, dense PyTorch tensors, native PyTorch COO/CSR tensors, backend
matrices, and optionally SciPy CSR matrices can enter the numerical API.
Conversion currently creates dense backend storage. Floating point ``float64``
is a useful starting point for reference experiments.

``return_dict=True`` exposes method-dependent diagnostics. Always inspect the
actual keys for your chosen method; not every method reports singular values,
solver diagnostics, or the same convergence criterion. ``max_iter`` and ``tol``
control supported iterative paths; dense backend wrappers may ignore them.

.. literalinclude:: ../../examples/api_workflows.py
   :language: python
   :caption: examples/api_workflows.py

Learning with or without matrix-element edge features
-----------------------------------------------------

This script supplies actual labels, disjoint training/validation/test matrices,
a fixed CPU seed, both strategy choices, batch prediction, and a checkpoint
round trip. Try the default and ``--no-edge-features`` under the same settings.
Five epochs on six matrices only verifies the workflow; it does not establish
accuracy or generalization to other matrix distributions.

.. code-block:: bash

   python examples/gnn_edge_features.py --epochs 5
   python examples/gnn_edge_features.py --epochs 5 --no-edge-features
   python examples/gnn_edge_features.py --norm 1 --strategy 1 --epochs 5
   python examples/gnn_edge_features.py --norm 2 --strategy 1 --epochs 5 --no-edge-features

.. literalinclude:: ../../examples/gnn_edge_features.py
   :language: python
   :caption: examples/gnn_edge_features.py

Practical validation
--------------------

For your own matrices, first use the numerical API on representative small
problems. Compare with a dense reference and repeat randomized estimators with
several seeds. For learned models, report relative condition-number error as
well as loss in log target space, keep test matrices out of training, and assess
all matrix families, sizes, and condition ranges expected in deployment.

Multiplying a nonsingular matrix by a nonzero scalar leaves its condition number
unchanged. Adding a diagonal shift or preconditioning changes the problem:
label and report the transformed matrix explicitly. Singular inputs are not
valid positive finite GNN training labels, and a finite GNN output does not
certify nonsingularity.
