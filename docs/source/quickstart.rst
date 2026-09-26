Quick start: estimate a condition number
========================================

No training dataset or GPU is needed for numerical estimation. Each code block
below is independently runnable after ``python -m pip install sparse-kappa``
(or ``python -m pip install -e .`` for this checkout).

Diagonal matrix: a known answer
-------------------------------

For a nonsingular diagonal matrix, both condition numbers equal the largest
absolute diagonal entry divided by the smallest: here the answer is **10**.

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.diag([1.0, 2.0, 4.0, 10.0])
   print(cond_estimate(A, norm=2))
   print(cond_estimate(A, norm=1, method="hager-higham", solver="lu"))
   print("Reference:", np.linalg.cond(A, p=2))

A discretized 1D Poisson operator
---------------------------------

The tridiagonal matrix with diagonal 2 and off-diagonal -1 is symmetric positive
definite. With eight interior points, its 2-norm condition number is about
**32.1634**. This example constructs the whole problem; no external data is needed.

.. code-block:: python

   import math
   import torch
   from sparse_kappa import cond_estimate
   from sparse_kappa.backend import sparse as sp

   n = 8
   dense = 2 * torch.eye(n, dtype=torch.float64)
   dense += torch.diag(-torch.ones(n - 1, dtype=torch.float64), diagonal=1)
   dense += torch.diag(-torch.ones(n - 1, dtype=torch.float64), diagonal=-1)
   A = sp.csr_matrix(dense)
   result = cond_estimate(A, norm=2, method="svds", return_dict=True)
   theta = math.pi / (n + 1)
   reference = (1 + math.cos(theta)) / (1 - math.cos(theta))
   print("Estimate:", result["condition_number"])
   print("Analytic reference:", reference)
   print("Method:", result["method"], "Converged:", result["converged"])
   print("Singular values:", result["sigma_max"], result["sigma_min"])

Nonsymmetric matrix and the class API
-------------------------------------

A 1-norm estimate uses repeated solves, so the solver choice matters. LU is a
useful baseline when its dense factorization fits in memory. The class API
retains the matrix and method options; ``estimate()`` always returns diagnostics.

.. code-block:: python

   import numpy as np
   from sparse_kappa import ConditionNumberEstimator, cond_estimate

   A = np.array([[1., 2., 0.], [0., 3., 1.], [0., 0., 4.]])
   estimator = ConditionNumberEstimator(A, norm=1, method="hager-higham", solver="lu")
   result = estimator.estimate()
   print("Estimated kappa_1:", result["condition_number"])
   print("Reference kappa_1:", np.linalg.cond(A, p=1))
   print("Estimated kappa_2:", cond_estimate(A, norm=2))

Run the complete examples
-------------------------

From the repository root, after installing this checkout:

.. code-block:: bash

   python examples/toy_models.py
   python examples/api_workflows.py
   python examples/gnn_edge_features.py --epochs 5
   python examples/gnn_edge_features.py --epochs 5 --no-edge-features
   python examples/gnn_edge_features.py --norm 1 --strategy 1 --epochs 5

Read :doc:`examples` for all toy matrices and reference values, :doc:`api/main`
for the numerical API, and :doc:`api/gnn` for learned prediction.

.. note::

   The current numerical backend stores matrices densely, even when the input
   uses CSR/COO. Start with modest matrix sizes and read :doc:`performance`
   before scaling up. Iterative estimator names do not imply sparse storage
   or sparse iterative solves in this backend.
