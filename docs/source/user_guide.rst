Using sparse-kappa
==================

Choose a numerical estimator when you need an estimate for a matrix now.
Choose a trained GNN when you have representative labels and have validated
prediction accuracy on the matrix distribution you intend to use.

What the condition number means
-------------------------------

For a nonsingular square matrix A,

.. math::

   \kappa_1(A)=\lVert A\rVert_1\lVert A^{-1}\rVert_1,
   \qquad
   \kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.

The 1-norm is the maximum absolute column sum. The 2-norm is the largest
singular value. A condition number near one indicates low normwise sensitivity;
a large value indicates potentially amplified perturbations in a linear system.
It does not measure actual solver error, and universal cutoffs for acceptable
conditioning depend on precision and the application.

A singular matrix has infinite condition number. Very small singular values
may be indistinguishable from zero at the chosen precision. Learned predictions
are not singularity tests and may even produce values below the mathematical
lower bound of one; outputs are not clamped to hide model error.

Construct a matrix and choose a norm
------------------------------------

Each example block is self-contained.

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.array([[3., -1., 0.], [-1., 3., -1.], [0., -1., 3.]])
   kappa_1 = cond_estimate(A, norm=1, solver="lu")
   kappa_2 = cond_estimate(A, norm=2)
   print("kappa_1:", kappa_1, "reference:", np.linalg.cond(A, p=1))
   print("kappa_2:", kappa_2, "reference:", np.linalg.cond(A, p=2))

The numerical API accepts nonempty square matrices with finite entries.
NumPy arrays, Python matrix lists, dense PyTorch tensors, native PyTorch COO/CSR
and SciPy sparse matrices are converted to the current backend representation.
Integer matrices are promoted to float64. Complex matrices are accepted by the
numerical API; GNN features currently support real matrices only. Use float64
for sensitive reference experiments and keep tensors on a consistent device.

Methods and solvers
-------------------

``method="auto"`` selects Hager-Higham for norm 1. For norm 2, the current
selection uses ``svds`` below size 5000, ``eigsh`` for larger matrices classified
as symmetric, ``golub-kahan`` when density is below 0.01, and otherwise
``lanczos``. These are heuristics, not accuracy or memory guarantees. The public
``eigsh`` option currently routes through ``lanczos_unsym`` on A^H A.

For norm 1, ``solver`` selects how inverse actions are computed. ``lu`` reuses a
factorization within the estimator invocation. Use ``cg`` only for matrices
satisfying its mathematical assumptions, and compare results with ``lu`` on
small test problems. The current backend compatibility implementations of
``lsmr``, ``lsqr``, ``cg``, ``gmres``, and ``bicgstab`` call dense solves rather
than implementing the named sparse iterations. See :doc:`performance`.

Diagnostics and method comparison
---------------------------------

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.diag([1., 2., 5., 10.])
   for method in ("svds", "power", "lanczos", "golub-kahan"):
       result = cond_estimate(A, norm=2, method=method, max_iter=100,
                              tol=1e-8, return_dict=True)
       print(method, result["condition_number"], result.get("converged"))

``cond_estimate`` normally returns a float. ``return_dict=True`` or
``verbose=True`` returns a dictionary. ``ConditionNumberEstimator.estimate()``
always returns a dictionary. Common metadata includes ``method``, ``norm``,
``condition_number``, ``iterations``, and ``converged``, with details varying by
method. A reported convergence flag is the algorithm's stopping result, not a
certificate of condition-number accuracy. Dense backend wrappers can report
iteration metadata even though the actual operation was a dense decomposition.

Randomized estimators may vary between runs. Seed ``torch`` and the backend RNG
when comparing them, repeat experiments when accuracy matters, and use a small
dense reference. Increasing iterations only helps methods that actually use
those controls. A 1-norm inverse-norm estimate is typically a lower estimate,
so a small result alone should not establish that a matrix is safe to solve.

Singular and nearly singular matrices
-------------------------------------

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   for A in (np.diag([1., 0.]), np.diag([1., 1e-8])):
       print("Estimate:", cond_estimate(A, norm=2, method="svds"))
       print("Reference:", np.linalg.cond(A))

The SVD example returns infinity for an exact zero singular value and about
1e8 for the second matrix. Solve-based norm-1 paths can raise on singular LU
factorization; backend solve fallbacks may use least squares. Do not interpret
a finite solve-based result as a proof of nonsingularity. Prefer singular-value
analysis on tractable problems when singularity is suspected.

Learned prediction workflow
---------------------------

Build accurate finite labels, split training/validation/test data, choose norm
and strategy, then compare edge modes on the same splits. ``use_edge_features``
can be True or False; it controls numeric information on edges, while default
node/global statistics retain values. Save the validated model and use
``GNNConditionEstimator.load`` to restore its mode. See :doc:`api/gnn` and the
complete scripts in :doc:`examples`.

Batch numerical estimates are a Python loop over matrices; GNN ``predict``
also accepts a list and evaluates one graph at a time. Current APIs do not
promise vectorized batch execution or concurrent mutation safety.
