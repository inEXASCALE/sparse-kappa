Methods and solver options
==========================

The methods below describe the estimator interface. Current backend wrappers
can use dense operations, and names do not establish a sparse runtime or
complexity bound. Read :doc:`performance` and validate on representative matrices.

1-norm methods
--------------

All estimate norm(A)*norm(A_inverse) using inverse actions. ``auto`` selects
``hager-higham``.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Name
     - Aliases
     - Role
   * - hager-higham
     - auto
     - Hager/Higham iteration
   * - hager
     - None
     - Hager iteration
   * - block-hager
     - block, higham
     - Block Higham-Tisseur inverse-norm estimate
   * - power
     - power-iteration
     - Power iteration
   * - oettli-prager
     - oettli, prager
     - Adaptive/random/hybrid sampling
   * - monte-carlo
     - sampling, random
     - Random inverse-action samples

Each is a numerical estimate, not an exact condition number for all matrices.
Solver accuracy and sampling affect the inverse-norm estimate.

2-norm methods
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Role / limitation
   * - svds
     - Extremal singular values directly, without forming A^H A in this path
   * - power
     - Power/inverse iteration; randomized stopping behavior
   * - lanczos
     - Lanczos-based estimation
   * - lanczos_unsym
     - Normal-operator estimate using A^H A
   * - golub-kahan
     - Bidiagonalization; a small projected problem may miss extreme values
   * - eigsh
     - Public interface currently routed through lanczos_unsym
   * - lobpcg
     - Block eigenvalue-based interface
   * - auto
     - Matrix size/symmetry/density selection heuristic

Normal-operator paths square the condition number and can lose relative
accuracy for small singular values. Direct singular-value comparison is useful
for small reference problems. The current ``svds`` backend wrapper performs a
dense SVD, and ``eigsh``/``lobpcg`` wrappers perform dense eigendecompositions.

Solver options
--------------

The 1-norm API accepts ``auto``, ``lu``, ``direct``, ``lsmr``, ``cg``, ``bicgstab``,
and ``gmres``. LU caches a factorization within one estimation call; direct
solves do not use that cache. Named iterative compatibility wrappers currently
use dense solves/least squares, so their nominal sparse iteration controls and
memory behavior should not be assumed. A future sparse backend should preserve
these interfaces while testing solver assumptions and convergence separately.

Example: compare estimation paths
---------------------------------

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 4.]])
   for method in ("hager-higham", "hager", "block-hager"):
       result = cond_estimate(A, norm=1, method=method, solver="lu", return_dict=True)
       print(method, result["condition_number"], "reference:", np.linalg.cond(A, p=1))

See :doc:`examples` for a complete comparison of both norms and supported input
forms. ``max_iter`` must be positive and ``tol`` finite and nonnegative;
backend dense decompositions may ignore those iteration controls.
