Solver API
==========

Supported solver families
-------------------------

The solver subsystem is used mainly by 1-norm estimators:

* ``LUSolver``: cached LU-style solves for repeated right-hand sides.
* ``LSMRSolver``: iterative least-squares minimal residual method.
* ``CGSolver``: conjugate gradient for SPD-like systems.
* ``BiCGSTABSolver``: stabilized BiCG for non-symmetric systems.
* ``GMRESSolver``: robust Krylov fallback for difficult non-symmetric systems.
* ``DirectSolver`` and ``AutoSolver``: practical wrappers for direct/automatic selection.

Factory function
----------------

Use ``create_solver`` for method-driven selection.

.. code-block:: python

   import torch
   from sparse_kappa import create_solver
   from sparse_kappa.backend import sparse as sp

   A = sp.csr_matrix(torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64)))
   b = torch.ones(3, dtype=torch.float64)

   solver = create_solver(A, 'lu')
   x = solver.solve(b)
   print(solver.info())

   iterative = create_solver(A, 'lsmr', atol=1e-4, maxiter=80)
   x2 = iterative.solve(b)

Backend contract
-----------------

The names above describe the solver interfaces. The current backend implements
several iterative wrappers via dense solve/least-squares; they do not yet
provide the nominal sparse iterative behavior. LU solves reuse a cached dense
factorization. Inputs and right-hand sides must use compatible dtype/device.
See :doc:`../performance` before relying on large-matrix memory behavior.
