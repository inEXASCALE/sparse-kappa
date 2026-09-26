Performance and implementation limits
=====================================

Current backend behavior
------------------------

The public backend exposes CSR/COO-like constructors, but ``TorchSparseMatrix``
currently stores a dense PyTorch tensor. Numerical API conversion, matrix
products, LU, and backend SVD/eigenvalue wrappers may all allocate dense arrays.
The wrappers for several iterative solvers use dense solve/least-squares.
Reducing input density does not reduce the matrix storage allocation.

A float64 n-by-n matrix alone needs 8*n*n bytes, before copies, factorization
workspace, intermediate operators, and vectors. A 10,000-by-10,000 matrix is
about 800 MB for one copy; the working set can be much larger. Changing the
estimator name does not eliminate this allocation. Large sparse production
workloads need a genuinely sparse backend and separate validation.

The GNN graph extractor handles native PyTorch and SciPy sparse matrices
without dense conversion. Extraction uses nonzeros plus row/column statistics
(and sparse coalescing); message passing stores two directed edges per nonzero.
This does not make every GNN operation sparse: inverse-norm strategies and
``return_dict=True`` still compute matrix norms, and the 2-norm uses dense SVD.
Direct scalar prediction avoids that norm computation.

Measure accuracy together with runtime
--------------------------------------

LU caching amortizes factorization over repeated inverse actions within one
estimation call. Whether it is faster depends on the matrix, device, method,
and backend. No universal speedup or best size threshold is claimed.

This complete benchmark synchronizes CUDA when present and reports both
estimation error and elapsed time. It uses a small, well-defined matrix.

.. code-block:: python

   import time
   import torch
   from sparse_kappa import cond_estimate

   device = "cuda" if torch.cuda.is_available() else "cpu"
   A = torch.diag(torch.linspace(1., 100., 32, dtype=torch.float64, device=device))
   reference = float(torch.linalg.cond(A))
   for method in ("svds", "power", "lanczos", "golub-kahan"):
       cond_estimate(A, norm=2, method=method)  # Warm up this path.
       if device == "cuda":
           torch.cuda.synchronize()
       start = time.perf_counter()
       estimate = cond_estimate(A, norm=2, method=method)
       if device == "cuda":
           torch.cuda.synchronize()
       elapsed = time.perf_counter() - start
       print(method, "seconds:", elapsed,
             "relative error:", abs(estimate / reference - 1))

Report hardware, dtype, matrix family/size, nonzeros, solver, tolerance, and
repeat statistics with benchmarks. CPU toy examples validate API contracts;
they do not establish CUDA performance or accuracy on arbitrary large matrices.
