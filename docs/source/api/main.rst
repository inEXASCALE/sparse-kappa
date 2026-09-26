Main API
========

Primary entry points
--------------------

* ``cond_estimate(A, norm=2, method='auto', ...)``
  Main convenience function for condition number estimation.
* ``ConditionNumberEstimator(A, norm=2, method='auto', ...)``
  Class-based estimator for richer method selection and internal property handling.

``cond_estimate`` parameters
----------------------------

Common options:

* ``A``: input square matrix (sparse preferred).
* ``norm``: ``1`` or ``2``.
* ``method``: algorithm name (for example ``svds``, ``lanczos``, ``hager-higham``).
* ``max_iter`` / ``tol``: convergence controls.
* ``return_dict``: when ``True``, returns diagnostics in addition to the estimate.

Typical usage
-------------

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate

   A = sp.csr_matrix([[1., 0., 0.], [0., 2., 0.], [0., 0., 10.]])

   cond = cond_estimate(A)  # auto-select 2-norm method
   detailed = cond_estimate(A, norm=2, method='svds', return_dict=True)

   print(cond)
   print(detailed['condition_number'], detailed['iterations'])

``ConditionNumberEstimator`` workflow
-------------------------------------

.. code-block:: python

   from sparse_kappa import ConditionNumberEstimator
   from sparse_kappa.backend import sparse as sp

   A = sp.csr_matrix([[1., 0.], [0., 10.]])
   estimator = ConditionNumberEstimator(A, norm=1, method='hager-higham', solver='lu')
   result = estimator.estimate()
   print(result['method'], result['condition_number'])

Input and output contract
-------------------------

Input must be a nonempty finite square matrix. NumPy arrays, matrix lists,
PyTorch dense/COO/CSR tensors, backend matrices, and SciPy sparse matrices are
converted to the dense current backend. Integer data is promoted to float64.
``max_iter`` must be a positive integer and ``tol`` finite and nonnegative.
Invalid dimensions, norms, or methods raise ``ValueError``.

A float is returned unless ``return_dict=True`` or ``verbose=True``. Result
fields are method-dependent; ``converged`` is not an accuracy certificate.
``svds`` computes extremal singular values directly, avoiding A^H A in this
path. Other methods may use normal operators and have different numerical
behavior. Read :doc:`../user_guide` for singular-matrix behavior and
:doc:`../performance` for storage and dense solver limits.
