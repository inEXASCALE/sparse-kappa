2-norm methods API
==================

Available functions
-------------------

* ``svds_cond``
* ``eigsh_cond``
* ``lobpcg_cond``
* ``power_method_cond``
* ``lanczos_cond``
* ``lanczos_unsym_cond``
* ``golub_kahan_cond``

Method guidance
---------------

* ``svds``: direct extremal singular-value baseline.
* ``eigsh``: the public API currently routes through a normal-operator estimate.
* ``lanczos`` / ``lanczos_unsym``: Lanczos-based interfaces.
* ``golub-kahan``: projected bidiagonalization estimate.
* ``power``: power/inverse iteration estimate.

See :doc:`../methods` and :doc:`../performance` for backend behavior and limits.

Example
-------

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.diag([1., 2., 10.])

   cond = cond_estimate(A, norm=2, method='golub-kahan', max_iter=40)
   print(cond)
