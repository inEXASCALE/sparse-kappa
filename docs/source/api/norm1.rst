1-norm methods API
==================

Available functions
-------------------

* ``hager_norm1_cond``
* ``hager_higham_norm1``
* ``block_higham_tisseur_norm1``
* ``power_iteration_norm1``
* ``oettli_prager_norm1``
* ``monte_carlo_norm1``

When to choose each method
--------------------------

* ``hager-higham``: default inverse-norm estimation baseline.
* ``block-hager`` / ``higham``: block variant.
* ``power``: power iteration estimate.
* ``oettli-prager``: adaptive/random/hybrid sampling.
* ``monte-carlo``: stochastic baseline.

See :doc:`../methods` and :doc:`../performance` for backend behavior and limits.

Example
-------

.. code-block:: python

   import numpy as np
   from sparse_kappa import cond_estimate

   A = np.diag([1., 2., 10.])

   result = cond_estimate(
       A,
       norm=1,
       method='hager-higham',
       solver='lu',
       return_dict=True,
   )
   print(result['condition_number'])
