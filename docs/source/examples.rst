Examples
========

Basic 2-norm estimation
-----------------------

.. code-block:: python

   from sparse_kappa.backend import sparse as sp
   from sparse_kappa import cond_estimate

   A = sp.random(2000, 2000, density=0.005, format='csr')
   cond = cond_estimate(A, norm=2, method='svds')
   print(f"kappa_2(A) = {cond:.3e}")

Compare multiple methods
------------------------

.. code-block:: python

   methods = ['svds', 'lanczos', 'golub-kahan']
   for method in methods:
       value = cond_estimate(A, norm=2, method=method)
       print(f"{method:12s} -> {value:.3e}")

1-norm with LU solver
---------------------

.. code-block:: python

   result = cond_estimate(
       A,
       norm=1,
       method='hager-higham',
       solver='lu',
       return_dict=True,
   )
   print(result['condition_number'])
   print(result['solver_info'])

GNN strategy 1: inverse-norm prediction
---------------------------------------

.. code-block:: python

   from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator

   train_samples = [
       {'matrix': A0, 'condition_number': 10.2, 'norm_A': 2.5},
       {'matrix': A1, 'condition_number': 15.8, 'norm_A': 3.1},
   ]

   config = make_gnn_strategy_config(norm=1, strategy=1, epochs=20, lr=1e-3)
   estimator = train_gnn_strategy_estimator(train_samples, norm=1, strategy=1, config=config)
   result = estimator.predict(A_test, return_dict=True)
   print(result['condition_number'], result['norm_A'], result['norm_Ainv'])

GNN strategy 2: direct condition prediction
-------------------------------------------

.. code-block:: python

   from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator

   train_samples = [
       {'matrix': A0, 'condition_number': 10.2},
       {'matrix': A1, 'condition_number': 15.8},
   ]

   config = make_gnn_strategy_config(norm=2, strategy=2, epochs=20, lr=1e-3)
   estimator = train_gnn_strategy_estimator(train_samples, norm=2, strategy=2, config=config)
   pred = estimator.predict(A_test)
   print(pred)
