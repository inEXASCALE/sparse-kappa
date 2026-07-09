GNN API
=======

Main components
---------------

* ``TrainingConfig``: training/task configuration.
* ``GNNConditionEstimator``: fit, evaluate, save/load, and predict interface.
* ``train_gnn_condition_estimator``: one-shot training helper for explicit targets.
* ``train_gnn_strategy_estimator``: one-shot training helper for the bundled strategy workflows.
* ``make_gnn_strategy_config``: build a strategy-aware ``TrainingConfig``.
* ``SparseMatrixGNN``: neural model implementation.
* ``MatrixConditionDataset`` / ``MatrixGraph``: graph dataset structures.

Strategy workflows
------------------

The GNN estimator supports the two training/prediction strategies used by the
reference scripts for both 1-norm and 2-norm condition numbers.

* ``strategy=1`` trains the model on ``log10(||A^{-1}||)``. Prediction computes
  ``kappa(A) = ||A|| * ||A^{-1}||`` for the configured norm.
* ``strategy=2`` trains the model on ``log10(kappa(A))`` and predicts the
  condition number directly.

``strategy=1`` accepts either explicit ``norm_Ainv`` labels or ordinary
``condition_number`` labels. When only ``condition_number`` is provided,
``sparse-kappa`` derives the inverse-norm target from ``condition_number / ||A||``.
If you already computed ``||A||`` during dataset generation, pass it as ``norm_A``,
``matrix_norm``, or ``norm`` in each sample to avoid recomputing it.

Training examples
-----------------

.. code-block:: python

   from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator

   train_samples = [
       {"matrix": A0, "condition_number": 12.0, "norm_A": 4.0},
       {"matrix": A1, "condition_number": 18.0, "norm_A": 6.0},
   ]

   # Strategy 1, 1-norm: learn ||A^{-1}||_1, then multiply by ||A||_1.
   config = make_gnn_strategy_config(norm=1, strategy=1, epochs=50, lr=1e-3)
   estimator = train_gnn_strategy_estimator(train_samples, norm=1, strategy=1, config=config)

   # Strategy 2, 2-norm: learn kappa_2(A) directly.
   direct_config = make_gnn_strategy_config(norm=2, strategy=2, epochs=50)
   direct_estimator = train_gnn_strategy_estimator(
       train_samples,
       norm=2,
       strategy=2,
       config=direct_config,
   )

Prediction example
------------------

.. code-block:: python

   result = estimator.predict(A_test, return_dict=True)
   print(result["condition_number"])
   print(result["norm_A"], result["norm_Ainv"], result["strategy"])

Backward-compatible target API
------------------------------

You can still configure the lower-level target directly.

.. code-block:: python

   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator

   train_samples = [
       {"matrix": A, "condition_number": 12.0, "norm_Ainv": 0.6},
   ]

   config = TrainingConfig(target="condition", norm=2, epochs=50, lr=1e-3)
   estimator = train_gnn_condition_estimator(train_samples, config=config)
   pred = estimator.predict(A)
   print(pred)
