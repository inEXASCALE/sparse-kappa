GNN API and edge feature options
================================

``GNNConditionEstimator`` learns matrix-to-scalar predictions. It needs labeled
training data and predicts only within the accuracy established by validation.
For numerical estimation without training, use :doc:`main`.

Choosing edge features
----------------------

Set ``TrainingConfig(use_edge_features=True)`` to use matrix-element attributes
in message passing, or ``False`` for unweighted message passing. The default
is **True**, preserving earlier checkpoints and the existing implementation.
The same option is accepted by ``make_gnn_strategy_config``.

.. code-block:: python

   import torch
   from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator

   torch.manual_seed(7)
   samples = [
       {"matrix": torch.diag(torch.tensor([1., 2., k], dtype=torch.float64)),
        "condition_number": k, "norm_A": k}
       for k in (2., 4., 8., 16.)
   ]
   config = make_gnn_strategy_config(
       norm=2, strategy=2, epochs=5, scheduler="none", device="cpu",
       use_edge_features=False,  # Set True to encode matrix elements on edges.
   )
   estimator = train_gnn_strategy_estimator(samples, norm=2, strategy=2, config=config)
   test = torch.diag(torch.tensor([1., 2., 6.], dtype=torch.float64))
   print(estimator.predict(test))
   print(estimator.predict(test, return_dict=True))

Graph representation
~~~~~~~~~~~~~~~~~~~~

An m-by-n real matrix produces m+n nodes: row nodes followed by column nodes.
Each nonzero A[i,j] creates row-to-column and column-to-row edges. Duplicate
sparse entries are summed; explicit zeros and cancellations create no edges.
Native PyTorch COO/CSR and SciPy sparse inputs are extracted without converting
the entire matrix to dense storage. Dense/backend input still uses its existing
storage. The extractor rejects complex, non-finite, or empty inputs.

Let s=max(max(abs(A)), eps). The edge tensor has four channels:

.. list-table::
   :header-rows: 1

   * - Channel
     - Meaning
   * - 0
     - A[i,j] / s
   * - 1
     - abs(A[i,j]) / s
   * - 2
     - sign(A[i,j])
   * - 3
     - Direction: 1 for row-to-column, 0 for column-to-row

With the option disabled, the extractor masks the first three channels and the
default model ignores the entire edge tensor. Graph connectivity is unchanged.
Node/global statistics still contain matrix values; this is an ablation of edge
attributes, not an entirely topology-only estimator.

Each node has seven features: row indicator, column indicator, normalized
nonzero count, log absolute row/column sum divided by log1p(s), row/column L2
norm divided by s, normalized index, and normalized absolute diagonal value.
The eight global features are normalized row/column counts, density,
log1p(mean(abs(nonzeros))), log1p(s), Frobenius norm divided by s, symmetry
score, and normalized diagonal length. Node indices encode ordering, so this
extractor does not promise invariance to arbitrary row/column permutations.

Strategy and label contracts
----------------------------

The norm is ``1`` or ``2`` and must match the labels supplied by the user.

* ``strategy=1`` predicts the inverse norm and computes
  kappa(A)=norm(A)*norm(A_inverse). Labels may be ``norm_Ainv`` or
  ``condition_number``; the latter is divided by ``norm_A``.
* ``strategy=2`` predicts the condition number directly.
* The strategy helpers default to base-10 log targets. Plain ``TrainingConfig``
  defaults to natural logs and ``target="condition"``; its lower-level
  ``target="inverse_norm"`` option remains supported.

Data may be dictionaries with ``matrix`` and ``condition_number`` (or
``norm_Ainv``), tuples ``(matrix, condition_number)``, or
``MatrixConditionDataset(matrices, labels=labels)``. A dataset also supports
custom matrix/label key names. Labels must be positive and finite. Empty
training/validation/evaluation datasets raise ``ValueError``. Training checks
all labels before performing optimizer steps and fails explicitly on non-finite
losses/metrics rather than saving them as successful results.

For inverse-norm targets, a precomputed ``norm_A`` avoids recomputation during
training. Otherwise the matrix norm is computed, and the 2-norm currently uses
a dense SVD. Direct scalar ``predict(A)`` skips matrix norm calculation;
``predict(A, return_dict=True)`` computes it to return all components.

Fit, predict, evaluate
----------------------

* ``fit(train_data, val_data=None, ..., save_path=None)`` trains on one matrix
  graph at a time, records losses, and restores the best model by validation
  loss (or training loss if no validation data is provided).
* ``evaluate(data)`` returns mean loss in the configured target space, not
  relative error of the condition number.
* ``predict(A)`` returns a float. ``predict([A, B])`` returns a list.
* ``return_dict=True`` returns ``condition_number``, ``norm_A``, ``norm_Ainv``,
  ``norm``, ``target``, ``strategy``, and ``predicted_quantity`` for each matrix.
  In direct mode, ``norm_Ainv`` is derived from the predicted condition number.

The default optimizer is AdamW; the default loss is MSE in target space.
Schedulers are ``plateau``, ``cosine``, or ``none``. Use ``grad_clip`` and
``early_stopping_patience`` to control training, and set ``device="cpu"`` or
``device="cuda"`` explicitly when needed. If omitted, CUDA is selected when
available. A fixed seed is useful for comparisons but does not guarantee bitwise
reproducibility across platforms or GPU kernels.

Saving and loading
------------------

.. code-block:: python

   import tempfile
   from pathlib import Path
   import torch
   from sparse_kappa import TrainingConfig, train_gnn_condition_estimator
   from sparse_kappa.gnn import GNNConditionEstimator

   samples = [(torch.diag(torch.tensor([1., k])), k) for k in (2., 4., 8.)]
   with tempfile.TemporaryDirectory() as directory:
       path = Path(directory) / "model.pt"
       estimator = train_gnn_condition_estimator(
           samples, config=TrainingConfig(epochs=2, scheduler="none", device="cpu",
                                          use_edge_features=False), save_path=path,
       )
       loaded = GNNConditionEstimator.load(path)
       print(loaded.config.use_edge_features)  # False, restored automatically.
       print(loaded.predict(torch.diag(torch.tensor([1., 6.]))))

Checkpoints contain model weights, model/training configuration, feature
extractor settings, and loss history. They are written atomically and loaded
with ``weights_only=True``. They do not contain optimizer/scheduler states and
are inference checkpoints, not exact training-resumption snapshots. Older
checkpoints without the new option load in weighted mode. ``map_location``
controls checkpoint tensor storage; inference uses the loaded config's device.

Customization
-------------

Pass a ``feature_extractor`` with ``node_feature_dim``, ``edge_feature_dim``,
``global_feature_dim`` and a callable returning ``MatrixGraph``; or pass a
PyTorch model returning one scalar per graph. Custom components control their
own edge handling; the switch configures the bundled components. Supply the
same custom component explicitly to ``load``. Built-in extractor dtype, eps,
and edge mode are restored automatically.

``fit`` also accepts ``optimizer_factory``, ``scheduler_factory``, ``loss_fn``,
and ``validator(estimator, validation_dataset)``. The validator must return a
finite scalar where smaller is better. A complete training/validation/save/load
example is included in :doc:`../examples`.
