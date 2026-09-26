Troubleshooting
===============

Import errors
-------------

**Problem:** missing runtime dependencies such as ``torch``.

**Fix:** install project dependencies first.

.. code-block:: bash

   pip install -e .

Convergence issues
------------------

**Problem:** iterative method does not converge or returns unstable values.

**Fixes:**

* Increase ``max_iter`` and tighten/relax ``tol`` as needed.
* Switch method (for example from ``power`` to ``svds`` or ``golub-kahan``).
* For 1-norm estimation, prefer ``solver='lu'`` when memory allows.

Very large condition numbers
----------------------------

A very large output can be expected for nearly singular matrices.
Validate with another method and inspect matrix rank/structure before treating it as an error.

GPU / device mismatch
---------------------

If tensors or sparse matrices are on different devices, move data to a consistent device before estimation.

Invalid data or configuration
-------------------------------

``ValueError`` identifies empty/non-square/non-finite numerical inputs, invalid
iteration controls, and invalid GNN labels/configuration. GNN labels must be
positive and finite; infinity is not a valid label for a singular matrix.
Complex GNN inputs are rejected explicitly. Normalize/rescale unusually large
values or select float64 features if feature statistics overflow.

Non-finite GNN loss or prediction
----------------------------------

``FloatingPointError`` means the training loss, validation metric, or predicted
quantity is non-finite (or a predicted quantity is nonpositive). Inspect the
matrix values, labels, learning rate, and custom model/loss. Do not use the
failed prediction as a condition estimate. A custom validation callback must
return a finite scalar where smaller is better.

Checkpoint behavior
---------------------

The built-in extractor and edge mode are restored automatically. If you used a
custom model/extractor, pass it explicitly to ``GNNConditionEstimator.load``.
Checkpoint loading uses restricted tensor/configuration deserialization; avoid
putting arbitrary Python objects into custom model configuration dictionaries.
Checkpoints do not preserve optimizer state for exact training continuation.

Documentation import or build issues
--------------------------------------

Use the same Python interpreter for editable installation, examples, tests,
and Sphinx. Install the ``docs`` extra for documentation dependencies. Build
from the repository root with the maintained ``docs/source`` tree; archived
``docs/build`` outputs are not the editable source.
