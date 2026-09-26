Changelog
=========

Unreleased
----------

* Added optional matrix-element edge features via ``use_edge_features`` while
  preserving existing weighted defaults and checkpoint compatibility.
* Added sparse graph extraction for native PyTorch and SciPy inputs, explicit
  data/configuration validation, atomic checkpoints, and extractor persistence.
* Corrected numerical input conversion and native CSR tensor handling.
* Estimate extremal singular values directly in the ``svds`` condition path.
* Added self-contained toy-model, API, and GNN tutorials with reference values.
* Clarified dense backend limitations and consolidated packaging metadata.


0.0.2
-----

* Added GNN estimator training/prediction interfaces.
* Improved sparse solver and method coverage for condition number estimation.

0.0.1
-----

* Initial public release with sparse condition number estimation APIs.
