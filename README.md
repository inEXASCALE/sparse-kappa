<div align="center">

# sparse-kappa

[![PyPI](https://img.shields.io/pypi/v/sparse-kappa?style=flat&color=2563eb)](https://pypi.org/project/sparse-kappa/)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.8-3776AB?logo=python&logoColor=white)](https://pypi.org/project/sparse-kappa/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://github.com/inEXASCALE/sparse-kappa/actions/workflows/test.yml/badge.svg)](https://github.com/inEXASCALE/sparse-kappa/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/sparse-kappa/badge/?version=latest)](https://sparse-kappa.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0f766e)](LICENSE)

**Condition Number Estimation on CPUs/GPUs for Sparse Matrices**

</div>

sparse-kappa estimates matrix condition numbers with numerical algorithms and optional graph neural networks using PyTorch. Condition numbers describe normwise sensitivity of linear systems to perturbations, supporting numerical analysis and mixed-precision experiments.

## Features

- **CPU and CUDA execution** through PyTorch; a GPU is optional.
- **1-norm and 2-norm estimates** with Hager/Higham, power, Lanczos, and Golub–Kahan methods.
- **Function and class APIs** with method-specific diagnostics and flexible inverse actions.
- **Reusable GNN models** for direct condition-number or inverse-norm prediction.
- **Optional matrix-element edge features** through `use_edge_features=True/False`.
- **Validated inputs and atomic model checkpoints**, with regression coverage for both edge modes.

The current numerical backend uses **dense storage internally**, including for CSR/COO inputs; several solver wrappers perform dense solves. It is not yet a memory-scalable sparse backend. Start with the small examples below and read [implementation limits](docs/source/performance.rst) before scaling up.

## Installation

Install from PyPI:

```bash
python -m pip install sparse-kappa
```

```bash
git clone https://github.com/inEXASCALE/sparse-kappa.git
cd sparse-kappa
python -m pip install -e .
```

## Quick Start

This complete example has a known answer: both norms of `diag(1, 2, 4, 10)` have condition number **10**. Numerical estimation does not require GNN training.

```python
import numpy as np
from sparse_kappa import cond_estimate

A = np.diag([1.0, 2.0, 4.0, 10.0])
print("κ₂(A):", cond_estimate(A, norm=2))
result = cond_estimate(A, norm=1, method="hager-higham", solver="lu", return_dict=True)
print("κ₁(A):", result["condition_number"])
print("Reference:", np.linalg.cond(A, p=2))
```

See the [quick start](docs/source/quickstart.rst) for complete Poisson and nonsymmetric matrix examples, and the [tutorials](docs/source/examples.rst) for reference comparisons and learned prediction.

## Available Methods

### 1-norm methods

| Method | Role |
|---|---|
| `hager-higham` / `auto` | Default inverse-norm estimate |
| `hager` | Hager iteration |
| `block-hager` / `higham` | Block Higham–Tisseur estimate |
| `power` | Power iteration |
| `oettli-prager` | Adaptive, random, or hybrid sampling |
| `monte-carlo` | Random inverse-action sampling |

Inverse actions accept `solver='auto'`, `'lu'`, `'direct'`, `'lsmr'`, `'cg'`, `'bicgstab'`, or `'gmres'`. LU caches its factorization within an estimation call. Current iterative solver wrappers use dense solves; see the [performance guide](docs/source/performance.rst).

### 2-norm methods

| Method | Role |
|---|---|
| `svds` | Estimate extremal singular values directly |
| `power` | Power/inverse iteration |
| `lanczos` | Lanczos-based estimate |
| `lanczos_unsym` | Estimate using the normal operator |
| `golub-kahan` | Bidiagonalization estimate |
| `eigsh`, `lobpcg` | Eigenvalue-based interfaces |
| `auto` | Select using matrix size and structural heuristics |

Method assumptions and implementation details are documented in the [user guide](docs/source/user_guide.rst). `return_dict=True` provides diagnostics; a convergence flag does not certify accuracy.

## GNN-Based Prediction

Choose `strategy=1` to learn `||A⁻¹||` and multiply by `||A||`, or `strategy=2` to learn the condition number directly. Strategy helpers use base-10 log targets. Supply positive, finite labels for the chosen norm and validate on held-out matrices.

The default bipartite graph already encodes matrix elements. Set **`use_edge_features=False`** for unweighted message passing, or **`True`** (the compatible default) to use numeric edge attributes. Node/global statistics retain matrix values in both modes. The mode is saved and restored with checkpoints.

```python
import torch
from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator

torch.manual_seed(7)
train_samples = [
    {"matrix": torch.diag(torch.tensor([1., 2., k], dtype=torch.float64)),
     "condition_number": k, "norm_A": k}
    for k in (2., 4., 8., 16.)
]
config = make_gnn_strategy_config(
    norm=2, strategy=2, epochs=5, device="cpu", scheduler="none",
    use_edge_features=True,  # False disables edge attributes in message passing.
)
estimator = train_gnn_strategy_estimator(train_samples, norm=2, strategy=2, config=config)
A_test = torch.diag(torch.tensor([1., 2., 6.], dtype=torch.float64))
print("Predicted κ₂:", estimator.predict(A_test))
print("Reference κ₂:", float(torch.linalg.cond(A_test)))
```

This tiny training set demonstrates usage; it does not establish predictive accuracy. The [GNN reference](docs/source/api/gnn.rst) explains features, targets, customization, validation, checkpoint compatibility, and remaining limits.

## Examples

After installing the checkout, run these complete tutorials from the repository root:

```bash
python examples/toy_models.py
python examples/api_workflows.py
python examples/gnn_edge_features.py --epochs 5
python examples/gnn_edge_features.py --epochs 5 --no-edge-features
python examples/gnn_edge_features.py --norm 1 --strategy 1 --epochs 5
```

- **Toy models:** identity, diagonal, 1D Poisson, nonsymmetric triangular, ill-conditioned, and singular matrices, with numerical references.
- **API workflows:** NumPy/PyTorch/SciPy inputs, native COO/CSR, class API, diagnostics, and method comparison.
- **GNN workflow:** labeled data, separate validation/test matrices, optional edge features, both strategies, batch prediction, and save/load consistency.

The [documentation](https://sparse-kappa.readthedocs.io/en/latest/) and [local documentation sources](docs/source/index.rst) cover these examples in detail. Local source changes appear on the hosted documentation only after publication.

## Testing

```bash
python -m pip install -e ".[dev,docs]"
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_norm2.py -v

# Run with coverage
python -m pytest tests/ --cov=sparse_kappa
```

## License

MIT License

## Contributing

See [architecture and reliability](docs/source/architecture.rst) and the [contributing guide](docs/source/contributing.rst) for module responsibilities and the validation workflow.

```bash
python -m sphinx -W --keep-going -b html docs/source /tmp/sparse-kappa-docs
```

## References

- Hager, W. W. (1984). "Condition estimates." *SIAM J. Sci. Stat. Comput.*, 5(2), 311-316.
- Higham, N. J., & Tisseur, F. (2000). "A block algorithm for matrix 1-norm estimation." *SIAM J. Matrix Anal. Appl.*, 21(4), 1185-1201.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2011). *Numerical Methods for Large Eigenvalue Problems* (2nd ed.). SIAM.
- Oettli, W., & Prager, W. (1964). "Compatibility of approximate solution of linear equations." *Numerische Mathematik*, 6(1), 405-409.
- Van der Vorst, H. A. (1992). "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems." SIAM J. Sci. Stat. Comput., 13(2), 631-644.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{carson2026estimatingconditionnumbergraph,
      title={Estimating condition number with Graph Neural Networks}, 
      author={Erin Carson and Xinye Chen},
      year={2026},
      eprint={2603.10277},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.10277}, 
}
```
