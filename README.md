<div align="center">

# sparse-kappa

[![Coverage](https://github.com/inEXASCALE/sparse-kappa/actions/workflows/test.yml/badge.svg)](https://github.com/inEXASCALE/sparse-kappa/actions/workflows/test.yml)
[![!pypi](https://img.shields.io/pypi/v/sparse-kappa?color=blue)](https://pypi.org/project/sparse-kappa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/sparse-kappa/badge/?version=latest)](https://sparse-kappa.readthedocs.io/en/latest/)


**Condition Number Estimation on CPUs/GPUs for Sparse Matrices**



</div>

Calculating the matrix condition number gives a bound on how inaccurate the solution x to the (perturbed) linear system Ax = b will be after approximation, which plays an important role in mixed-precision computing. sparse-kappa is a CPU/GPU-accelerated library for estimating condition numbers of sparse matrices using PyTorch. It supports a variety of estimation methods associated with linear solvers. sparse-kappa is designed for benchmarking condition number estimators and practical use in the science and engineering community.



## Features

- **GPU-Accelerated**: All computations run on NVIDIA GPUs via PyTorch
- **Multiple Norms**: Support for 1-norm and 2-norm condition numbers
- **Rich Algorithm Suite**:
  - **1-norm**: Hager-Higham, Power iteration, Oettli-Prager sampling, Block Hager
  - **2-norm**: Power method, Lanczos, Golub-Kahan bidiagonalization
  - **PyTorch integrations**: SVDS, EIGSH, LOBPCG wrappers
- **Flexible Solver System**: LU, LSMR, CG, GMRES, Direct, Auto-selection
- **GNN Prediction Module**: Train reusable models that predict condition numbers directly or via inverse-norm prediction
- **Smart LU Caching**: Reuses factorizations for multiple solves (10-20x speedup)
- **Memory Efficient**: Designed for large sparse matrices


## Installation


Simply via pip manager
```bash
pip install sparse-kappa
```

```bash
git clone https://github.com/chenxinye/sparse-kappa
pip install torch
pip install -e .
```

## Quick Start

```python
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate

# Create sparse matrix
A = sp.random(10000, 10000, density=0.01, format='csr')

# Estimate condition number
cond = cond_estimate(A)
print(f"κ(A) = {cond:.2e}")

# Use specific method with LU solver
cond = cond_estimate(A, norm=1, method='hager-higham', solver='lu')
```

# Available Methods

## 1-Norm Methods

| Method          | Description                              | Best For                        | Complexity     |
|-----------------|------------------------------------------|---------------------------------|----------------|
| `hager`         | Hager algorithm (default)                | High accuracy, general matrices | O(k·nnz)       |
| `power`         | Power iteration                          | Fast rough estimates            | O(k·nnz)       |
| `oettli-prager` | Random/adaptive sampling                 | Quick estimates with variants   | O(m·nnz)       |
| `hager-higham`   | Hager-Higham  (Block algorith, multiple vectors)           | Improved robustness             | O(k·b·nnz)     |

**Recommended:** Use `solver='lu'` for all 1-norm methods (10-20x faster)

## 2-Norm Methods

| Method            | Description                                                | Best For                                | Complexity      |
|-------------------|------------------------------------------------------------|-----------------------------------------|-----------------|
| `svds`            | Partial SVD (most accurate)                                | Small-medium matrices (<5k)             | O(k·nnz)        |
| `eigsh`           | Symmetric eigenvalue solver                                | Symmetric / Hermitian matrices          | O(k·nnz)        |
| `lobpcg`          | Block preconditioned CG                                    | Large matrices                          | O(k·nnz)        |
| `power`           | Power iteration                                            | Quick estimates                         | O(k·nnz)        |
| `lanczos`         | Lanczos tridiagonalization                                 | Medium symmetric matrices               | O(k²·nnz)       |
| `lanczos_unsym`   | Lanczos-style condition estimation via `eigsh` on `A^H A`  | Non-symmetric / rectangular matrices    | O(k·nnz)        |
| `golub-kahan`     | Bidiagonalization                                          | Numerically stable                      | O(k·nnz)        |
| `auto`            | Automatic selection                                        | All cases                               | -               |

## Solver Options

All 1-norm methods support flexible solver selection:

| Solver | Description | Best For | Speed | Memory |
|--------|-------------|----------|-------|--------|
| `auto` | Automatic selection (default) | General use | Good | Low |
| `lu` | LU factorization with caching | Small matrices (<5k), multiple solves | **Fastest** | High |
| `lsmr` | LSMR iterative solver | Large matrices, single solve | Medium | Low |
| `cg` | Conjugate Gradient | SPD matrices | Fast | Low |
| `bicgstab` | BiCGSTAB (stabilized BiCG) | **Non-symmetric matrices** | **Fast** | Low |
| `gmres` | GMRES | Non-symmetric, when BiCGSTAB fails | Medium | Low |
| `direct` | Direct solver (no caching) | Single solve, small matrices | Fast | Medium |

**Legend:**  
`k` = iterations, `b` = block size, `m` = samples, `nnz` = non-zeros

## GNN-Based Prediction

The `sparse_kappa.gnn` module learns a mapping from sparse matrices to
condition-number related scalars. It supports two explicit strategy workflows
for both 1-norm and 2-norm condition numbers:

- `strategy=1`: train on `log10(||A^{-1}||)` and compute `kappa(A) = ||A|| * ||A^{-1}||` at prediction time.
- `strategy=2`: train on `log10(kappa(A))` and predict the condition number directly.

The lower-level target API is still available: `target="condition"` predicts
`kappa(A)` directly, while `target="inverse_norm"` predicts `||A^{-1}||` and
multiplies by `||A||` at inference time.

The default graph builder turns a sparse matrix into a row/column bipartite
graph. You can replace the feature extractor, model, optimizer, scheduler,
loss function, and validation callback.

```python
from sparse_kappa import make_gnn_strategy_config, train_gnn_strategy_estimator
from sparse_kappa.gnn import GNNConditionEstimator
from sparse_kappa.backend import sparse as sp

train_samples = [
    {"matrix": A0, "condition_number": 12.3, "norm_A": 4.1},
    {"matrix": A1, "condition_number": 18.9, "norm_A": 5.7},
]

# Strategy 1: inverse-norm prediction. If norm_A is not supplied, sparse-kappa
# computes ||A|| for the configured norm and derives ||A^{-1}|| = kappa(A) / ||A||.
config = make_gnn_strategy_config(norm=1, strategy=1, epochs=100, lr=1e-3)
estimator = train_gnn_strategy_estimator(
    train_samples,
    norm=1,
    strategy=1,
    val_data=None,
    config=config,
    save_path="models/gnn_strategy1_norm1.pt",
)

# Load and predict one matrix or a list of matrices.
estimator = GNNConditionEstimator.load("models/gnn_strategy1_norm1.pt")
result = estimator.predict(sp.random(100, 100, density=0.02, format="csr"), return_dict=True)
print(result["condition_number"], result["norm_A"], result["norm_Ainv"])

# Strategy 2: direct condition-number prediction.
direct_config = make_gnn_strategy_config(norm=2, strategy=2, epochs=100)
direct_estimator = train_gnn_strategy_estimator(train_samples, norm=2, strategy=2, config=direct_config)
pred = direct_estimator.predict(A_test)
```

Customization hooks follow the same shape:

```python
import torch

estimator.fit(
    train_samples,
    val_data=val_samples,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=5e-4),
    scheduler_factory=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50),
    loss_fn=torch.nn.SmoothL1Loss(),
    validator=my_validation_callback,
)
```

## Examples

### Example 1: Compare Methods

```python
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate

A = sp.random(2000, 2000, density=0.005, format='csr')

# Compare 1-norm methods
methods_1 = ['hager-higham', 'power', 'oettli-prager', 'block-hager']
for method in methods_1:
    result = cond_estimate(A, norm=1, method=method, solver='lu', 
                          return_dict=True)
    print(f"{method:15s}: κ={result['condition_number']:.4e}, "
          f"iters={result['iterations']}")

# Compare 2-norm methods
methods_2 = ['svds', 'lanczos', 'golub-kahan']
for method in methods_2:
    cond = cond_estimate(A, norm=2, method=method)
    print(f"{method:12s}: κ={cond:.4e}")
```

### Example 2: LU Solver (for 1-norm)

```python
# Highly recommended: use LU solver for Hager-Higham
result = cond_estimate(A, norm=1, method='hager-higham',
                      solver='lu', return_dict=True)

print(f"Condition number: {result['condition_number']:.4e}")
print(f"Solver info:")
print(f"  Type: {result['solver_info']['solver_A']['method']}")
print(f"  Factorized: {result['solver_info']['solver_A']['factorized']}")
print(f"  Solves: {result['solver_info']['solver_A']['solve_count']}")
```

### Example 3: Oettli-Prager Variants

```python
# Adaptive (most accurate)
result = cond_estimate(A, norm=1, method='oettli-prager',
                      solver='lu', variant='adaptive', max_iter=15)

# Random sampling (fastest)
result = cond_estimate(A, norm=1, method='oettli-prager',
                      solver='lu', variant='random', max_iter=20)

# Hybrid (balanced)
result = cond_estimate(A, norm=1, method='oettli-prager',
                      solver='lu', variant='hybrid', max_iter=15)
```

### Example 4: Custom Solver Parameters
```python
# LSMR with relaxed tolerance for large matrices
result = cond_estimate(A, norm=1, method='hager-higham',
                      solver='lsmr',
                      solver_kwargs={'atol': 1e-3, 'maxiter': 20})

# CG for symmetric matrices
A_spd = A @ A.T + sp.eye(A.shape[0]) * 10
result = cond_estimate(A_spd, norm=1, method='hager-higham',
                      solver='cg',
                      solver_kwargs={'atol': 1e-3, 'maxiter': 30})
```

## Performance Tips

1. **Auto mode is recommended** for first-time usage
2. **For symmetric matrices**, use `eigsh` or `lanczos`
3. **For large sparse matrices** (>10k), use `golub-kahan` or `lobpcg`
4. **For highest accuracy on small matrices**, use `svds`
5. **Increase `max_iter`** if convergence fails

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_norm2.py -v

# Run with coverage
pytest tests/ --cov=sparse_kappa
```


## License

MIT License

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

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
