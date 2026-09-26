"""Compare estimates with small, independently computed reference values.

Run from the repository root: python examples/toy_models.py
No dataset, SciPy installation, or GPU is required.
"""

import math

import torch

from sparse_kappa import cond_estimate
from sparse_kappa.backend import sparse as sp


def toy_matrices():
    identity = torch.eye(4, dtype=torch.float64)
    diagonal = torch.diag(torch.tensor([1.0, 2.0, 4.0, 10.0], dtype=torch.float64))
    n = 8
    poisson = 2 * torch.eye(n, dtype=torch.float64)
    poisson += torch.diag(-torch.ones(n - 1, dtype=torch.float64), diagonal=1)
    poisson += torch.diag(-torch.ones(n - 1, dtype=torch.float64), diagonal=-1)
    triangular = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]], dtype=torch.float64
    )
    ill_conditioned = torch.diag(torch.tensor([1.0, 1e-4, 1e-8], dtype=torch.float64))
    return {
        "identity": identity,
        "diagonal": diagonal,
        "1D Poisson": poisson,
        "nonsymmetric triangular": triangular,
        "ill-conditioned diagonal": ill_conditioned,
    }


def main():
    torch.manual_seed(7)
    for name, dense in toy_matrices().items():
        A = sp.csr_matrix(dense)
        for norm in (1, 2):
            options = {"solver": "lu"} if norm == 1 else {}
            result = cond_estimate(A, norm=norm, return_dict=True, **options)
            reference = float(torch.linalg.cond(dense, p=norm))
            error = abs(result["condition_number"] / reference - 1)
            print(
                f"{name:26s} norm={norm} estimate={result['condition_number']:.6g} "
                f"reference={reference:.6g} relative_error={error:.2e}"
            )
    # Closed-form eigenvalues of the 1D Dirichlet Poisson operator.
    theta = math.pi / 9
    print(
        f"Poisson kappa_2 analytic: {(1 + math.cos(theta)) / (1 - math.cos(theta)):.6g}"
    )
    # Exact zero singular values give an infinite 2-norm condition number.
    singular = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    print("Singular kappa_2:", cond_estimate(singular, norm=2))


if __name__ == "__main__":
    main()
