"""Dense/sparse inputs, diagnostics, method comparison, and reusable API.

Run: python examples/api_workflows.py
"""

import numpy as np
import torch

from sparse_kappa import ConditionNumberEstimator, cond_estimate
from sparse_kappa.backend import sparse as sp


def main():
    torch.manual_seed(7)
    dense = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
    tensor = torch.as_tensor(dense)
    native_coo = tensor.to_sparse_coo()
    native_csr = tensor.to_sparse_csr()
    inputs = {
        "numpy": dense,
        "torch dense": tensor,
        "torch COO": native_coo,
        "torch CSR": native_csr,
        "backend matrix": sp.csr_matrix(tensor),
    }
    for name, A in inputs.items():
        print(name, cond_estimate(A, norm=2))
    # SciPy input is optional. It is converted to the current backend storage.
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        pass
    else:
        print("SciPy CSR", cond_estimate(csr_matrix(dense), norm=2))

    estimator = ConditionNumberEstimator(
        dense, norm=1, method="hager-higham", solver="lu"
    )
    result = estimator.estimate()
    print("Reusable API:", result["condition_number"], result["method"])
    for norm, methods in [
        (1, ["hager-higham", "hager", "block-hager"]),
        (2, ["svds", "power", "lanczos", "golub-kahan"]),
    ]:
        reference = np.linalg.cond(dense, p=norm)
        for method in methods:
            options = {"solver": "lu"} if norm == 1 else {}
            result = cond_estimate(
                dense,
                norm=norm,
                method=method,
                max_iter=100,
                tol=1e-8,
                return_dict=True,
                **options,
            )
            print(
                f"norm={norm} {method:14s} estimate={result['condition_number']:.6g} "
                f"reference={reference:.6g} converged={result.get('converged', 'unreported')}"
            )


if __name__ == "__main__":
    main()
