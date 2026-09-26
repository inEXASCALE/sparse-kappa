import warnings
from typing import Dict, Any

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.backend.sparse import linalg as splinalg


def svds_cond(
    A: sp.spmatrix,
    max_iter: int = None,
    tol: float = 0.0,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimate 2-norm condition number for a sparse matrix.

    Strategy
    --------
    - sigma_max: via sparse partial SVD (`svds`, largest singular value)
    - sigma_min: via partial SVD (`svds`, smallest singular value)

    Notes
    -----
    This is a sparse-oriented method, but dense SVD may be used as a
    fallback for very small matrices.
    """
    m, n = A.shape
    min_dim = min(m, n)

    if verbose:
        print("Condition estimation via extremal singular values")
        print(f"Matrix size: {A.shape}")

    if min_dim == 0:
        return {
            "condition_number": float("inf"),
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "singular_values": [],
            "iterations": 0,
            "converged": False,
        }

    # Very small matrix: use dense SVD directly
    if min_dim <= 2:
        s = cp.linalg.svd(A.toarray(), compute_uv=False)
        sigma_max = float(s[0])
        sigma_min = float(s[-1])
        cond = float("inf") if sigma_min == 0.0 else float(sigma_max / sigma_min)
        return {
            "condition_number": cond,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "singular_values": s.tolist(),
            "iterations": 0,
            "converged": True,
        }

    try:
        # Largest singular value only
        s_large = splinalg.svds(
            A,
            k=1,
            which="LM",
            return_singular_vectors=False,
            maxiter=max_iter,
            tol=tol,
        )
        sigma_max = float(cp.abs(s_large[0]))

        # Estimate sigma_min directly: forming A^H A squares the condition
        # number and can lose the smallest singular value to roundoff.
        s_small = splinalg.svds(
            A, k=1, which="SM", return_singular_vectors=False,
            maxiter=max_iter, tol=tol,
        )
        sigma_min = float(cp.abs(s_small[0]))

        cond = float("inf") if sigma_min == 0.0 else float(sigma_max / sigma_min)

        if sigma_min < 1e-14:
            warnings.warn(
                f"Very small sigma_min={sigma_min:.3e}; condition number may be unstable."
            )

        all_sigma = cp.array([sigma_max, sigma_min])
        all_sigma = cp.flip(cp.sort(all_sigma), 0)

        converged = True

    except Exception as e:
        if verbose:
            print(f"Sparse path failed: {e}")
            print("Falling back to dense SVD")

        s = cp.linalg.svd(A.toarray(), compute_uv=False)
        sigma_max = float(s[0])
        sigma_min = float(s[-1])
        cond = float("inf") if sigma_min == 0.0 else float(sigma_max / sigma_min)
        all_sigma = s
        converged = False

    if verbose:
        print(f"sigma_max = {sigma_max:.6e}")
        print(f"sigma_min = {sigma_min:.6e}")
        print(f"kappa_2(A) = {cond:.6e}")

    return {
        "condition_number": float(cond),
        "sigma_max": float(sigma_max),
        "sigma_min": float(sigma_min),
        "singular_values": all_sigma.tolist(),
        "iterations": max_iter or 0,
        "converged": converged,
    }