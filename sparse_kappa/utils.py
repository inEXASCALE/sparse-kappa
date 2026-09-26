"""
Utility functions for condition number estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.backend.sparse import linalg as splinalg
from typing import Union, Dict, Any
import warnings
import torch


def validate_matrix(A: Union[sp.spmatrix, cp.ndarray]) -> sp.spmatrix:
    """
    Validate and convert input matrix to CSR format.
    
    Parameters
    ----------
    A : sparse matrix or dense array
        Input matrix
    
    Returns
    -------
    A_csr : sparse matrix in CSR format
    """
    # Convert dense to sparse if needed
    if isinstance(A, cp.ndarray):
        if A.ndim != 2:
            raise ValueError("Input must be a 2D array")
        A = sp.csr_matrix(A)
    
    # Convert to CSR format for efficiency
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    
    if A.shape[0] == 0:
        raise ValueError("Matrix must be nonempty")
    if not torch.isfinite(A.data).all():
        raise ValueError("Matrix entries must be finite")
    if not (A.dtype.is_floating_point or A.dtype.is_complex):
        A = A.astype(cp.float64)
    return A


def get_matrix_properties(A: sp.spmatrix) -> Dict[str, Any]:
    """
    Extract properties of sparse matrix.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    
    Returns
    -------
    properties : dict
        Dictionary with matrix properties
    """
    n = A.shape[0]
    nnz = A.nnz
    density = nnz / (n * n)
    
    # Check if symmetric (approximately)
    is_symmetric = is_matrix_symmetric(A)
    
    # Check if Hermitian
    is_hermitian = is_matrix_hermitian(A)
    
    return {
        'size': n,
        'nnz': nnz,
        'density': density,
        'is_symmetric': is_symmetric,
        'is_hermitian': is_hermitian,
        'dtype': A.dtype,
    }


def is_matrix_symmetric(A: sp.spmatrix, tol: float = 1e-10) -> bool:
    """
    Check if matrix is symmetric.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    tol : float
        Tolerance for symmetry check
    
    Returns
    -------
    is_sym : bool
        True if matrix is symmetric
    """
    if A.shape[0] != A.shape[1]:
        return False
    
    if A.nnz != A.T.nnz:
        return False
    
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    
    n = A.shape[0]
    if n > 1000:
        num_samples = min(100, n)
        indices = cp.random.choice(n, size=num_samples, replace=False)
        
        for i in indices:
            for j in indices:
                if i != j:
                    val_ij = A[i, j]
                    val_ji = A[j, i]
                    if abs(val_ij - val_ji) > tol * (abs(val_ij) + abs(val_ji) + 1e-14):
                        return False
        return True
    else:
        diff = A - A.T
        norm_diff = splinalg.norm(diff)
        norm_A = splinalg.norm(A)
        
        return norm_diff <= tol * norm_A


def is_matrix_hermitian(A: sp.spmatrix, tol: float = 1e-10) -> bool:
    """
    Check if matrix is Hermitian.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    tol : float
        Tolerance for Hermitian check
    
    Returns
    -------
    is_herm : bool
        True if matrix is Hermitian
    """
    if A.shape[0] != A.shape[1]:
        return False
    
    if not cp.iscomplexobj(A.data):
        return is_matrix_symmetric(A, tol)
    
    # Check if ||A - A^H|| is small
    diff = A - A.conj().T
    norm_diff = splinalg.norm(diff)
    norm_A = splinalg.norm(A)
    
    return norm_diff <= tol * norm_A


def check_convergence(
    values: cp.ndarray,
    tol: float = 1e-6,
    window: int = 5
) -> bool:
    """
    Check convergence of iterative method.
    
    Parameters
    ----------
    values : array
        History of values (e.g., condition number estimates)
    tol : float
        Convergence tolerance
    window : int
        Number of recent iterations to check
    
    Returns
    -------
    converged : bool
        True if converged
    """
    if len(values) < window + 1:
        return False
    
    recent = values[-window:]
    relative_change = cp.abs(recent[-1] - recent[0]) / (cp.abs(recent[-1]) + 1e-14)
    
    return float(relative_change) < tol


def sparse_matrix_norm(A: sp.spmatrix, ord: int = 2) -> float:
    """
    Compute matrix norm for sparse matrix.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    ord : int
        Norm order (1 or 2)
    
    Returns
    -------
    norm : float
        Matrix norm
    """
    if ord == 1:
        # 1-norm: max column sum
        # For sparse matrices, we need to compute absolute values carefully
        if sp.isspmatrix_csr(A):
            # Convert to CSC for efficient column operations
            A_csc = A.tocsc()
            # Sum absolute values of each column
            col_sums = cp.array([
                float(cp.sum(cp.abs(A_csc[:, j].data)))
                for j in range(A_csc.shape[1])
            ])
        else:
            # For other formats, convert to dense temporarily for small matrices
            # or use element-wise operations
            A_abs = A.copy()
            A_abs.data = cp.abs(A_abs.data)
            col_sums = cp.array(A_abs.sum(axis=0)).ravel()
        
        return float(cp.max(col_sums))
    
    elif ord == 2:
        # 2-norm: use sparse linalg norm
        return float(splinalg.norm(A))
    else:
        raise ValueError(f"Unsupported norm: {ord}")


def apply_inverse(
    A: sp.spmatrix,
    b: cp.ndarray,
    method: str = 'lsmr',
    **kwargs
) -> cp.ndarray:
    """
    Apply inverse of matrix A to vector b: x = A^{-1} b.
    
    Uses iterative solvers from PyTorch.
    
    Parameters
    ----------
    A : sparse matrix
        Input matrix
    b : array
        Right-hand side vector
    method : str
        Solver method ('lsmr', 'lsqr', 'cg', 'gmres')
    **kwargs : dict
        Additional solver parameters
    
    Returns
    -------
    x : array
        Solution vector
    """
    if method == 'lsmr':
        result = splinalg.lsmr(A, b, **kwargs)
        return result[0]
    elif method == 'lsqr':
        result = splinalg.lsqr(A, b, **kwargs)
        return result[0]
    elif method == 'cg':
        x, info = splinalg.cg(A, b, **kwargs)
        if info != 0:
            warnings.warn(f"CG solver did not converge: info={info}")
        return x
    elif method == 'gmres':
        x, info = splinalg.gmres(A, b, **kwargs)
        if info != 0:
            warnings.warn(f"GMRES solver did not converge: info={info}")
        return x
    else:
        raise ValueError(f"Unknown solver method: {method}")


def print_iteration(
    iteration: int,
    value: float,
    residual: float = None,
    extra: str = ""
):
    """
    Print iteration information.
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    value : float
        Current estimate
    residual : float, optional
        Residual norm
    extra : str, optional
        Extra information to print
    """
    msg = f"Iter {iteration:4d}: value = {value:.6e}"
    if residual is not None:
        msg += f", residual = {residual:.6e}"
    if extra:
        msg += f", {extra}"
    print(msg)