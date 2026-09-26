"""
Core API for condition number estimation.
"""

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from typing import Union, Dict, Any
import warnings
import math

from .norm1.hager_higham import hager_higham_norm1, block_higham_tisseur_norm1
from .norm1.power_iteration import power_iteration_norm1
from .norm1.oettli_prager import oettli_prager_norm1
from .norm1.monte_carlo import monte_carlo_norm1
from .norm1.hager import hager_norm1_cond
from .norm2.power_method import power_method_cond
from .norm2.lanczos import lanczos_cond
from .norm2.lanczos_unsym import lanczos_unsym_cond
from .norm2.svds import svds_cond
from .norm2.golub_kahan import golub_kahan_cond
from .norm2.torch_wrappers import eigsh_cond, lobpcg_cond
from .utils import validate_matrix, get_matrix_properties



class ConditionNumberEstimator:
    """
    Main class for condition number estimation on GPU.
    
    Supports both 1-norm and 2-norm condition number estimation
    with multiple algorithms.
    """
    
    METHODS_1NORM = {
        'hager-higham', 'hager', 'higham',
        'power', 'power-iteration',
        'oettli-prager', 'oettli', 'prager',
        'monte-carlo', 'sampling', 'random',
        'block-hager', 'block'
    }
    METHODS_2NORM = {
        'power', 'lanczos', 'lanczos_unsym', 'golub-kahan', 
        'svds', 'eigsh', 'lobpcg', 'auto'
    }
    
    def __init__(
        self,
        A: Union[sp.spmatrix, cp.ndarray],
        norm: int = 2,
        method: str = 'auto',
        **kwargs
    ):
        """
        Initialize condition number estimator.
        
        Parameters
        ----------
        A : sparse matrix or dense array
            Input matrix (must be square)
        norm : int, default=2
            Which norm to use (1 or 2)
        method : str, default='auto'
            Estimation method to use
        **kwargs : dict
            Additional parameters for the specific method
        """
        self.A = validate_matrix(A)
        self.norm = norm
        self.method = method
        self.kwargs = kwargs
        self.properties = get_matrix_properties(self.A)
        
        if self.norm not in [1, 2]:
            raise ValueError(f"Unsupported norm: {norm}. Use 1 or 2.")
        
        if self.method == 'auto':
            self.method = self._select_method()
        
        self._validate_method()
    
    def _select_method(self) -> str:
        """Auto-select best method based on matrix properties."""
        if self.norm == 1:
            return 'hager-higham'
        
        # For 2-norm, select based on matrix properties
        n = self.properties['size']
        is_symmetric = self.properties['is_symmetric']
        nnz = self.properties['nnz']
        density = self.properties['density']
        
        # Small matrices: use svds (most accurate)
        if n < 5000:
            return 'svds'
        
        # Symmetric matrices: use eigsh
        if is_symmetric:
            return 'eigsh'
        
        # Large sparse matrices: use Lanczos or Golub-Kahan
        if density < 0.01:
            return 'golub-kahan'
        
        # Default: Lanczos
        return 'lanczos'
    
    def _validate_method(self):
        """Validate method selection."""
        if self.norm == 1 and self.method not in self.METHODS_1NORM:
            raise ValueError(
                f"Invalid method '{self.method}' for 1-norm. "
                f"Available: {self.METHODS_1NORM}"
            )
        
        if self.norm == 2 and self.method not in self.METHODS_2NORM:
            raise ValueError(
                f"Invalid method '{self.method}' for 2-norm. "
                f"Available: {self.METHODS_2NORM}"
            )
        
        # Special checks
        if self.method == 'eigsh':
            if not self.properties['is_symmetric']:
                if self.kwargs.get('verbose', False):
                    print(
                        "Note: Method 'eigsh' works best for symmetric matrices. "
                        "Using it on non-symmetric matrix may give approximate results."
                    )
            self.method = 'lanczos_unsym'
    
    def estimate(self) -> Dict[str, Any]:
        """
        Compute condition number estimation.
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'condition_number': estimated condition number
            - 'iterations': number of iterations
            - 'converged': convergence status
            - 'method': method used
            - Additional method-specific info
        """
        if self.norm == 1:
            return self._estimate_norm1()
        else:
            return self._estimate_norm2()
    
    def _estimate_norm1(self) -> Dict[str, Any]:
        """Estimate 1-norm condition number."""
        method_map = {
            'hager-higham': hager_higham_norm1,
            'hager': hager_norm1_cond,
            'higham': block_higham_tisseur_norm1,
            'power': power_iteration_norm1,
            'power-iteration': power_iteration_norm1,
            'oettli-prager': oettli_prager_norm1,
            'oettli': oettli_prager_norm1,
            'prager': oettli_prager_norm1,
            'monte-carlo': monte_carlo_norm1,
            'sampling': monte_carlo_norm1,
            'random': monte_carlo_norm1,
            'block-hager': block_higham_tisseur_norm1,
            'block': block_higham_tisseur_norm1,
        }
        
        estimator = method_map[self.method]
        result = estimator(self.A, **self.kwargs)
        result['norm'] = 1
        result['method'] = self.method
        return result
    
    def _estimate_norm2(self) -> Dict[str, Any]:
        """Estimate 2-norm condition number."""
        method_map = {
            'power': power_method_cond,
            'lanczos': lanczos_cond,
            'lanczos_unsym': lanczos_unsym_cond,
            'golub-kahan': golub_kahan_cond,
            'svds': svds_cond,
            'eigsh': eigsh_cond,
            'lobpcg': lobpcg_cond,
        }
        
        estimator = method_map[self.method]
        result = estimator(self.A, **self.kwargs)
        result['norm'] = 2
        result['method'] = self.method
        return result


def cond_estimate(
    A: Union[sp.spmatrix, cp.ndarray],
    norm: int = 2,
    method: str = 'auto',
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    return_dict: bool = False,  # New parameter
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Estimate condition number of a sparse matrix on GPU.
    
    Parameters
    ----------
    A : sparse matrix or dense array
        Input square matrix
    norm : int, default=2
        Norm to use (1 or 2)
    method : str, default='auto'
        Estimation method:
        - For 1-norm: 'hager-higham', 'power', 'oettli-prager', 'block-hager'
        - For 2-norm: 'power', 'lanczos', 'lanczos_unsym', 'golub-kahan',
                      'svds', 'eigsh', 'lobpcg', 'auto'
    
    max_iter : int, default=100
        Maximum iterations for iterative methods
    
    tol : float, default=1e-6
        Convergence tolerance
    
    verbose : bool, default=False
        Print iteration information
    
    return_dict : bool, default=False
        If True, always return dict with detailed information.
        If False, return float when verbose=False, dict when verbose=True.
    
    **kwargs : dict
        Additional method-specific parameters
    
    Returns
    -------
    condition_number : float or dict
        If verbose=False, returns condition number as float.
        If verbose=True, returns dict with detailed information.
    
    Examples
    --------
    >>> from sparse_kappa.backend import torch_api as cp
    >>> from sparse_kappa.backend import sparse as sp
    >>> from sparse_kappa import cond_estimate
    >>> 
    >>> # 2-norm condition number
    >>> A = sp.random(1000, 1000, density=0.01, format='csr')
    >>> cond = cond_estimate(A, norm=2, method='lanczos')
    >>> 
    >>> # 1-norm condition number
    >>> cond = cond_estimate(A, norm=1, method='hager-higham')
    >>> 
    >>> # Compare methods
    >>> for method in ['hager-higham', 'power', 'block-hager']:
    >>>     result = cond_estimate(A, norm=1, method=method, verbose=True)
    >>>     print(f"{method}: {result['condition_number']:.4e}")
    """
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    kwargs.update({'max_iter': max_iter, 'tol': tol, 'verbose': verbose})
    
    estimator = ConditionNumberEstimator(A, norm=norm, method=method, **kwargs)
    result = estimator.estimate()
    
    if verbose or return_dict:
        return result
    else:
        return result['condition_number']