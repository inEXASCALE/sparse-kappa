"""Sparse matrix graph construction and default feature extraction."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from sparse_kappa.backend import sparse as sp

from .data import MatrixGraph


def matrix_to_dense_tensor(
    matrix: Any, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert common dense/sparse matrix inputs to a dense torch tensor."""
    if sp.isspmatrix(matrix):
        tensor = matrix.toarray()
    elif isinstance(matrix, torch.Tensor):
        tensor = matrix.to_dense() if matrix.layout != torch.strided else matrix
    elif hasattr(matrix, "toarray"):
        tensor = torch.as_tensor(matrix.toarray())
    else:
        tensor = torch.as_tensor(np.asarray(matrix))

    if tensor.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if min(tensor.shape) == 0:
        raise ValueError("matrix dimensions must be nonzero")
    if tensor.is_complex():
        raise ValueError("GNN features currently support real-valued matrices only")
    tensor = tensor.to(dtype=dtype)
    if not torch.isfinite(tensor).all():
        raise ValueError(
            "matrix entries must be finite and representable in the feature dtype"
        )
    return tensor


class DefaultGraphFeatureExtractor:
    """
    Build a row/column bipartite graph from a sparse matrix.

    Row nodes occupy ``[0, m)`` and column nodes occupy ``[m, m+n)``. Each
    nonzero value creates two directed edges so the default model can pass
    messages in both directions. ``use_edge_features=False`` masks the numeric
    edge channels; node/global statistics still include matrix values.
    Native PyTorch and SciPy sparse inputs are processed without densification.
    """

    node_feature_dim = 7
    edge_feature_dim = 4
    global_feature_dim = 8

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
        use_edge_features: bool = True,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("feature dtype must be torch.float32 or torch.float64")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        if not isinstance(use_edge_features, bool):
            raise ValueError("use_edge_features must be a bool")
        self.dtype = dtype
        self.eps = eps
        self.use_edge_features = use_edge_features

    def config(self) -> dict:
        return {
            "dtype": str(self.dtype).split(".")[-1],
            "eps": self.eps,
            "use_edge_features": self.use_edge_features,
        }

    def __call__(
        self,
        matrix: Any,
        target: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MatrixGraph:
        indices, values, (m, n) = self._entries(matrix)
        device = values.device
        rows, cols = indices
        abs_values = values.abs()
        scale = (
            abs_values.max().clamp_min(self.eps)
            if values.numel()
            else values.new_tensor(self.eps)
        )
        row_abs = values.new_zeros(m).index_add_(0, rows, abs_values)
        col_abs = values.new_zeros(n).index_add_(0, cols, abs_values)
        # Normalize before squaring to avoid overflow for large finite values.
        squares = (values / scale).square()
        floor = self.eps / scale.square()
        row_sq = (
            values.new_zeros(m).index_add_(0, rows, squares).clamp_min(floor).sqrt()
        )
        col_sq = (
            values.new_zeros(n).index_add_(0, cols, squares).clamp_min(floor).sqrt()
        )
        ones = torch.ones_like(values)
        row_nnz = values.new_zeros(m).index_add_(0, rows, ones)
        col_nnz = values.new_zeros(n).index_add_(0, cols, ones)
        max_dim = float(max(m, n, 1))
        row_nodes = torch.stack(
            [
                torch.ones(m, dtype=self.dtype, device=device),
                torch.zeros(m, dtype=self.dtype, device=device),
                row_nnz / max(float(n), 1.0),
                torch.log1p(row_abs) / torch.log1p(scale),
                row_sq,
                torch.arange(m, dtype=self.dtype, device=device) / max_dim,
                torch.zeros(m, dtype=self.dtype, device=device),
            ],
            dim=1,
        )
        col_nodes = torch.stack(
            [
                torch.zeros(n, dtype=self.dtype, device=device),
                torch.ones(n, dtype=self.dtype, device=device),
                col_nnz / max(float(m), 1.0),
                torch.log1p(col_abs) / torch.log1p(scale),
                col_sq,
                torch.arange(n, dtype=self.dtype, device=device) / max_dim,
                torch.zeros(n, dtype=self.dtype, device=device),
            ],
            dim=1,
        )

        diag_len = min(m, n)
        if diag_len:
            diagonal = rows == cols
            diag_abs = values.new_zeros(diag_len)
            diag_abs.index_add_(0, rows[diagonal], abs_values[diagonal] / scale)
            row_nodes[:diag_len, 6] = diag_abs
            col_nodes[:diag_len, 6] = diag_abs

        x = torch.cat([row_nodes, col_nodes], dim=0)
        if not torch.isfinite(x).all():
            raise ValueError(
                "matrix statistics overflowed; rescale the matrix or use float64 features"
            )

        src_forward = rows
        dst_forward = m + cols
        src_backward = m + cols
        dst_backward = rows
        edge_index = torch.stack(
            [
                torch.cat([src_forward, src_backward]),
                torch.cat([dst_forward, dst_backward]),
            ],
            dim=0,
        )
        edge_base = torch.stack(
            [values / scale, values.abs() / scale, torch.sign(values)], dim=1
        )
        if not self.use_edge_features:
            edge_base = torch.zeros_like(edge_base)
        edge_attr = torch.cat(
            [
                torch.cat(
                    [
                        edge_base,
                        torch.ones(
                            edge_base.shape[0], 1, device=device, dtype=self.dtype
                        ),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        edge_base,
                        torch.zeros(
                            edge_base.shape[0], 1, device=device, dtype=self.dtype
                        ),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

        density = float(values.numel()) / float(max(m * n, 1))
        global_features = torch.tensor(
            [
                float(m) / max_dim,
                float(n) / max_dim,
                density,
                float(torch.log1p(values.abs().mean())) if values.numel() else 0.0,
                float(torch.log1p(scale)),
                float(torch.linalg.vector_norm(values / scale)),
                self._symmetry_score(indices, values, (m, n)),
                float(diag_len) / max_dim,
            ],
            dtype=self.dtype,
            device=device,
        )

        if not torch.isfinite(global_features).all():
            raise ValueError(
                "matrix statistics overflowed; rescale the matrix or use float64 features"
            )
        target_tensor = (
            None
            if target is None
            else torch.tensor(float(target), dtype=self.dtype, device=device)
        )
        return MatrixGraph(
            x=x,
            edge_index=edge_index.long(),
            edge_attr=edge_attr,
            global_features=global_features,
            shape=(m, n),
            matrix=matrix,
            target=target_tensor,
            metadata=metadata,
        )

    def _entries(self, matrix: Any):
        if isinstance(matrix, torch.Tensor) and matrix.layout != torch.strided:
            coo = matrix.to_sparse_coo().coalesce()
        elif hasattr(matrix, "tocoo") and not sp.isspmatrix(matrix):
            # SciPy is optional: use its public COO protocol without importing it.
            source = matrix.tocoo(copy=True)
            source.sum_duplicates()
            indices = torch.as_tensor(
                np.stack([source.row, source.col]), dtype=torch.long
            )
            coo = torch.sparse_coo_tensor(
                indices,
                torch.as_tensor(source.data),
                source.shape,
                check_invariants=True,
            ).coalesce()
        else:
            dense = matrix_to_dense_tensor(matrix, dtype=self.dtype)
            indices = dense.nonzero().T
            return indices, dense[indices[0], indices[1]], tuple(dense.shape)
        if coo.ndim != 2 or coo.sparse_dim() != 2 or coo.dense_dim() != 0:
            raise ValueError("matrix must be two-dimensional with scalar entries")
        if min(coo.shape) == 0:
            raise ValueError("matrix dimensions must be nonzero")
        values = coo.values()
        if values.is_complex():
            raise ValueError("GNN features currently support real-valued matrices only")
        values = values.to(dtype=self.dtype)
        if not torch.isfinite(values).all():
            raise ValueError(
                "matrix entries must be finite and representable in the feature dtype"
            )
        keep = values != 0
        return coo.indices()[:, keep], values[keep], tuple(coo.shape)

    def _symmetry_score(self, indices, values, shape) -> float:
        if shape[0] != shape[1]:
            return 0.0
        scale = (
            values.abs().max().clamp_min(self.eps)
            if values.numel()
            else values.new_tensor(self.eps)
        )
        normalized = values / scale
        difference = (
            torch.sparse_coo_tensor(
                torch.cat([indices, indices.flip(0)], dim=1),
                torch.cat([normalized, -normalized]),
                shape,
                check_invariants=True,
            )
            .coalesce()
            .values()
        )
        denom = torch.linalg.vector_norm(normalized).clamp_min(self.eps / scale)
        diff = torch.linalg.vector_norm(difference)
        return float(1.0 / (1.0 + diff / denom))
