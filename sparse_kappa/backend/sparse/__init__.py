"""Sparse matrix compatibility layer backed by PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch

from .. import torch_api as cp
from . import linalg


@dataclass
class TorchSparseMatrix:
    _dense: torch.Tensor

    def __post_init__(self) -> None:
        if self._dense.ndim != 2:
            raise ValueError("sparse matrices must be 2D")

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self._dense.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._dense.dtype

    @property
    def nnz(self) -> int:
        return int(torch.count_nonzero(self._dense).item())

    @property
    def data(self) -> torch.Tensor:
        return self._dense[self._dense != 0]

    @data.setter
    def data(self, values: torch.Tensor) -> None:
        mask = self._dense != 0
        self._dense[mask] = values.to(dtype=self._dense.dtype, device=self._dense.device)

    @property
    def T(self) -> "TorchSparseMatrix":
        return TorchSparseMatrix(self._dense.mT)

    def conj(self) -> "TorchSparseMatrix":
        return TorchSparseMatrix(torch.conj(self._dense))

    def copy(self) -> "TorchSparseMatrix":
        return TorchSparseMatrix(self._dense.clone())

    def toarray(self) -> torch.Tensor:
        return self._dense

    def tocsr(self) -> "TorchSparseMatrix":
        return self

    def tocsc(self) -> "TorchSparseMatrix":
        return self

    def astype(self, dtype: Any) -> "TorchSparseMatrix":
        return TorchSparseMatrix(self._dense.to(dtype=cp._normalize_dtype(dtype)))

    def sum(self, axis: Optional[int] = None):
        return self._dense.sum() if axis is None else self._dense.sum(dim=axis, keepdim=True)

    def __getitem__(self, key: Any):
        out = self._dense.__getitem__(key)
        return TorchSparseMatrix(out) if isinstance(out, torch.Tensor) and out.ndim == 2 else out

    def __matmul__(self, other: Any):
        rhs = other._dense if isinstance(other, TorchSparseMatrix) else other
        out = self._dense @ rhs
        return TorchSparseMatrix(out) if isinstance(out, torch.Tensor) and out.ndim == 2 else out

    def __rmatmul__(self, other: Any):
        lhs = other._dense if isinstance(other, TorchSparseMatrix) else other
        out = lhs @ self._dense
        return TorchSparseMatrix(out) if isinstance(out, torch.Tensor) and out.ndim == 2 else out

    def _binary(self, other: Any, op):
        rhs = other._dense if isinstance(other, TorchSparseMatrix) else other
        out = op(self._dense, rhs)
        return TorchSparseMatrix(out) if isinstance(out, torch.Tensor) and out.ndim == 2 else out

    def __add__(self, other: Any):
        return self._binary(other, torch.add)

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self, other: Any):
        return self._binary(other, torch.sub)

    def __rsub__(self, other: Any):
        lhs = other._dense if isinstance(other, TorchSparseMatrix) else other
        return TorchSparseMatrix(torch.sub(lhs, self._dense))

    def __mul__(self, other: Any):
        return self._binary(other, torch.mul)

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __truediv__(self, other: Any):
        return self._binary(other, torch.true_divide)

    def __repr__(self) -> str:
        return f"TorchSparseMatrix(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"


spmatrix = TorchSparseMatrix


def _dense_from_any(arg: Any, dtype: Any = None) -> torch.Tensor:
    if isinstance(arg, TorchSparseMatrix):
        dense = arg.toarray()
    elif isinstance(arg, torch.Tensor):
        dense = arg.to_dense() if arg.layout != torch.strided else arg
    elif hasattr(arg, "toarray"):
        dense = cp.asarray(arg.toarray(), dtype=dtype)
    else:
        dense = cp.asarray(arg, dtype=dtype)
    if dtype is not None:
        dense = dense.to(dtype=cp._normalize_dtype(dtype))
    return dense


def csr_matrix(arg: Any, shape: Optional[Tuple[int, int]] = None, dtype: Any = None) -> TorchSparseMatrix:
    if isinstance(arg, tuple) and len(arg) == 2:
        values, coords = arg
        rows, cols = coords
        values = cp.asarray(values, dtype=dtype)
        rows = cp.asarray(rows, dtype=cp.int64)
        cols = cp.asarray(cols, dtype=cp.int64)
        if shape is None:
            shape = (int(rows.max().item()) + 1, int(cols.max().item()) + 1)
        dense = torch.zeros(shape, dtype=values.dtype, device=values.device)
        dense[rows.long(), cols.long()] = values
        return TorchSparseMatrix(dense)
    dense = _dense_from_any(arg, dtype=dtype)
    if shape is not None and tuple(dense.shape) != tuple(shape):
        dense = dense.reshape(shape)
    return TorchSparseMatrix(dense.clone())


def coo_matrix(arg: Any, shape: Optional[Tuple[int, int]] = None, dtype: Any = None) -> TorchSparseMatrix:
    return csr_matrix(arg, shape=shape, dtype=dtype)


def isspmatrix(x: Any) -> bool:
    return isinstance(x, TorchSparseMatrix)


def isspmatrix_csr(x: Any) -> bool:
    return isinstance(x, TorchSparseMatrix)


def eye(n: int, m: Optional[int] = None, dtype: Any = cp.float64, format: str = "csr") -> TorchSparseMatrix:
    return TorchSparseMatrix(cp.eye(n, m, dtype=dtype))


def identity(n: int, dtype: Any = cp.float64, format: str = "csr") -> TorchSparseMatrix:
    return eye(n, dtype=dtype, format=format)


def diags(diagonals: Any, offsets: int = 0, shape: Optional[Tuple[int, int]] = None, format: str = "csr", dtype: Any = None) -> TorchSparseMatrix:
    diag = cp.asarray(diagonals, dtype=dtype)
    if offsets != 0:
        raise NotImplementedError("Only the main diagonal is supported")
    n = diag.numel()
    if shape is None:
        shape = (n, n)
    dense = torch.zeros(shape, dtype=diag.dtype, device=diag.device)
    idx = torch.arange(n, device=diag.device)
    dense[idx, idx] = diag
    return TorchSparseMatrix(dense)


def random(m: int, n: int, density: float = 0.01, format: str = "csr", dtype: Any = cp.float64, random_state: Optional[int] = None) -> TorchSparseMatrix:
    dtype = cp._normalize_dtype(dtype)
    generator = torch.Generator(device=cp._default_device())
    if random_state is not None:
        generator.manual_seed(random_state)
    mask = torch.rand((m, n), generator=generator, device=cp._default_device()) < density
    values = torch.rand((m, n), generator=generator, device=cp._default_device(), dtype=dtype)
    return TorchSparseMatrix(torch.where(mask, values, torch.zeros((), dtype=dtype, device=values.device)))
