"""Numerical/API regressions for supported input forms and explicit failures."""

import numpy as np
import pytest
import torch

from sparse_kappa import ConditionNumberEstimator, cond_estimate


@pytest.mark.parametrize("input_kind", ["numpy", "list", "torch", "coo", "csr"])
@pytest.mark.parametrize("norm", [1, 2])
def test_supported_inputs_share_condition_number(input_kind, norm):
    dense = torch.diag(torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64))
    variants = {
        "numpy": dense.numpy(),
        "list": dense.tolist(),
        "torch": dense,
        "coo": dense.to_sparse_coo(),
        "csr": dense.to_sparse_csr(),
    }
    kwargs = {"solver": "lu"} if norm == 1 else {}
    result = cond_estimate(variants[input_kind], norm=norm, return_dict=True, **kwargs)
    assert result["condition_number"] == pytest.approx(8.0)
    assert result["norm"] == norm


def test_integer_array_and_class_api():
    matrix = np.diag([1, 2, 4])
    result = ConditionNumberEstimator(matrix).estimate()
    assert result["condition_number"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "matrix",
    [np.empty((0, 0)), np.ones((2, 3)), np.array([[np.nan]]), np.array([[np.inf]])],
)
def test_invalid_matrices_raise_value_error(matrix):
    with pytest.raises(ValueError):
        cond_estimate(matrix)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"tol": -1},
        {"tol": float("nan")},
        {"norm": 3},
    ],
)
def test_invalid_numerical_options(kwargs):
    with pytest.raises(ValueError):
        cond_estimate(np.eye(2), **kwargs)


def test_ill_conditioned_nonsymmetric_svd_does_not_square_condition_number():
    angle = 0.7
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    dense = torch.diag(torch.tensor([1.0, 1e-10, 0.2], dtype=torch.float64)) @ rotation
    reference = float(torch.linalg.cond(dense))
    result = cond_estimate(dense, method="svds")
    assert np.isfinite(result)
    assert result == pytest.approx(reference, rel=1e-8)
