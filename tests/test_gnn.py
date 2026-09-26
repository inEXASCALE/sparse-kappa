"""Tests for GNN condition number prediction helpers."""

from pathlib import Path

import pytest

from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa.gnn import (
    DefaultGraphFeatureExtractor,
    GNNConditionEstimator,
    MatrixConditionDataset,
    TrainingConfig,
    make_gnn_strategy_config,
    train_gnn_condition_estimator,
    train_gnn_strategy_estimator,
)


def _toy_samples():
    A1 = sp.diags(cp.array([1.0, 2.0, 4.0]), format="csr")
    A2 = sp.diags(cp.array([1.0, 3.0, 9.0]), format="csr")
    A3 = sp.diags(cp.array([2.0, 4.0, 8.0]), format="csr")
    return [
        {"matrix": A1, "condition_number": 4.0, "norm_Ainv": 1.0},
        {"matrix": A2, "condition_number": 9.0, "norm_Ainv": 1.0},
        {"matrix": A3, "condition_number": 4.0, "norm_Ainv": 0.5},
    ]


def _strategy_samples():
    A1 = sp.diags(cp.array([1.0, 2.0, 4.0]), format="csr")
    A2 = sp.diags(cp.array([1.0, 3.0, 9.0]), format="csr")
    return [
        {"matrix": A1, "condition_number": 4.0, "norm_A": 4.0},
        {"matrix": A2, "condition_number": 9.0, "norm_A": 9.0},
    ]


def test_default_feature_extractor_builds_bipartite_graph():
    extractor = DefaultGraphFeatureExtractor()
    graph = extractor(sp.diags(cp.array([1.0, 2.0, 4.0]), format="csr"))

    assert graph.x.shape == (6, extractor.node_feature_dim)
    assert graph.edge_attr.shape[1] == extractor.edge_feature_dim
    assert graph.edge_index.shape[0] == 2
    assert graph.global_features.shape == (extractor.global_feature_dim,)


def test_train_save_load_and_predict_direct_condition(tmp_path: Path):
    config = TrainingConfig(epochs=2, lr=1e-3, scheduler="none", target="condition")
    path = tmp_path / "gnn.pt"

    estimator = train_gnn_condition_estimator(
        _toy_samples(), save_path=path, config=config
    )
    pred = estimator.predict(sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"))
    loaded = GNNConditionEstimator.load(path)
    loaded_pred = loaded.predict(sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"))

    assert path.exists()
    assert pred > 0
    assert loaded_pred > 0


def test_inverse_norm_mode_reports_condition_components():
    config = TrainingConfig(
        epochs=1, lr=1e-3, scheduler="none", target="inverse_norm", norm=1
    )
    dataset = MatrixConditionDataset(_toy_samples())
    estimator = train_gnn_condition_estimator(dataset, config=config)
    result = estimator.predict(
        sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"), return_dict=True
    )

    assert result["condition_number"] > 0
    assert result["norm_A"] == 5.0
    assert result["norm_Ainv"] > 0


@pytest.mark.parametrize(
    ("norm", "strategy", "target"),
    [
        (1, 1, "inverse_norm"),
        (2, 1, "inverse_norm"),
        (1, 2, "condition"),
        (2, 2, "condition"),
    ],
)
def test_strategy_config_maps_norm_strategy_pairs(norm, strategy, target):
    config = make_gnn_strategy_config(
        norm=norm, strategy=strategy, epochs=1, scheduler="none"
    )

    assert config.norm == norm
    assert config.strategy == strategy
    assert config.target == target
    assert config.log_base == 10.0


def test_strategy1_derives_inverse_norm_target_from_condition_and_matrix_norm():
    config = make_gnn_strategy_config(norm=1, strategy=1, epochs=1, scheduler="none")
    estimator = GNNConditionEstimator(config=config)
    dataset = MatrixConditionDataset(_strategy_samples())
    sample = dataset[0]

    assert estimator._target_value_from_sample(sample, dataset) == pytest.approx(1.0)


def test_strategy2_uses_direct_condition_target():
    config = make_gnn_strategy_config(norm=2, strategy=2, epochs=1, scheduler="none")
    estimator = GNNConditionEstimator(config=config)
    dataset = MatrixConditionDataset(_strategy_samples())
    sample = dataset[1]

    assert estimator._target_value_from_sample(sample, dataset) == pytest.approx(9.0)


def test_train_strategy_helper_predicts_labeled_quantity():
    config = make_gnn_strategy_config(
        norm=2, strategy=1, epochs=1, lr=1e-3, scheduler="none"
    )
    estimator = train_gnn_strategy_estimator(
        _strategy_samples(), norm=2, strategy=1, config=config
    )
    result = estimator.predict(
        sp.diags(cp.array([1.0, 2.0, 5.0]), format="csr"), return_dict=True
    )

    assert result["condition_number"] > 0
    assert result["target"] == "inverse_norm"
    assert result["strategy"] == 1
    assert result["predicted_quantity"] == "norm_Ainv"


@pytest.mark.parametrize("use_edge_features", [True, False])
@pytest.mark.parametrize("norm,strategy", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_edge_mode_survives_training_and_reload(
    tmp_path, use_edge_features, norm, strategy
):
    import torch

    torch.manual_seed(7)
    config = make_gnn_strategy_config(
        norm=norm,
        strategy=strategy,
        epochs=1,
        scheduler="none",
        device="cpu",
        use_edge_features=use_edge_features,
    )
    estimator = train_gnn_strategy_estimator(
        _strategy_samples(), norm=norm, strategy=strategy, config=config
    )
    matrix = _strategy_samples()[0]["matrix"]
    before = estimator.predict(matrix, return_dict=True)
    path = tmp_path / "checkpoint.pt"
    estimator.save(path)
    loaded = GNNConditionEstimator.load(path)
    assert loaded.config.use_edge_features is use_edge_features
    assert loaded.model.use_edge_features is use_edge_features
    assert loaded.feature_extractor.use_edge_features is use_edge_features
    assert loaded.predict(matrix, return_dict=True) == before


def test_unweighted_model_ignores_edge_attributes():
    import torch
    from dataclasses import replace
    from sparse_kappa.gnn import SparseMatrixGNN

    torch.manual_seed(7)
    graph = DefaultGraphFeatureExtractor()(torch.tensor([[1.0, -2.0], [0.0, 3.0]]))
    changed = replace(graph, edge_attr=graph.edge_attr + 100)
    unweighted = SparseMatrixGNN(use_edge_features=False).eval()
    weighted = SparseMatrixGNN(use_edge_features=True).eval()
    assert torch.equal(unweighted(graph), unweighted(changed))
    assert not torch.allclose(weighted(graph), weighted(changed))
    masked = DefaultGraphFeatureExtractor(use_edge_features=False)(graph.matrix)
    assert torch.count_nonzero(masked.edge_attr[:, :3]) == 0
    assert torch.equal(masked.edge_index, graph.edge_index)
    assert torch.equal(masked.x, graph.x)
    assert torch.equal(masked.global_features, graph.global_features)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_extraction_matches_dense_without_densification(monkeypatch, layout):
    import torch

    dense = torch.tensor([[2.0, -1.0, 0.0], [0.0, 3.0, 4.0]], dtype=torch.float64)
    extractor = DefaultGraphFeatureExtractor()
    expected = extractor(dense)
    sparse = dense.to_sparse_coo() if layout == "coo" else dense.to_sparse_csr()

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse graph construction must not densify")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    actual = extractor(sparse)
    for name in ("x", "edge_index", "edge_attr", "global_features"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


def test_duplicate_sparse_entries_and_explicit_zeros():
    import torch

    indices = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]])
    values = torch.tensor([2.0, -2.0, 3.0, 0.0])
    sparse = torch.sparse_coo_tensor(indices, values, (2, 2), check_invariants=True)
    graph = DefaultGraphFeatureExtractor()(sparse)
    assert graph.edge_index.shape == (2, 2)
    torch.testing.assert_close(graph.edge_attr[:, 0], torch.ones(2))


def test_scipy_sparse_extraction_without_densification(monkeypatch):
    import torch

    scipy_sp = pytest.importorskip("scipy.sparse")

    matrix = scipy_sp.csr_matrix([[1.0, -2.0], [3.0, 0.0]])
    expected = DefaultGraphFeatureExtractor()(matrix.toarray())

    def forbidden(*args, **kwargs):
        raise AssertionError("SciPy graph construction must not densify")

    monkeypatch.setattr(type(matrix), "toarray", forbidden)
    actual = DefaultGraphFeatureExtractor()(matrix)
    torch.testing.assert_close(actual.x, expected.x)
    torch.testing.assert_close(actual.edge_attr, expected.edge_attr)
    torch.testing.assert_close(actual.global_features, expected.global_features)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [[float("nan")]],
        [[float("inf")]],
        [[1 + 2j]],
        [1, 2],
    ],
)
def test_invalid_feature_inputs_fail_explicitly(matrix):
    with pytest.raises(ValueError):
        DefaultGraphFeatureExtractor()(matrix)


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"lr": float("nan")},
        {"norm": 3},
        {"use_edge_features": "false"},
        {"grad_clip": -1},
        {"log_base": float("inf")},
        {"scheduler": "unknown"},
    ],
)
def test_invalid_training_config(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options)


@pytest.mark.parametrize("label", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_training_label_does_not_mutate_model(label):
    import torch

    estimator = GNNConditionEstimator(config=TrainingConfig(epochs=1, device="cpu"))
    before = {k: v.clone() for k, v in estimator.model.state_dict().items()}
    samples = [
        {"matrix": torch.eye(2), "condition_number": 2.0},
        {"matrix": torch.eye(2), "condition_number": label},
    ]
    with pytest.raises(ValueError, match="positive and finite"):
        estimator.fit(samples)
    for key, value in before.items():
        assert torch.equal(value, estimator.model.state_dict()[key])


def test_empty_datasets_are_rejected():
    estimator = GNNConditionEstimator()
    with pytest.raises(ValueError, match="training dataset"):
        estimator.fit([])
    with pytest.raises(ValueError, match="validation dataset"):
        estimator.fit(_toy_samples(), val_data=[])
    with pytest.raises(ValueError, match="evaluation dataset"):
        estimator.evaluate([])


def test_legacy_checkpoint_retains_original_weighted_behavior(tmp_path):
    import torch

    estimator = GNNConditionEstimator(config=TrainingConfig(device="cpu"))
    path = tmp_path / "legacy.pt"
    estimator.save(path)
    payload = torch.load(path, weights_only=True)
    for key in (
        "feature_extractor_config",
        "custom_feature_extractor",
        "checkpoint_version",
    ):
        payload.pop(key)
    payload["training_config"].pop("use_edge_features")
    payload["model_config"].pop("use_edge_features")
    torch.save(payload, path)
    loaded = GNNConditionEstimator.load(path)
    assert loaded.model.use_edge_features is True
    matrix = _toy_samples()[0]["matrix"]
    assert loaded.predict(matrix) == estimator.predict(matrix)


def test_failed_save_preserves_checkpoint(tmp_path, monkeypatch):
    import torch

    estimator = GNNConditionEstimator(config=TrainingConfig(device="cpu"))
    path = tmp_path / "checkpoint.pt"
    estimator.save(path)
    original = path.read_bytes()

    def broken_save(payload, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        estimator.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_float64_feature_configuration_survives_reload(tmp_path):
    import torch

    extractor = DefaultGraphFeatureExtractor(
        dtype=torch.float64, eps=1e-10, use_edge_features=False
    )
    estimator = GNNConditionEstimator(
        feature_extractor=extractor,
        config=TrainingConfig(
            epochs=1, scheduler="none", device="cpu", use_edge_features=False
        ),
    ).fit(_toy_samples())
    path = tmp_path / "float64.pt"
    estimator.save(path)
    loaded = GNNConditionEstimator.load(path)
    assert loaded.feature_extractor.dtype == torch.float64
    assert loaded.feature_extractor.eps == 1e-10
    assert loaded.predict(_toy_samples()[0]["matrix"]) == estimator.predict(
        _toy_samples()[0]["matrix"]
    )


def test_direct_prediction_skips_matrix_norm(monkeypatch):
    import sparse_kappa.gnn.training as training

    def forbidden(*args):
        raise AssertionError("direct scalar prediction must not compute matrix norms")

    monkeypatch.setattr(training, "matrix_norm", forbidden)
    estimator = GNNConditionEstimator(config=TrainingConfig(device="cpu"))
    assert estimator.predict(_toy_samples()[0]["matrix"]) > 0
