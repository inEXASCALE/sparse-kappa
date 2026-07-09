"""GNN-based sparse matrix condition number prediction."""

from .data import MatrixConditionDataset, MatrixGraph
from .features import DefaultGraphFeatureExtractor
from .models import SparseMatrixGNN
from .training import (
    GNNConditionEstimator,
    TrainingConfig,
    make_gnn_strategy_config,
    normalize_strategy,
    strategy_target,
    train_gnn_condition_estimator,
    train_gnn_strategy_estimator,
)

__all__ = [
    "DefaultGraphFeatureExtractor",
    "GNNConditionEstimator",
    "MatrixConditionDataset",
    "MatrixGraph",
    "SparseMatrixGNN",
    "TrainingConfig",
    "make_gnn_strategy_config",
    "normalize_strategy",
    "strategy_target",
    "train_gnn_condition_estimator",
    "train_gnn_strategy_estimator",
]
