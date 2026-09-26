"""Default GNN architecture for sparse matrix scalar prediction."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .data import MatrixGraph


class SparseMatrixGNN(nn.Module):
    """
    Lightweight message-passing network without third-party graph dependencies.

    The model predicts a log-scaled positive scalar. Callers decide whether that
    scalar represents ``log(kappa(A))`` or ``log(||A^{-1}||)``.
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        edge_feature_dim: int = 4,
        global_feature_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_edge_features: bool = True,
    ):
        super().__init__()
        if hidden_dim < 2 or num_layers < 1:
            raise ValueError("hidden_dim must be >= 2 and num_layers must be >= 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(use_edge_features, bool):
            raise ValueError("use_edge_features must be a bool")
        self.use_edge_features = use_edge_features
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: MatrixGraph) -> torch.Tensor:
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        global_features = graph.global_features

        h = self.node_encoder(x)
        e = (
            self.edge_encoder(edge_attr)
            if self.use_edge_features
            else h.new_zeros((edge_index.shape[1], self.hidden_dim))
        )
        src = edge_index[0]
        dst = edge_index[1]

        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            msg_input = torch.cat([h[src], h[dst], e], dim=1)
            messages = message_layer(msg_input)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, messages)
            degree = torch.zeros(h.shape[0], 1, dtype=h.dtype, device=h.device)
            degree.index_add_(
                0, dst, torch.ones(messages.shape[0], 1, dtype=h.dtype, device=h.device)
            )
            agg = agg / degree.clamp_min(1.0)
            h = h + update_layer(torch.cat([h, agg], dim=1))

        pooled_mean = h.mean(dim=0)
        pooled_max = h.max(dim=0).values
        readout_input = torch.cat([pooled_mean, pooled_max, global_features], dim=0)
        return self.readout(readout_input).squeeze(-1)

    def config(self) -> dict:
        return {
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_edge_features": self.use_edge_features,
        }
