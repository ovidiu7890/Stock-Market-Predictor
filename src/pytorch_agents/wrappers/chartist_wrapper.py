

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ChartistWrapper(nn.Module):


    def __init__(self, rf_model, d_model: int = 128, n_tree_features: int = 32):
        super().__init__()
        self.rf_model = rf_model
        self.n_trees = len(rf_model.estimators_)
        self.n_tree_features = min(n_tree_features, self.n_trees)

        raw_dim = 4 + self.n_tree_features
        self.projector = nn.Linear(raw_dim, d_model)

    def _extract_features(self, X_np: np.ndarray) -> np.ndarray:

        tree_preds = np.array(
            [tree.predict(X_np) for tree in self.rf_model.estimators_]
        )
        tree_preds = tree_preds.T

        momentum = self.rf_model.predict(X_np)

        features = np.column_stack(
            [
                momentum,
                tree_preds.std(axis=1),
                tree_preds.min(axis=1),
                tree_preds.max(axis=1),
                tree_preds[:, : self.n_tree_features],
            ]
        )
        return features.astype(np.float32)

    def forward(self, X_np: np.ndarray) -> torch.Tensor:

        raw = self._extract_features(X_np)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.projector.weight.device)
        projected = self.projector(raw_t)
        return projected.unsqueeze(1)
