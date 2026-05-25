"""
ChartistWrapper — wraps the frozen RandomForestRegressor.

Extracts per-tree predictions and summary statistics, then projects them
into the shared ``d_model`` embedding space via a learnable linear layer.

The Random Forest model itself is **never modified** — it operates in pure
NumPy space.  Only the projection layer is trainable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ChartistWrapper(nn.Module):
    """
    Immutable-RF  →  ``[batch, 1, d_model]`` PyTorch embedding.

    Raw feature vector per sample::

        [momentum_score, tree_std, tree_min, tree_max, tree_pred_0, …, tree_pred_k]

    Parameters
    ----------
    rf_model : sklearn.ensemble.RandomForestRegressor
        Trained (frozen) Random Forest model.
    d_model : int
        Target embedding dimension.
    n_tree_features : int
        How many individual tree predictions to include (capped at
        ``n_estimators``).
    """

    def __init__(self, rf_model, d_model: int = 128, n_tree_features: int = 32):
        super().__init__()
        self.rf_model = rf_model
        self.n_trees = len(rf_model.estimators_)
        self.n_tree_features = min(n_tree_features, self.n_trees)

        # 4 summary stats + k individual tree predictions
        raw_dim = 4 + self.n_tree_features
        self.projector = nn.Linear(raw_dim, d_model)

    # ------------------------------------------------------------------
    def _extract_features(self, X_np: np.ndarray) -> np.ndarray:
        """
        Run the frozen RF and assemble the raw feature vector.

        Parameters
        ----------
        X_np : np.ndarray, shape ``[batch, n_input_features]``
            The 9-feature OHLCV+indicator snapshot (original scale).

        Returns
        -------
        np.ndarray, shape ``[batch, raw_dim]``
        """
        # Individual tree predictions  — shape [n_trees, batch]
        tree_preds = np.array(
            [tree.predict(X_np) for tree in self.rf_model.estimators_]
        )
        tree_preds = tree_preds.T  # → [batch, n_trees]

        momentum = self.rf_model.predict(X_np)  # [batch,]

        features = np.column_stack(
            [
                momentum,
                tree_preds.std(axis=1),
                tree_preds.min(axis=1),
                tree_preds.max(axis=1),
                tree_preds[:, : self.n_tree_features],
            ]
        )  # [batch, raw_dim]
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    def forward(self, X_np: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        X_np : np.ndarray, shape ``[batch, n_input_features]``

        Returns
        -------
        Tensor, shape ``[batch, 1, d_model]``
        """
        raw = self._extract_features(X_np)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.projector.weight.device)
        projected = self.projector(raw_t)          # [batch, d_model]
        return projected.unsqueeze(1)               # [batch, 1, d_model]
