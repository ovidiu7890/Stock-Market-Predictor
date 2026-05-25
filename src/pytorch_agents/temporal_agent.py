"""
Temporal Agent ("The Oracle") — Bidirectional GRU with temporal attention pooling.

Processes a sliding window of daily OHLCV + engineered features and produces
a single embedding vector that captures sequential momentum patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """
    Learnable attention over timesteps — lets the model decide which days
    in the window matter most for the prediction.

    Instead of naively taking the last hidden state, this computes a
    weighted sum over all GRU outputs.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        gru_outputs : Tensor[batch, seq_len, hidden_dim]

        Returns
        -------
        Tensor[batch, hidden_dim]
        """
        # Compute attention scores
        scores = self.attn(gru_outputs).squeeze(-1)  # [batch, seq_len]
        weights = F.softmax(scores, dim=-1)            # [batch, seq_len]

        # Weighted sum
        pooled = torch.bmm(
            weights.unsqueeze(1), gru_outputs
        ).squeeze(1)  # [batch, hidden_dim]
        return pooled


class TemporalAgent(nn.Module):
    """
    Bidirectional GRU encoder for sequential OHLCV data.

    Architecture::

        LayerNorm(n_features)
            → BiGRU(hidden=d_model//2, layers=2)   # bidirectional doubles → d_model
            → TemporalAttentionPool
            → [batch, 1, d_model]

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    d_model : int
        Output embedding dimension.
    num_layers : int
        Stacked GRU layers.
    dropout : float
        Inter-layer dropout (only applied when ``num_layers > 1``).
    """

    def __init__(
        self,
        n_features: int = 9,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(n_features)

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=d_model // 2,           # bidirectional → d_model
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.temporal_pool = TemporalAttentionPool(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[batch, seq_len, n_features]

        Returns
        -------
        Tensor[batch, 1, d_model]
        """
        x = self.input_norm(x)                       # normalise features
        gru_out, _ = self.gru(x)                     # [batch, seq_len, d_model]
        pooled = self.temporal_pool(gru_out)          # [batch, d_model]
        return pooled.unsqueeze(1)                    # [batch, 1, d_model]
