

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):


    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:

        scores = self.attn(gru_outputs).squeeze(-1)
        weights = F.softmax(scores, dim=-1)

        pooled = torch.bmm(
            weights.unsqueeze(1), gru_outputs
        ).squeeze(1)
        return pooled


class TemporalAgent(nn.Module):

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
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.temporal_pool = TemporalAttentionPool(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_norm(x)
        gru_out, _ = self.gru(x)
        pooled = self.temporal_pool(gru_out)
        return pooled.unsqueeze(1)
