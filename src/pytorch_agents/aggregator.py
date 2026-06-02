
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionAggregator(nn.Module):

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        oracle_emb: torch.Tensor,
        chartist_emb: torch.Tensor,
        newsroom_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        kv = torch.cat(
            [chartist_emb, newsroom_emb, oracle_emb], dim=1
        )

        attn_out, attn_weights = self.cross_attn(
            query=oracle_emb, key=kv, value=kv
        )

        x = self.norm1(oracle_emb + attn_out)

        x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class ClassificationHead(nn.Module):

    def __init__(self, d_model: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.squeeze(1))
