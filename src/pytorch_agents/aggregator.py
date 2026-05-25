"""
Cross-Attention Aggregator and Classification Head.

The aggregator lets the Temporal Agent ("Oracle") query the Chartist and
Newsroom embeddings via multi-head cross-attention, dynamically weighting
each agent's contribution based on the current market context.

The classification head maps the aggregated representation to
Buy / Sell / Hold logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionAggregator(nn.Module):
    """
    Transformer-style cross-attention block.

    Q = Oracle (Temporal Agent) token
    KV = [Chartist, Newsroom, Oracle]  (all three agents)

    Includes residual connections, LayerNorm, and a position-wise FFN.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for attention and FFN.
    """

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
        """
        Parameters
        ----------
        oracle_emb   : Tensor[batch, 1, d_model]
        chartist_emb : Tensor[batch, 1, d_model]
        newsroom_emb : Tensor[batch, 1, d_model]

        Returns
        -------
        output       : Tensor[batch, 1, d_model]
        attn_weights : Tensor[batch, 1, 3]  — attention over the 3 agents
        """
        # Concatenate all three agent tokens as keys/values
        kv = torch.cat(
            [chartist_emb, newsroom_emb, oracle_emb], dim=1
        )  # [batch, 3, d_model]

        # Cross-attention: Oracle queries all agents
        attn_out, attn_weights = self.cross_attn(
            query=oracle_emb, key=kv, value=kv
        )  # attn_out: [batch, 1, d_model]

        # Residual + LayerNorm
        x = self.norm1(oracle_emb + attn_out)

        # Feed-forward + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class ClassificationHead(nn.Module):
    """
    Maps the aggregated ``[batch, 1, d_model]`` representation to
    3-class logits  (Sell=0, Hold=1, Buy=2).

    Architecture::

        Linear(d_model → d_model // 2) → GELU → Dropout → Linear(→ 3)

    Parameters
    ----------
    d_model : int
        Input embedding dimension.
    n_classes : int
        Number of output classes (default 3).
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[batch, 1, d_model]

        Returns
        -------
        logits : Tensor[batch, n_classes]
        """
        return self.head(x.squeeze(1))
