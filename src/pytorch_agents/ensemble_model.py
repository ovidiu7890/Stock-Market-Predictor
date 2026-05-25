"""
EnsembleModel — top-level ``nn.Module`` that wires together all three
agents and the cross-attention aggregator.

This is the single entry point for both training and inference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .aggregator import ClassificationHead, CrossAttentionAggregator
from .temporal_agent import TemporalAgent
from .wrappers.chartist_wrapper import ChartistWrapper
from .wrappers.sentiment_wrapper import SentimentWrapper


class EnsembleModel(nn.Module):
    """
    Full multi-agent ensemble with cross-attention.

    Parameters
    ----------
    rf_model : sklearn RandomForestRegressor (frozen).
    svm_model : sklearn SVC (frozen).
    vectorizer : sklearn TfidfVectorizer (frozen).
    cfg : dict
        Merged configuration dictionary (from ``default.yaml``).
    """

    def __init__(self, rf_model, svm_model, vectorizer, cfg: dict):
        super().__init__()

        d_model = cfg["model"]["d_model"]

        # --- Agent 1: The Chartist (frozen RF → projection) ---
        self.chartist = ChartistWrapper(
            rf_model,
            d_model=d_model,
            n_tree_features=cfg["chartist"]["n_tree_features"],
        )

        # --- Agent 2: The Newsroom (frozen SVM → projection) ---
        self.newsroom = SentimentWrapper(
            svm_model,
            vectorizer,
            d_model=d_model,
            top_k_tfidf=cfg["sentiment"]["top_k_tfidf"],
        )

        # --- Agent 3: The Oracle (trainable GRU) ---
        self.oracle = TemporalAgent(
            n_features=cfg["temporal"]["n_features"],
            d_model=d_model,
            num_layers=cfg["temporal"]["num_layers"],
            dropout=cfg["temporal"]["dropout"],
        )

        # --- Cross-Attention Aggregator ---
        self.aggregator = CrossAttentionAggregator(
            d_model=d_model,
            n_heads=cfg["model"]["n_heads"],
            dropout=cfg["model"]["dropout"],
        )

        # --- Classification Head ---
        self.classifier = ClassificationHead(
            d_model=d_model,
            n_classes=3,
            dropout=cfg["model"]["head_dropout"],
        )

    def forward(
        self,
        windows: torch.Tensor,
        snapshots_np: np.ndarray,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        windows     : Tensor[batch, seq_len, n_features] — scaled OHLCV windows
        snapshots_np : np.ndarray[batch, n_features]     — raw-scale last-day features
        texts       : list[str], length batch            — aligned news texts

        Returns
        -------
        logits       : Tensor[batch, 3]
        attn_weights : Tensor[batch, 1, 3]
        """
        # Produce embeddings from each agent
        oracle_emb = self.oracle(windows)                   # [batch, 1, d_model]
        chartist_emb = self.chartist(snapshots_np)          # [batch, 1, d_model]
        newsroom_emb = self.newsroom(texts)                 # [batch, 1, d_model]

        # Cross-attention aggregation
        aggregated, attn_weights = self.aggregator(
            oracle_emb, chartist_emb, newsroom_emb
        )  # [batch, 1, d_model]

        # Classify
        logits = self.classifier(aggregated)                # [batch, 3]

        return logits, attn_weights
