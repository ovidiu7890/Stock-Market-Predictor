
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .aggregator import ClassificationHead, CrossAttentionAggregator
from .temporal_agent import TemporalAgent
from .wrappers.chartist_wrapper import ChartistWrapper
from .wrappers.sentiment_wrapper import SentimentWrapper


class EnsembleModel(nn.Module):


    def __init__(self, rf_model, svm_model, vectorizer, cfg: dict):
        super().__init__()

        d_model = cfg["model"]["d_model"]

        self.chartist = ChartistWrapper(
            rf_model,
            d_model=d_model,
            n_tree_features=cfg["chartist"]["n_tree_features"],
        )

        self.newsroom = SentimentWrapper(
            svm_model,
            vectorizer,
            d_model=d_model,
            top_k_tfidf=cfg["sentiment"]["top_k_tfidf"],
        )

        self.oracle = TemporalAgent(
            n_features=cfg["temporal"]["n_features"],
            d_model=d_model,
            num_layers=cfg["temporal"]["num_layers"],
            dropout=cfg["temporal"]["dropout"],
        )

        self.aggregator = CrossAttentionAggregator(
            d_model=d_model,
            n_heads=cfg["model"]["n_heads"],
            dropout=cfg["model"]["dropout"],
        )

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
        oracle_emb = self.oracle(windows)
        chartist_emb = self.chartist(snapshots_np)
        newsroom_emb = self.newsroom(texts)

        aggregated, attn_weights = self.aggregator(
            oracle_emb, chartist_emb, newsroom_emb
        )

        logits = self.classifier(aggregated)

        return logits, attn_weights
