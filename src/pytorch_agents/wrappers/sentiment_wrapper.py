

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SentimentWrapper(nn.Module):


    def __init__(
        self,
        svm_model,
        vectorizer,
        d_model: int = 128,
        top_k_tfidf: int = 30,
    ):
        super().__init__()
        self.svm_model = svm_model
        self.vectorizer = vectorizer
        self.top_k = top_k_tfidf

        raw_dim = 2 + top_k_tfidf
        self.projector = nn.Linear(raw_dim, d_model)

    def _extract_features(self, texts: list[str]) -> np.ndarray:

        safe_texts = [t if t.strip() else "no news available" for t in texts]

        X_tfidf = self.vectorizer.transform(safe_texts)
        proba = self.svm_model.predict_proba(X_tfidf)

        batch_size = len(safe_texts)
        top_k_vals = np.zeros((batch_size, self.top_k), dtype=np.float32)
        for i in range(batch_size):
            row = X_tfidf[i].toarray().flatten()
            if len(row) >= self.top_k:
                top_indices = row.argsort()[-self.top_k :]
                top_k_vals[i] = row[top_indices]
            else:
                top_k_vals[i, : len(row)] = np.sort(row)[-len(row) :]

        features = np.column_stack([proba, top_k_vals])
        return features.astype(np.float32)

    def forward(self, texts: list[str]) -> torch.Tensor:
        raw = self._extract_features(texts)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.projector.weight.device)
        projected = self.projector(raw_t)
        return projected.unsqueeze(1)
