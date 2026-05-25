"""
SentimentWrapper — wraps the frozen SVM + TfidfVectorizer.

Extracts ``predict_proba`` confidence scores and the top-k TF-IDF
activation values, then projects them into the shared ``d_model``
embedding space via a learnable linear layer.

The SVM and vectorizer are **never modified**.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SentimentWrapper(nn.Module):
    """
    Immutable-SVM  →  ``[batch, 1, d_model]`` PyTorch embedding.

    Raw feature vector per sample::

        [P(Good), P(Rubbish), tfidf_top1, tfidf_top2, …, tfidf_topK]

    Parameters
    ----------
    svm_model : sklearn.svm.SVC
        Trained (frozen) SVM with ``probability=True``.
    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Trained (frozen) TF-IDF vectorizer.
    d_model : int
        Target embedding dimension.
    top_k_tfidf : int
        Number of top TF-IDF activation values to keep per sample.
    """

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

        raw_dim = 2 + top_k_tfidf  # P(Good) + P(Rubbish) + top-k values
        self.projector = nn.Linear(raw_dim, d_model)

    # ------------------------------------------------------------------
    def _extract_features(self, texts: list[str]) -> np.ndarray:
        """
        Run the frozen SVM pipeline and assemble the raw feature vector.

        Parameters
        ----------
        texts : list[str]
            Raw text inputs (Article_title + Article).

        Returns
        -------
        np.ndarray, shape ``[batch, raw_dim]``
        """
        # Handle empty strings — replace with a neutral placeholder so the
        # vectorizer doesn't choke.
        safe_texts = [t if t.strip() else "no news available" for t in texts]

        X_tfidf = self.vectorizer.transform(safe_texts)
        proba = self.svm_model.predict_proba(X_tfidf)  # [batch, 2]

        # Extract top-k TF-IDF activation values per sample
        batch_size = len(safe_texts)
        top_k_vals = np.zeros((batch_size, self.top_k), dtype=np.float32)
        for i in range(batch_size):
            row = X_tfidf[i].toarray().flatten()
            if len(row) >= self.top_k:
                top_indices = row.argsort()[-self.top_k :]
                top_k_vals[i] = row[top_indices]
            else:
                # Fewer features than top_k — pad with zeros
                top_k_vals[i, : len(row)] = np.sort(row)[-len(row) :]

        features = np.column_stack([proba, top_k_vals])  # [batch, raw_dim]
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        Tensor, shape ``[batch, 1, d_model]``
        """
        raw = self._extract_features(texts)
        raw_t = torch.tensor(raw, dtype=torch.float32, device=self.projector.weight.device)
        projected = self.projector(raw_t)          # [batch, d_model]
        return projected.unsqueeze(1)               # [batch, 1, d_model]
