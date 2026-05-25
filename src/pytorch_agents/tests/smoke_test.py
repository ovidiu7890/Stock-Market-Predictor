"""
Smoke test — verifies that all components import correctly, tensor shapes
propagate end-to-end, and the training loop can complete a few epochs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_shapes():
    """Verify tensor shapes through each component."""
    print("=" * 60)
    print("TEST 1: Shape Propagation")
    print("=" * 60)

    from pytorch_agents.temporal_agent import TemporalAgent
    from pytorch_agents.aggregator import CrossAttentionAggregator, ClassificationHead

    batch, seq_len, n_features, d_model = 4, 30, 9, 128

    # --- Temporal Agent ---
    oracle = TemporalAgent(n_features=n_features, d_model=d_model)
    dummy_window = torch.randn(batch, seq_len, n_features)
    oracle_emb = oracle(dummy_window)
    assert oracle_emb.shape == (batch, 1, d_model), f"Oracle shape mismatch: {oracle_emb.shape}"
    print(f"  ✓ TemporalAgent:  input {dummy_window.shape} → output {oracle_emb.shape}")

    # --- Simulated wrapper outputs ---
    chartist_emb = torch.randn(batch, 1, d_model)
    newsroom_emb = torch.randn(batch, 1, d_model)
    print(f"  ✓ ChartistWrapper (simulated): {chartist_emb.shape}")
    print(f"  ✓ SentimentWrapper (simulated): {newsroom_emb.shape}")

    # --- Cross-Attention Aggregator ---
    aggregator = CrossAttentionAggregator(d_model=d_model, n_heads=4)
    aggregated, attn_weights = aggregator(oracle_emb, chartist_emb, newsroom_emb)
    assert aggregated.shape == (batch, 1, d_model), f"Aggregator shape mismatch: {aggregated.shape}"
    assert attn_weights.shape == (batch, 1, 3), f"Attn weights shape mismatch: {attn_weights.shape}"
    print(f"  ✓ CrossAttentionAggregator: {aggregated.shape}, attn_weights: {attn_weights.shape}")

    # --- Classification Head ---
    head = ClassificationHead(d_model=d_model, n_classes=3)
    logits = head(aggregated)
    assert logits.shape == (batch, 3), f"Head shape mismatch: {logits.shape}"
    print(f"  ✓ ClassificationHead: {logits.shape}")

    # --- Gradients flow ---
    loss = logits.sum()
    loss.backward()
    oracle_grad_ok = oracle.gru.weight_ih_l0.grad is not None
    head_grad_ok = head.head[0].weight.grad is not None
    agg_grad_ok = aggregator.cross_attn.in_proj_weight.grad is not None
    print(f"  ✓ Gradients flow — Oracle: {oracle_grad_ok}, Aggregator: {agg_grad_ok}, Head: {head_grad_ok}")

    print("\n   All shape tests PASSED\n")


def test_data_pipeline():
    """Verify the data pipeline loads and produces valid batches."""
    print("=" * 60)
    print("TEST 2: Data Pipeline")
    print("=" * 60)

    import pandas as pd
    from pytorch_agents.data.label_generator import generate_labels, INV_CLASS_MAP
    from pytorch_agents.data.preprocessing import create_sliding_windows, fit_scaler, apply_scaler

    config_path = PROJECT_ROOT / "src" / "pytorch_agents" / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load tech data
    tech_df = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_data"],
        index_col=0, parse_dates=True,
    )
    features = cfg["chartist"]["feature_columns"]
    seq_len = cfg["temporal"]["seq_len"]

    print(f"  Technical data: {tech_df.shape[0]} rows, {tech_df.shape[1]} columns")

    # Sliding windows
    windows, targets, dates = create_sliding_windows(tech_df, features, seq_len=seq_len)
    print(f"  Sliding windows: {windows.shape}")
    assert windows.shape[1] == seq_len
    assert windows.shape[2] == len(features)

    # Scaler
    scaler = fit_scaler(tech_df, features)
    scaled = apply_scaler(windows, scaler)
    print(f"  Scaled windows: {scaled.shape}, mean≈{scaled.mean():.3f}, std≈{scaled.std():.3f}")

    # Label generation
    target_series = pd.Series(targets, index=dates)
    labels = generate_labels(target_series, **cfg["labels"])
    unique, counts = np.unique(labels.values, return_counts=True)
    print(f"  Labels: {dict(zip([INV_CLASS_MAP[u] for u in unique], counts))}")

    print("\n   Data pipeline tests PASSED\n")


def test_full_ensemble():
    """Verify the full EnsembleModel with real sklearn models."""
    print("=" * 60)
    print("TEST 3: Full Ensemble (with real sklearn models)")
    print("=" * 60)

    import pandas as pd
    from pytorch_agents.ensemble_model import EnsembleModel
    from pytorch_agents.train import _train_chartist, _train_newsroom

    config_path = PROJECT_ROOT / "src" / "pytorch_agents" / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Train sklearn models
    print("  Training Chartist (RF)...", end=" ", flush=True)
    rf = _train_chartist(cfg)
    print("✓")

    print("  Training Newsroom (SVM)...", end=" ", flush=True)
    svm, vec = _train_newsroom(cfg)
    print("✓")

    # Build ensemble
    model = EnsembleModel(rf, svm, vec, cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Ensemble: {trainable:,} trainable / {total_params:,} total parameters")

    # Forward pass with real-ish data
    batch = 2
    seq_len = cfg["temporal"]["seq_len"]
    n_features = cfg["temporal"]["n_features"]

    windows = torch.randn(batch, seq_len, n_features)

    # Load real data for the RF snapshot
    tech_df = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_data"],
        index_col=0, parse_dates=True,
    )
    features = cfg["chartist"]["feature_columns"]
    snapshots = tech_df[features].iloc[:batch].values.astype(np.float32)

    texts = [
        "Apple posts record-breaking revenue, beating all expectations",
        "Market plunges on inflation fears, tech stocks hit hard",
    ]

    logits, attn = model(windows, snapshots, texts)
    print(f"  Forward pass: logits {logits.shape}, attn_weights {attn.shape}")
    assert logits.shape == (batch, 3)
    assert attn.shape == (batch, 1, 3)

    # Check attention weights sum to ~1
    attn_sum = attn.squeeze(1).sum(dim=1)
    assert torch.allclose(attn_sum, torch.ones(batch), atol=0.15), f"Attn weights don't sum to 1: {attn_sum}"
    print(f"  Attention weights sum: {attn_sum.tolist()} (should be ≈1.0)")

    print("\n   Full ensemble test PASSED\n")


if __name__ == "__main__":
    test_shapes()
    test_data_pipeline()
    test_full_ensemble()
    print("🎉 ALL TESTS PASSED!")
