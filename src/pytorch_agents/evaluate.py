"""
Evaluation script — loads a trained checkpoint and reports detailed metrics.

Usage::

    python -m pytorch_agents.evaluate
    python -m pytorch_agents.evaluate --checkpoint src/pytorch_agents/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pytorch_agents.data.dataset import StockDataset, collate_fn
from pytorch_agents.data.label_generator import INV_CLASS_MAP
from pytorch_agents.ensemble_model import EnsembleModel
from pytorch_agents.train import _train_chartist, _train_newsroom


def evaluate(cfg: dict, checkpoint_path: str):
    """Load a checkpoint and evaluate on the validation split."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-create the sklearn models
    print(" Re-training Chartist (RF) for evaluation wrapper...")
    rf_model = _train_chartist(cfg)
    print(" Re-training Newsroom (SVM) for evaluation wrapper...")
    svm_model, vectorizer = _train_newsroom(cfg)

    # Load data
    tech_df = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_data"],
        index_col=0, parse_dates=True,
    )
    news_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["news_data"])

    features = cfg["chartist"]["feature_columns"]
    seq_len = cfg["temporal"]["seq_len"]
    label_cfg = cfg["labels"]

    # Use the same split as training
    split_idx = int(len(tech_df) * (1 - cfg["training"]["val_split"]))
    train_tech = tech_df.iloc[:split_idx]
    val_tech = tech_df.iloc[split_idx - seq_len:]

    train_ds = StockDataset(
        train_tech, news_df, features, seq_len=seq_len, label_cfg=label_cfg,
    )
    val_ds = StockDataset(
        val_tech, news_df, features,
        seq_len=seq_len, scaler=train_ds.scaler, label_cfg=label_cfg,
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
    )

    # Build model & load weights
    model = EnsembleModel(rf_model, svm_model, vectorizer, cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.3f})")

    # Predict
    all_preds = []
    all_labels = []
    all_attn = []

    with torch.no_grad():
        for windows, snapshots, texts, labels in val_loader:
            windows = windows.to(device)
            snapshots_np = snapshots.numpy()

            logits, attn_weights = model(windows, snapshots_np, texts)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_attn.append(attn_weights.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_attn = np.concatenate(all_attn, axis=0)  # [N, 1, 3]

    # --- Classification Report ---
    target_names = [INV_CLASS_MAP[i] for i in range(3)]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    cm_formatted = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm[i, j]:>3d} ({cm_pct[i, j]:>3.0%})")
        cm_formatted.append(row)

    print("CONFUSION MATRIX (with row percentages)")
    print("-" * 60)
    cm_df = pd.DataFrame(cm_formatted, index=target_names, columns=target_names)
    print(cm_df.to_string())

    correct_preds = (all_labels == all_preds).sum()
    total_preds = len(all_labels)
    print(f"\nOVERALL ACCURACY: {correct_preds} / {total_preds} ({correct_preds/total_preds:.1%})")

    # --- Attention Weight Analysis ---
    avg_attn = all_attn.squeeze(1).mean(axis=0)  # [3]
    agent_names = ["Chartist", "Newsroom", "Oracle"]
    print("\n" + "-" * 40)
    print("AVERAGE CROSS-ATTENTION WEIGHTS")
    print("-" * 40)
    for name, weight in zip(agent_names, avg_attn):
        bar = "█" * int(weight * 40)
        print(f"  {name:>10s}:  {weight:.3f}  {bar}")

    return all_preds, all_labels, all_attn


def main():
    parser = argparse.ArgumentParser(description="Evaluate the multi-agent ensemble")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "src" / "pytorch_agents" / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "src" / "pytorch_agents" / "checkpoints" / "best.pt"),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
