
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pytorch_agents.data.dataset import StockDataset, collate_fn
from pytorch_agents.data.label_generator import INV_CLASS_MAP
from pytorch_agents.ensemble_model import EnsembleModel


def _train_chartist(train_tech: pd.DataFrame, cfg: dict):
    features = cfg["chartist"]["feature_columns"]
    X_train = train_tech[features]
    y_train = train_tech["Target_5d_Return"]

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train.values, y_train)
    return rf


def _train_newsroom(cfg: dict):
    path = PROJECT_ROOT / cfg["paths"]["news_data"]
    df = pd.read_csv(path)
    df = df.dropna(subset=["Article"])
    df["Full_Text"] = df["Article_title"].fillna("") + " " + df["Article"].fillna("")

    X = df["Full_Text"]
    y = df["Article_Quality"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    svm = SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=42
    )
    svm.fit(X_train_tfidf, y_train)

    return svm, vectorizer


def train(cfg: dict, epochs: int | None = None, subset: int | None = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n Loading technical data...")
    tech_df = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_data"],
        index_col=0, parse_dates=True,
    )
    news_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["news_data"])

    features = cfg["chartist"]["feature_columns"]
    seq_len = cfg["temporal"]["seq_len"]
    label_cfg = cfg["labels"]

    train_dfs = []
    val_dfs = []

    groups = tech_df.groupby("Company") if "Company" in tech_df.columns else [(None, tech_df)]
    for comp, grp in groups:
        split_idx = int(len(grp) * (1 - cfg["training"]["val_split"]))
        train_dfs.append(grp.iloc[:split_idx])
        val_dfs.append(grp.iloc[split_idx - seq_len:])

    train_tech = pd.concat(train_dfs)
    val_tech = pd.concat(val_dfs)

    print("\n Training the Chartist (Random Forest)...")
    rf_model = _train_chartist(train_tech, cfg)
    print("   [OK] Chartist ready.")

    print(" Training the Newsroom (SVM + TF-IDF)...")
    svm_model, vectorizer = _train_newsroom(cfg)
    print("   [OK] Newsroom ready.")

    train_ds = StockDataset(
        train_tech, news_df, features,
        seq_len=seq_len, label_cfg=label_cfg,
    )
    val_ds = StockDataset(
        val_tech, news_df, features,
        seq_len=seq_len, scaler=train_ds.scaler, label_cfg=label_cfg,
    )

    if subset is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(subset, len(val_ds))))

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    print(f"   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = EnsembleModel(rf_model, svm_model, vectorizer, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Ensemble model built - {trainable_params:,} trainable / {total_params:,} total parameters")

    if hasattr(train_ds, "labels"):
        labels_arr = train_ds.labels
    else:
        labels_arr = np.array([train_ds.dataset.labels[i] for i in train_ds.indices])

    class_counts = np.bincount(labels_arr, minlength=3).astype(float)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    n_epochs = epochs or cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    patience = cfg["training"]["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Starting training for {n_epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for windows, snapshots, texts, labels in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            snapshots_np = snapshots.numpy()

            logits, _ = model(windows, snapshots_np, texts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for windows, snapshots, texts, labels in val_loader:
                windows = windows.to(device)
                labels = labels.to(device)
                snapshots_np = snapshots.numpy()

                logits, _ = model(windows, snapshots_np, texts)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{n_epochs}  |  "
            f"Train Loss {train_loss:.4f}  Acc {train_acc:.3f}  |  "
            f"Val Loss {val_loss:.4f}  Acc {val_acc:.3f}  |  "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"         ->  New best model saved -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print("-" * 70)
    print(f" Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {ckpt_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train the multi-agent ensemble")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "src" / "pytorch_agents" / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--subset", type=int, default=None, help="Use N samples (smoke test)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg, epochs=args.epochs, subset=args.subset)


if __name__ == "__main__":
    main()
