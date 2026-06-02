
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pytorch_agents.data.label_generator import INV_CLASS_MAP
from pytorch_agents.data.preprocessing import apply_scaler, fit_scaler
from pytorch_agents.ensemble_model import EnsembleModel
from pytorch_agents.train import _train_chartist, _train_newsroom


def predict(
    cfg: dict,
    checkpoint_path: str,
    headline: str | None = None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(" Loading Chartist (RF)...")
    rf_model = _train_chartist(cfg)
    print(" Loading Newsroom (SVM)...")
    svm_model, vectorizer = _train_newsroom(cfg)

    model = EnsembleModel(rf_model, svm_model, vectorizer, cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f" Loaded checkpoint (epoch {ckpt['epoch']})")

    features = cfg["chartist"]["feature_columns"]
    seq_len = cfg["temporal"]["seq_len"]

    tech_prepared = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_data"],
        index_col=0, parse_dates=True,
    )
    tech_live = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["technical_live"],
        index_col=0, parse_dates=True,
    )
    combined = pd.concat([tech_prepared, tech_live]).tail(seq_len)

    if len(combined) < seq_len:
        print(f"  Only {len(combined)} rows available, need {seq_len}. "
              "Using what we have.")
        pad_count = seq_len - len(combined)
        padding = pd.DataFrame(
            [combined.iloc[0]] * pad_count,
            columns=combined.columns,
        )
        padding.index = pd.date_range(
            end=combined.index[0] - pd.Timedelta(days=1),
            periods=pad_count, freq="B",
        )
        combined = pd.concat([padding, combined])

    scaler = fit_scaler(tech_prepared, features)
    window = combined[features].values[-seq_len:]
    window_scaled = scaler.transform(window)
    window_tensor = torch.tensor(
        window_scaled, dtype=torch.float32
    ).unsqueeze(0).to(device)

    snapshot_np = combined[features].values[-1:].astype(np.float32)

    if headline is None:
        news_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["news_data"])
        news_df = news_df.dropna(subset=["Article"])
        news_df["Full_Text"] = (
            news_df["Article_title"].fillna("") + " " + news_df["Article"].fillna("")
        )
        text = news_df["Full_Text"].iloc[-1]
    else:
        text = headline

    with torch.no_grad():
        logits, attn_weights = model(window_tensor, snapshot_np, [text])

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(logits.argmax(dim=1).item())
    pred_label = INV_CLASS_MAP[pred_class]
    attn = attn_weights.cpu().numpy().squeeze()

    print("\n" + "=" * 60)
    print("    MULTI-AGENT ENSEMBLE PREDICTION")
    print("=" * 60)
    print(f"\n    Signal:  {pred_label.upper()}")
    print(f"\n  Confidence breakdown:")
    for cls_id in range(3):
        label = INV_CLASS_MAP[cls_id]
        bar = "█" * int(probs[cls_id] * 30)
        print(f"    {label:>5s}:  {probs[cls_id]:.1%}  {bar}")

    print(f"\n  Agent attention weights:")
    agent_names = ["Chartist", "Newsroom", "Oracle"]
    for name, w in zip(agent_names, attn):
        bar = "█" * int(w * 30)
        print(f"    {name:>10s}:  {w:.3f}  {bar}")

    if headline:
        print(f"\n    Headline used: \"{headline}\"")
    print("=" * 60)

    return pred_label, probs, attn


def main():
    parser = argparse.ArgumentParser(description="Run inference with the ensemble")
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
    parser.add_argument(
        "--headline",
        type=str,
        default=None,
        help='Custom news headline, e.g. "Apple posts record revenue"',
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    predict(cfg, args.checkpoint, headline=args.headline)


if __name__ == "__main__":
    main()
