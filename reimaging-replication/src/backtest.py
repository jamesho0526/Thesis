from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .training import evaluate, predict


def attach_predictions(
    metadata: pd.DataFrame, probs: torch.Tensor, label_column: str
) -> pd.DataFrame:
    """Merge prediction probabilities with metadata and true labels."""
    if len(metadata) < len(probs):
        raise ValueError("Metadata length is shorter than predictions.")
    frame = metadata.iloc[: len(probs)].copy().reset_index(drop=True)
    frame["prob_up"] = probs[:, 1].numpy()
    frame["pred_label"] = (frame["prob_up"] >= 0.5).astype(int)
    frame["true_label"] = (frame[label_column] > 0).astype(int)
    return frame


def classification_metrics(pred_frame: pd.DataFrame) -> Dict[str, float]:
    """Compute accuracy metrics from the prediction frame."""
    correct = (pred_frame["pred_label"] == pred_frame["true_label"]).mean()
    up_mask = pred_frame["true_label"] == 1
    down_mask = ~up_mask
    up_acc = (pred_frame.loc[up_mask, "pred_label"] == 1).mean()
    down_acc = (pred_frame.loc[down_mask, "pred_label"] == 0).mean()
    return {
        "accuracy": float(correct),
        "up_accuracy": float(up_acc),
        "down_accuracy": float(down_acc),
    }


def decile_backtest(
    pred_frame: pd.DataFrame,
    return_column: str,
    date_column: str = "Date",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Sort by prob_up within each date and compute high-low spreads."""
    records = []
    grouped = pred_frame.groupby(date_column)
    for date, group in grouped:
        if group.shape[0] < n_bins:
            continue
        try:
            bins = pd.qcut(group["prob_up"], n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        group = group.assign(decile=bins)
        decile_returns = group.groupby("decile")[return_column].mean()
        high_ret = decile_returns.iloc[-1]
        low_ret = decile_returns.iloc[0]
        records.append(
            {
                date_column: date,
                "high_return": high_ret,
                "low_return": low_ret,
                "spread": high_ret - low_ret,
            }
        )
    if not records:
        return pd.DataFrame(columns=[date_column, "high_return", "low_return", "spread"])
    df = pd.DataFrame(records).sort_values(date_column).reset_index(drop=True)
    df["cum_spread"] = df["spread"].cumsum()
    return df


def run_backtest(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metadata: pd.DataFrame,
    label_column: str,
    return_column: str,
    date_column: str,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Run inference over loader, collect metadata-merged predictions, and
    compute classification & decile metrics.
    """
    probs = predict(model, loader, device)
    frame = attach_predictions(metadata, probs, label_column=label_column)
    metrics = classification_metrics(frame)
    deciles = decile_backtest(frame, return_column=return_column, date_column=date_column)
    return frame, metrics, deciles
