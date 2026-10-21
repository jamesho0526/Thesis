from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history: Dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    epochs = history.get("epochs", [])
    plt.plot(epochs, history.get("train_loss", []), label="Train Loss")
    plt.plot(epochs, history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cumulative_spread(decile_df: pd.DataFrame, output_path: Path, date_column: str = "Date") -> None:
    if decile_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(decile_df[date_column], decile_df["cum_spread"], label="H-L cumulative")
    plt.xlabel("Date")
    plt.ylabel("Cumulative High-Low Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
