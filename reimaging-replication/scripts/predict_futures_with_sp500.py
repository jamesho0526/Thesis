#!/usr/bin/env python3
"""Apply the trained S&P 500 I5/R5 ensemble to futures images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import data_utils, device, models, training


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(252.0) * clean.mean() / clean.std(ddof=1))


def _non_overlapping(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["Symbol", "Date"]).copy()
    position = ordered.groupby("Symbol", sort=False).cumcount()
    return ordered.loc[position.mod(5).eq(0)].reset_index(drop=True)


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual"].to_numpy()
    predicted = frame["prediction"].to_numpy()
    accuracy = float((actual == predicted).mean()) if len(frame) else 0.0
    signal = np.where(frame["prob_up"] >= 0.5, 1.0, -1.0)
    strategy_returns = signal * frame["Ret_5d"].to_numpy(float)
    strategy = pd.Series(strategy_returns, index=frame.index)
    non_overlapping = _non_overlapping(frame)
    non_overlap_signal = np.where(non_overlapping["prob_up"] >= 0.5, 1.0, -1.0)
    non_overlap_returns = pd.Series(
        non_overlap_signal * non_overlapping["Ret_5d"].to_numpy(float)
    )
    non_overlapping["strategy_return"] = non_overlap_returns.to_numpy()
    by_symbol = non_overlapping.groupby("Symbol")["strategy_return"]
    symbol_cumulative = by_symbol.apply(lambda values: float((1.0 + values).prod() - 1.0))
    portfolio = (
        non_overlapping.groupby("Date")["strategy_return"]
        .mean()
        .sort_index()
    )
    return {
        "samples": int(len(frame)),
        "symbols": int(frame["Symbol"].nunique()),
        "accuracy": accuracy,
        "mean_five_day_return": float(frame["Ret_5d"].mean()),
        "mean_signal_return": float(strategy.mean()),
        "non_overlapping_samples": int(len(non_overlapping)),
        "mean_symbol_cumulative_signal_return": float(symbol_cumulative.mean()),
        "median_symbol_cumulative_signal_return": float(symbol_cumulative.median()),
        "portfolio_dates": int(len(portfolio)),
        "equal_weight_date_portfolio_cumulative_return": float(
            (1.0 + portfolio).prod() - 1.0
        ),
        "equal_weight_date_portfolio_annualized_sharpe_approx": _annualized_sharpe(
            portfolio
        ),
        "equal_weight_date_portfolio_max_drawdown_approx": _max_drawdown(portfolio),
        "up_signal_fraction": float((signal > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use the S&P 500 I5/R5 ensemble to predict futures."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data/extensions/futures/images_i5_r5"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./experiments/paper_replication/sp500_i5_r5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiments/extensions/futures_with_sp500_model"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    history_path = args.checkpoint_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing S&P training history: {history_path}")
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    pixel_mean = float(history["pixel_mean"])
    pixel_std = float(history["pixel_std"])

    pairs = data_utils.discover_file_pairs(
        args.data_root,
        glob_pattern="i5_r5_*_images.dat",
        label_suffix="_labels.feather",
    )
    futures_data = data_utils.FinancialImageDataset(
        pairs,
        label_column="Ret_5d",
        image_width=15,
        image_height=32,
    )
    futures_data.set_normalization(pixel_mean, pixel_std)
    loader = DataLoader(
        futures_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    compute_device = device.get_device()
    ensemble_probabilities: list[torch.Tensor] = []
    for seed in seeds:
        checkpoint = args.checkpoint_dir / f"best_i5_r5_seed_{seed}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing S&P checkpoint: {checkpoint}")
        model = models.TradingCNN(input_shape=(1, 32, 15), num_blocks=2).to(compute_device)
        # These checkpoints were created by this project and contain DirectML
        # device metadata that PyTorch 2.4 cannot load with weights_only=True.
        state_dict = torch.load(
            checkpoint, map_location=compute_device, weights_only=False
        )
        model.load_state_dict(state_dict)
        ensemble_probabilities.append(training.predict(model, loader, compute_device))

    ensemble = torch.stack(ensemble_probabilities).mean(dim=0)
    frame = futures_data.metadata_frame().copy()
    frame["prob_down"] = ensemble[:, 0].numpy()
    frame["prob_up"] = ensemble[:, 1].numpy()
    frame["prediction"] = ensemble.argmax(dim=1).numpy()
    frame["actual"] = (frame["Ret_5d"] > 0).astype(int)
    frame["signal_return"] = np.where(frame["prob_up"] >= 0.5, 1.0, -1.0) * frame["Ret_5d"]
    frame = frame.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "futures_predictions_from_sp500.csv", index=False)
    metrics = _metrics(frame)
    metrics.update(
        {
            "device": device.describe_device(compute_device),
            "checkpoint_dir": str(args.checkpoint_dir),
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "seeds": seeds,
        }
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        f"Device={device.describe_device(compute_device)} | "
        f"samples={metrics['samples']:,} | symbols={metrics['symbols']} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"equal_weight_portfolio_return="
        f"{metrics['equal_weight_date_portfolio_cumulative_return']:.4f}"
    )


if __name__ == "__main__":
    main()
