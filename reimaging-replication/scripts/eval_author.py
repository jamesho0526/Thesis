#!/usr/bin/env python3
"""Evaluate a trained model on the author dataset and generate diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src import backtest, data_utils, device, models, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference/backtests on author data.")
    parser.add_argument("--data-root", type=Path, default=Path("../data/paper_replication/author_images/monthly_20d"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./experiments/paper_replication/table_iv_i5_r5/best_author_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/paper_replication/table_iv_i5_r5/eval"))
    parser.add_argument("--label-column", type=str, default="Ret_20d")
    parser.add_argument("--return-column", type=str, default="Ret_month")
    parser.add_argument("--date-column", type=str, default="Date")
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_root.expanduser().resolve()

    file_pairs = data_utils.discover_file_pairs(data_dir)
    eval_pairs = [p for p in file_pairs if p.year and args.start_year <= p.year <= args.end_year]
    if not eval_pairs:
        raise ValueError("No file pairs found for selected years.")
    max_samples = None if args.max_samples_per_file <= 0 else args.max_samples_per_file
    dataset = data_utils.FinancialImageDataset(
        eval_pairs, label_column=args.label_column, max_samples_per_file=max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    compute_device = device.get_device()
    model = models.TradingCNN().to(compute_device)
    state_dict = torch.load(args.checkpoint, map_location=compute_device)
    model.load_state_dict(state_dict)

    metadata = dataset.metadata_frame()
    pred_frame, metrics, deciles = backtest.run_backtest(
        model,
        loader,
        compute_device,
        metadata,
        label_column=args.label_column,
        return_column=args.return_column,
        date_column=args.date_column,
    )

    pred_path = args.output_dir / f"predictions_{args.start_year}_{args.end_year}.csv"
    metrics_path = args.output_dir / f"metrics_{args.start_year}_{args.end_year}.json"
    decile_path = args.output_dir / f"deciles_{args.start_year}_{args.end_year}.csv"

    pred_frame.to_csv(pred_path, index=False)
    data_utils.save_metrics(metrics, metrics_path)
    deciles.to_csv(decile_path, index=False)
    visualization.plot_cumulative_spread(deciles, args.output_dir / "cumulative_spread.png", args.date_column)
    print(f"Saved predictions to {pred_path}")
    print(metrics)


if __name__ == "__main__":
    main()
