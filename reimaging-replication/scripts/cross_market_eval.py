#!/usr/bin/env python3
"""Transfer author-trained model to CN data and compare with CN-trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src import backtest, data_utils, models, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-market evaluation scripts.")
    parser.add_argument("--cn-data-root", type=Path, default=Path("../data/extensions/china_images"))
    parser.add_argument("--glob-pattern", type=str, default="cn_*_images*.dat")
    parser.add_argument("--label-suffix", type=str, default="_labels.feather")
    parser.add_argument("--label-column", type=str, default="Ret_20d")
    parser.add_argument("--return-column", type=str, default="Ret_20d")
    parser.add_argument("--date-column", type=str, default="Trddt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--author-checkpoint", type=Path, default=Path("./experiments/paper_replication/table_iv_i5_r5/best_author_model.pth"))
    parser.add_argument("--cn-checkpoint", type=Path, default=Path("./experiments/extensions/china/best_cn_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/extensions/cross_market"))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: data_utils.FinancialImageDataset,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
) -> None:
    model = models.TradingCNN().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    metadata = dataset.metadata_frame()
    pred_frame, metrics, deciles = backtest.run_backtest(
        model,
        loader,
        device,
        metadata,
        label_column=args.label_column,
        return_column=args.return_column,
        date_column=args.date_column,
    )
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(out_dir / "predictions.csv", index=False)
    data_utils.save_metrics(metrics, out_dir / "metrics.json")
    deciles.to_csv(out_dir / "deciles.csv", index=False)
    visualization.plot_cumulative_spread(deciles, out_dir / "cumulative_spread.png", args.date_column)
    print(f"{tag} metrics: {metrics}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.cn_data_root.expanduser().resolve()
    file_pairs = data_utils.discover_file_pairs(
        data_dir, glob_pattern=args.glob_pattern, label_suffix=args.label_suffix
    )
    max_samples = None if args.max_samples <= 0 else args.max_samples
    dataset = data_utils.FinancialImageDataset(
        file_pairs, label_column=args.label_column, max_samples_per_file=max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_checkpoint(args.author_checkpoint, dataset, loader, device, args, tag="author_model_on_cn")
    evaluate_checkpoint(args.cn_checkpoint, dataset, loader, device, args, tag="cn_model_on_cn")


if __name__ == "__main__":
    main()
