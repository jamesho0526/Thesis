#!/usr/bin/env python3
"""Train the CNN model on the authors' US monthly image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src import data_utils, device, models, training, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on author-provided image data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data/paper_replication/author_images/monthly_20d"),
        help="Directory containing *_images.dat and *_labels_w_delay.feather files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiments/paper_replication/table_iv_i5_r5"),
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument("--label-column", type=str, default="Ret_20d")
    parser.add_argument("--return-column", type=str, default="Ret_month")
    parser.add_argument("--train-start", type=int, default=1993)
    parser.add_argument("--train-end", type=int, default=2000)
    parser.add_argument("--test-start", type=int, default=2001)
    parser.add_argument("--test-end", type=int, default=2019)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples-per-file", type=int, default=0, help="Optional subsample cap per year.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument("--quiet", action="store_true", help="Disable live training progress.")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Limit to a few samples/epochs just to verify the pipeline locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_pairs = data_utils.discover_file_pairs(data_dir)
    train_pairs, test_pairs, fallback = data_utils.chronological_split(
        file_pairs, args.train_start, args.train_end, args.test_start, args.test_end
    )
    if fallback:
        print("Requested year range not available; using chronological 70/30 split.")

    max_samples = None
    epochs = args.epochs
    if args.max_samples_per_file > 0:
        max_samples = args.max_samples_per_file
    if args.quick_test:
        max_samples = 256
        epochs = 1
        print("Quick test mode: using 256 samples per file and 1 epoch.")

    train_dataset = data_utils.FinancialImageDataset(
        train_pairs, label_column=args.label_column, max_samples_per_file=max_samples
    )
    train_subset, val_subset = data_utils.random_train_val_split(train_dataset, ratio=0.7, seed=42)
    loaders = {
        "train": DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    compute_device = device.get_device()
    model = models.TradingCNN().to(compute_device)
    checkpoint_path = args.output_dir / "best_author_model.pth"

    history = training.fit_model(
        model,
        loaders,
        epochs=epochs,
        patience=args.early_stopping,
        device=compute_device,
        checkpoint_path=checkpoint_path,
        progress=not args.quiet,
    )
    data_utils.save_metrics(history, args.output_dir / "training_history.json")
    visualization.plot_training_history(history, args.output_dir / "training_curve.png")

    if test_pairs:
        test_dataset = data_utils.FinancialImageDataset(
            test_pairs, label_column=args.label_column, max_samples_per_file=max_samples
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = training.evaluate(model, test_loader, criterion, compute_device)
        print(f"Test Loss={test_loss:.4f} Acc={test_acc:.4f}")
        history["test_loss"] = test_loss
        history["test_acc"] = test_acc
        data_utils.save_metrics(history, args.output_dir / "training_history.json")


if __name__ == "__main__":
    main()
