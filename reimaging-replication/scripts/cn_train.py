#!/usr/bin/env python3
"""Train and evaluate the CNN using the Chinese image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from src import data_utils, device, models, training, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on CN image dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("../data/extensions/china_images"))
    parser.add_argument("--glob-pattern", type=str, default="cn_*_images*.dat")
    parser.add_argument("--label-suffix", type=str, default="_labels.feather")
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/extensions/china"))
    parser.add_argument("--label-column", type=str, default="Ret_20d")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable live training progress.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_root.expanduser().resolve()

    file_pairs = data_utils.discover_file_pairs(
        data_dir, glob_pattern=args.glob_pattern, label_suffix=args.label_suffix
    )
    max_samples = 512 if args.quick_test else None
    if args.quick_test:
        print("Quick test mode enabled for CN training.")

    dataset = data_utils.FinancialImageDataset(
        file_pairs, label_column=args.label_column, max_samples_per_file=max_samples
    )
    total_len = len(dataset)
    test_len = max(1, int(0.2 * total_len))
    train_val_len = total_len - test_len
    generator = torch.Generator().manual_seed(42)
    train_val_dataset, test_dataset = random_split(dataset, [train_val_len, test_len], generator=generator)
    val_len = max(1, int(0.2 * len(train_val_dataset)))
    train_len = len(train_val_dataset) - val_len
    train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len], generator=generator)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    compute_device = device.get_device()
    model = models.TradingCNN().to(compute_device)
    checkpoint_path = args.output_dir / "best_cn_model.pth"

    history = training.fit_model(
        model,
        loaders,
        epochs=1 if args.quick_test else args.epochs,
        patience=args.early_stopping,
        device=compute_device,
        checkpoint_path=checkpoint_path,
        progress=not args.quiet,
    )
    data_utils.save_metrics(history, args.output_dir / "training_history.json")
    visualization.plot_training_history(history, args.output_dir / "training_curve.png")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = training.evaluate(model, test_loader, criterion, compute_device)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    data_utils.save_metrics(history, args.output_dir / "training_history.json")
    print(
        f"Device={device.describe_device(compute_device)} "
        f"({compute_device}) | CN Test Loss={test_loss:.4f} Acc={test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
