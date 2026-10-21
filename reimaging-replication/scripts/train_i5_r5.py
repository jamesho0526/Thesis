#!/usr/bin/env python3
"""Train one resource-conscious I5/R5 CNN with a 1993-2000/2001-2019 split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import data_utils, device, models, training, visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="Comma-separated random seeds for the five-model ensemble.",
    )
    args = parser.parse_args()

    pairs = data_utils.discover_file_pairs(
        args.data_root, glob_pattern="i5_r5_*_images.dat", label_suffix="_labels.feather"
    )
    train_pairs, test_pairs, fallback = data_utils.chronological_split(
        pairs, 1993, 2000, 2001, 2019
    )
    if fallback:
        print(
            "Requested 1993-2000/2001-2019 split is unavailable; "
            "using a chronological 70/30 year split for this extension."
        )
    if not fallback:
        assert train_pairs and test_pairs
        assert all(pair.year is not None and 1993 <= pair.year <= 2000 for pair in train_pairs)
        assert all(pair.year is not None and 2001 <= pair.year <= 2019 for pair in test_pairs)
    cap = args.max_samples_per_file if args.max_samples_per_file > 0 else None
    train_data = data_utils.FinancialImageDataset(
        train_pairs, "Ret_5d", cap, image_width=15, image_height=32
    )
    pixel_mean, pixel_std = train_data.pixel_statistics()
    train_data.set_normalization(pixel_mean, pixel_std)
    print(f"Training pixel z-score: mean={pixel_mean:.6f}, std={pixel_std:.6f}")
    compute_device = device.get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 5:
        raise ValueError("--seeds must contain exactly five comma-separated integers")
    test_data = data_utils.FinancialImageDataset(
        test_pairs, "Ret_5d", cap, image_width=15, image_height=32
    )
    test_data.set_normalization(pixel_mean, pixel_std)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ensemble_probabilities = []
    histories = {}
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_subset, val_subset = data_utils.random_train_val_split(
            train_data, ratio=0.7, seed=seed
        )
        loaders = {
            "train": DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            ),
            "val": DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            ),
        }
        model = models.TradingCNN(input_shape=(1, 32, 15), num_blocks=2).to(compute_device)
        history = training.fit_model(
            model,
            loaders,
            epochs=args.epochs,
            patience=args.early_stopping,
            device=compute_device,
            checkpoint_path=args.output_dir / f"best_i5_r5_seed_{seed}.pth",
            progress=not args.quiet,
        )
        probabilities = training.predict(model, test_loader, compute_device)
        ensemble_probabilities.append(probabilities)
        histories[str(seed)] = history
    ensemble = torch.stack(ensemble_probabilities).mean(dim=0)
    labels = torch.cat([labels for _, labels in test_loader], dim=0)
    predictions = ensemble.argmax(dim=1)
    test_acc = float((predictions == labels).float().mean().item())
    test_loss = float(torch.nn.functional.nll_loss(torch.log(ensemble.clamp_min(1e-8)), labels).item())
    data_utils.save_metrics(
        {
            "seeds": seeds,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "models": histories,
        },
        args.output_dir / "training_history.json",
    )
    prediction_frame = test_data.metadata_frame().copy()
    prediction_frame["prob_down"] = ensemble[:, 0].numpy()
    prediction_frame["prob_up"] = ensemble[:, 1].numpy()
    prediction_frame["prediction"] = predictions.numpy()
    prediction_frame["actual"] = labels.numpy()
    prediction_frame.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    visualization.plot_training_history(histories[str(seeds[0])], args.output_dir / "training_curve_seed_first.png")
    print(
        f"Device={device.describe_device(compute_device)} ({compute_device}) | "
        f"ensemble={len(seeds)} models | Test loss={test_loss:.4f} accuracy={test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
