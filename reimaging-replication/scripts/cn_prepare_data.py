#!/usr/bin/env python3
"""Convert raw Chinese daily Excel files into CSV + image datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import cn_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CN candlestick images from tabular files.")
    parser.add_argument("--data-root", type=Path, default=Path("../日个股回报率文件140354670(仅供复旦大学使用)"))
    parser.add_argument("--file-type", type=str, choices=["excel", "csv"], default="excel")
    parser.add_argument("--tabular-output", type=Path, default=Path("../data/extensions/china_tabular/cn_tabular.csv"))
    parser.add_argument("--image-output-dir", type=Path, default=Path("../data/extensions/china_images"))
    parser.add_argument("--prefix", type=str, default="cn_20d")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--future", type=int, default=20)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    tabular_path = args.tabular_output.expanduser().resolve()
    output_dir = args.image_output_dir.expanduser().resolve()

    df = cn_data.merge_daily_files(data_root, file_type=args.file_type)
    if args.start_year is not None:
        df = df[df["Trddt"].dt.year >= args.start_year]
    if args.end_year is not None:
        df = df[df["Trddt"].dt.year <= args.end_year]

    print(f"Loaded {len(df):,} rows from {args.file_type.upper()} files.")
    cn_data.export_tabular_csv(df, tabular_path)
    print(f"Tabular CSV written to {tabular_path}")

    df = cn_data.compute_adjusted_fields(df, window_days=args.window)
    config = cn_data.CNWindowConfig(window_days=args.window, future_days=args.future)
    images, labels = cn_data.build_cn_dataset(df, config)
    dat_path, feather_path = cn_data.save_image_dataset(images, labels, output_dir, args.prefix)
    print(f"Wrote {len(images)} images to {dat_path}")
    print(f"Wrote labels to {feather_path}")


if __name__ == "__main__":
    main()
