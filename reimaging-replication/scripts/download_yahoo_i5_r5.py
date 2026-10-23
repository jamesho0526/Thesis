#!/usr/bin/env python3
"""下載 Yahoo Finance 資料，並保守清理拆股與異常值資料列。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.yahoo_cleaning import clean_tickers


def parse_args() -> argparse.Namespace:
    # 日期範圍要足夠寬，才能建立第一個五日輸入視窗。
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+", help="Yahoo tickers, e.g. AAPL MSFT")
    parser.add_argument("--start", default="1992-12-01")
    parser.add_argument("--end", default="2020-01-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/extensions/yahoo_i5_r5"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 清理函式分別寫出原始資料、建模資料與拒絕資料 audit，
    # 讓資料品質決策可以重現。
    clean_tickers(args.tickers, args.output_dir, start=args.start, end=args.end)
    print(f"Wrote raw, cleaned, and rejected files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
