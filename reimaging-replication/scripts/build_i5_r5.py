#!/usr/bin/env python3
"""為 S&P 500 或期貨資料建立共用的 I5/R5 影像資料集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.i5_r5 import build_i5_r5_files


def main() -> None:
    # 論文近似復現與期貨延伸共用同一個 builder，
    # 只有輸入格式與資料夾不同。
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sp500", "futures"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="2019-12-31")
    args = parser.parse_args()
    # 建立按年份的二進位影像與 Feather 標籤，
    # 讓訓練使用 memory map，而不必將完整資料集載入 RAM。
    counts = build_i5_r5_files(
        args.source_dir, args.output_dir, args.source, args.start, args.end
    )
    print(f"Generated {sum(counts.values()):,} samples across {len(counts)} years.")
    for year, count in counts.items():
        print(f"{year}: {count:,}")


if __name__ == "__main__":
    main()
