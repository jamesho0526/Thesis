#!/usr/bin/env python3
"""Filter company-named Yahoo CSV files using the 2000-12-29 S&P list."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


ALIASES = {
    "BF.B": "Brown Forman",
    "RDS.A": "Royal Dutch Shell",
    "AZA.A": "Azco",
    "CIT.A": "Citigroup",
    "COC.B": "Coca Cola",
    "F": "Ford",
    "G": "Gillette",
    "H": "Hartford",
    "I": "Intelsat",
    "J": "Jabil",
    "K": "Kellogg",
    "L": "Loews",
    "M": "Macy",
    "Q": "Quintiles",
    "R": "Ryder",
    "S": "Sears",
    "T": "AT&T",
    "U": "USX",
    "X": "US Steel",
}


def normalize(value: str) -> str:
    value = value.upper().replace("&", " AND ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(
        r"\b(THE|INC|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|LP|GROUP)\b",
        " ",
        value,
    )
    return re.sub(r"[^A-Z0-9]+", "", value)


def read_symbols(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    return [symbol.strip() for symbol in re.split(r"[,\s]+", text) if symbol.strip()]


def yahoo_name(symbol: str) -> str | None:
    try:
        import yfinance as yf

        info = yf.Ticker(symbol.replace(".", "-")).get_info()
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None


def load_or_fetch_names(symbols: list[str], cache_path: Path, workers: int) -> dict[str, str]:
    names: dict[str, str] = {}
    if cache_path.exists():
        names.update(json.loads(cache_path.read_text(encoding="utf-8")))
    missing = [symbol for symbol in symbols if symbol not in names]
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(yahoo_name, symbol): symbol for symbol in missing}
            for job in as_completed(jobs):
                symbol = jobs[job]
                try:
                    name = job.result()
                except Exception:
                    name = None
                if name:
                    names[symbol] = name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
    return names


def similarity(left: str, right: str) -> float:
    left_normalized = normalize(left)
    right_normalized = normalize(right)
    if left_normalized and right_normalized and (
        left_normalized in right_normalized or right_normalized in left_normalized
    ):
        return 0.97
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def choose_file(
    symbol: str,
    company_name: str | None,
    files: list[Path],
    min_score: float,
) -> tuple[Path | None, str, float, list[Path]]:
    by_stem = {normalize(path.stem): path for path in files}
    direct = by_stem.get(normalize(symbol))
    if direct:
        return direct, "ticker_stem", 1.0, [direct]

    candidates = []
    names = [name for name in [company_name, ALIASES.get(symbol)] if name]
    for path in files:
        score = max((similarity(name, path.stem) for name in names), default=0.0)
        candidates.append((score, path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return None, "unmatched", 0.0, []
    best_score, best = candidates[0]
    tied = [path for score, path in candidates if best_score - score < 0.015]
    if best_score < min_score:
        return None, "low_confidence", best_score, tied[:5]
    if len(tied) > 1:
        return None, "ambiguous", best_score, tied[:5]
    return best, "company_name", best_score, [best]


def filter_date_range(source: Path, destination: Path, start: str, end: str) -> int:
    frame = pd.read_csv(source)
    date_column = next((column for column in ("Date", "Datetime", "date") if column in frame), None)
    if date_column is None:
        raise ValueError(f"No date column in {source.name}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    keep = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    result = frame.loc[keep].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../data/paper_replication/sp500/raw_1day"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/paper_replication/sp500/filtered_1day"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("../data/paper_replication/sp500/yahoo_names.json"),
    )
    parser.add_argument("--start", default="1993-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.84)
    args = parser.parse_args()

    symbols = read_symbols(args.symbols)
    files = sorted(args.source_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.source_dir}")
    names = load_or_fetch_names(symbols, args.cache, args.workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = 0
    used_files: set[str] = set()
    for symbol in symbols:
        path, method, score, candidates = choose_file(
            symbol, names.get(symbol), files, args.min_score
        )
        if score < args.min_score:
            path = None
            method = "below_min_score"
        if path and path.name in used_files:
            path = None
            method = "duplicate_source_file"
        output_name = path.name if path else ""
        row = {
            "symbol": symbol,
            "yahoo_name": names.get(symbol, ""),
            "source_file": output_name,
            "match_method": method,
            "score": round(score, 4),
            "candidates": "|".join(candidate.name for candidate in candidates),
            "rows_1993_2019": 0,
        }
        if path:
            try:
                row["rows_1993_2019"] = filter_date_range(
                    path, args.output_dir / path.name, args.start, args.end
                )
                if row["rows_1993_2019"] > 0:
                    used_files.add(path.name)
                    selected += 1
                else:
                    (args.output_dir / path.name).unlink(missing_ok=True)
            except Exception as exc:
                row["match_method"] = f"date_filter_error:{type(exc).__name__}"
                target = args.output_dir / path.name
                if target.exists():
                    target.unlink()
        rows.append(row)

    report = args.output_dir / "sp500_2000_filter_report.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Symbols: {len(symbols)}")
    print(f"Selected files: {selected}")
    print(f"Unmatched/ambiguous: {len(symbols) - selected}")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
