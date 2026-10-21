"""Yahoo Finance download and conservative daily OHLCV cleaning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_history(
    ticker: str,
    start: str = "1992-12-01",
    end: str = "2020-01-01",
) -> pd.DataFrame:
    """Download one ticker with raw prices, actions, and Yahoo Adj Close."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'yfinance'. Install with: py -m pip install yfinance"
        ) from exc

    frame = yf.Ticker(ticker).history(
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        repair=True,
    )
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no rows for {ticker}.")
    frame = frame.reset_index()
    date_column = "Datetime" if "Datetime" in frame.columns else "Date"
    frame = frame.rename(columns={date_column: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
    frame["Ticker"] = ticker
    return frame


def _robust_return_limit(log_returns: pd.Series, hard_limit: float) -> float:
    """Return a high threshold while avoiding deletion of ordinary crashes."""
    values = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return hard_limit
    median = float(values.median())
    mad = float((values - median).abs().median())
    robust_limit = abs(median) + 12.0 * 1.4826 * mad
    return max(hard_limit, robust_limit)


def clean_history(
    frame: pd.DataFrame,
    *,
    outlier_hard_limit: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned rows and an audit table of rejected rows.

    Yahoo's raw OHLC is retained in the audit/output columns. The modeling
    columns are split-adjusted with Adj Close / Close, so a stock split is
    normalized rather than incorrectly treated as a return shock.
    """
    required = {"Ticker", "Date", *RAW_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Yahoo columns: {sorted(missing)}")

    work = frame.copy()
    work = work.sort_values(["Ticker", "Date"]).drop_duplicates(
        ["Ticker", "Date"], keep="last"
    )
    numeric = list(set(RAW_COLUMNS) | {"Dividends", "Stock Splits"})
    for column in numeric:
        if column not in work:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["AdjFactor"] = work["Adj Close"] / work["Close"]
    work["AdjOpen"] = work["Open"] * work["AdjFactor"]
    work["AdjHigh"] = work["High"] * work["AdjFactor"]
    work["AdjLow"] = work["Low"] * work["AdjFactor"]
    work["AdjClose"] = work["Adj Close"]
    work["AdjVolume"] = work["Volume"] / work["AdjFactor"].replace(0, np.nan)

    group = work.groupby("Ticker", sort=False, group_keys=False)
    work["AdjReturn"] = group["AdjClose"].pct_change()
    work["RawReturn"] = group["Close"].pct_change()
    work["LogAdjReturn"] = np.log1p(work["AdjReturn"].where(work["AdjReturn"] > -1))
    limits = work.groupby("Ticker")["LogAdjReturn"].transform(
        lambda values: _robust_return_limit(values, outlier_hard_limit)
    )

    invalid = (
        work[PRICE_COLUMNS + ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose"]]
        .isna()
        .any(axis=1)
    )
    invalid |= (work[PRICE_COLUMNS + ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose"]] <= 0).any(axis=1)
    invalid |= work["Volume"].isna() | (work["Volume"] < 0)
    invalid |= work["AdjHigh"] < work[["AdjOpen", "AdjClose"]].max(axis=1)
    invalid |= work["AdjLow"] > work[["AdjOpen", "AdjClose"]].min(axis=1)
    invalid |= work["LogAdjReturn"].abs() > limits

    # A raw split jump that remains after adjustment is a Yahoo data error.
    raw_jump = work["RawReturn"].abs() > 0.40
    split_nearby = work["Stock Splits"].ne(0) | group["Stock Splits"].shift().fillna(0).ne(0)
    adjusted_jump = work["AdjReturn"].abs() > outlier_hard_limit
    invalid |= raw_jump & ~split_nearby & adjusted_jump

    reasons = pd.Series("", index=work.index, dtype="object")
    reasons.loc[invalid & work[PRICE_COLUMNS].isna().any(axis=1)] = "missing_raw_price"
    reasons.loc[invalid & (work[PRICE_COLUMNS] <= 0).any(axis=1)] = "nonpositive_raw_price"
    reasons.loc[invalid & (work["Volume"].isna() | (work["Volume"] < 0))] = "invalid_volume"
    reasons.loc[invalid & (work["AdjHigh"] < work[["AdjOpen", "AdjClose"]].max(axis=1))] = "high_below_ohlc"
    reasons.loc[invalid & (work["AdjLow"] > work[["AdjOpen", "AdjClose"]].min(axis=1))] = "low_above_ohlc"
    reasons.loc[invalid & (work["LogAdjReturn"].abs() > limits)] = "robust_return_outlier"
    reasons.loc[invalid & (raw_jump & ~split_nearby & adjusted_jump)] = "unadjusted_jump"
    rejected = work.loc[invalid].copy()
    rejected["RejectReason"] = reasons.loc[invalid]
    cleaned = work.loc[~invalid].copy()
    cleaned["SplitAdjusted"] = cleaned["Stock Splits"].ne(0)
    return cleaned, rejected


def clean_tickers(
    tickers: Iterable[str],
    output_dir: Path,
    start: str = "1992-12-01",
    end: str = "2020-01-01",
) -> None:
    """Download, clean, and write one CSV plus one audit CSV per ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        raw = download_history(ticker, start=start, end=end)
        raw.to_csv(output_dir / f"{ticker}_raw.csv.gz", index=False, compression="gzip")
        cleaned, rejected = clean_history(raw)
        cleaned.to_csv(output_dir / f"{ticker}_clean.csv.gz", index=False, compression="gzip")
        rejected.to_csv(output_dir / f"{ticker}_rejected.csv.gz", index=False, compression="gzip")
