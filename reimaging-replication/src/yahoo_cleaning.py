"""下載 Yahoo Finance 資料，並保守清理每日 OHLCV。"""

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
    """下載單一 ticker 的原始價格、公司行動與 Yahoo Adj Close。"""
    # auto_adjust=False 保留原始 OHLC，讓 clean_history 能將同一個
    # 明確的拆股調整因子一致套用到所有價格欄位。
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
    """回傳較高的異常值門檻，同時避免刪除一般市場崩跌。"""
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
    """回傳清理後資料，以及被拒絕資料列的 audit 表。

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
    # 將 Yahoo 的調整後收盤價比例套用至 OHLC，並對成交量做反向縮放。
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
    # 明確拒絕格式錯誤的資料列；不插值，也不靜默修補會影響影像視窗的資料。
    invalid |= (work[PRICE_COLUMNS + ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose"]] <= 0).any(axis=1)
    invalid |= work["Volume"].isna() | (work["Volume"] < 0)
    invalid |= work["AdjHigh"] < work[["AdjOpen", "AdjClose"]].max(axis=1)
    invalid |= work["AdjLow"] > work[["AdjOpen", "AdjClose"]].min(axis=1)
    invalid |= work["LogAdjReturn"].abs() > limits

    # 調整後仍存在的原始拆股跳躍，視為 Yahoo 資料錯誤。
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
    """下載、清理資料，並為每個 ticker 寫出一份 CSV 與 audit CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        # 將原始、清理後與拒絕資料放在一起，方便日後稽核每次排除。
        raw = download_history(ticker, start=start, end=end)
        raw.to_csv(output_dir / f"{ticker}_raw.csv.gz", index=False, compression="gzip")
        cleaned, rejected = clean_history(raw)
        cleaned.to_csv(output_dir / f"{ticker}_clean.csv.gz", index=False, compression="gzip")
        rejected.to_csv(output_dir / f"{ticker}_rejected.csv.gz", index=False, compression="gzip")
