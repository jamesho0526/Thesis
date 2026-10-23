from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

IMG_HEIGHT = 32
IMG_WIDTH = 15
PRICE_ROWS = 26
VOLUME_START = 26


def render_i5(
    window: np.ndarray,
    volume: np.ndarray,
    returns: np.ndarray | None = None,
) -> np.ndarray:
    """使用日報酬率重構價格後，繪製一個 I5 視窗。"""
    # 前 26 列存放 OHLC 與 MA 像素；最後 6 列專門保留給成交量柱。
    raw_ohlc = window[:, :4]
    if returns is None:
        returns = np.zeros(len(raw_ohlc), dtype=float)
    if (
        raw_ohlc.shape != (5, 4)
        or returns.shape != (5,)
        or not np.isfinite(raw_ohlc).all()
        or (raw_ohlc <= 0).any()
        or not np.isfinite(returns).all()
        or (returns <= -1).any()
    ):
        raise ValueError("invalid OHLC window")
    reconstructed_close = np.ones(5, dtype=float)
    # 將第一天收盤價固定為 1，再使用調整後日報酬率重構後續價格，
    # 消除拆股與股利造成的價格幾何變形。
    for day in range(1, 5):
        reconstructed_close[day] = reconstructed_close[day - 1] * (1.0 + returns[day])
    ratios = raw_ohlc / raw_ohlc[:, 3:4]
    adjusted_ohlc = ratios * reconstructed_close[:, None]
    moving_average = (
        pd.Series(reconstructed_close).rolling(5, min_periods=1).mean().to_numpy()
    )
    price_values = np.concatenate((adjusted_ohlc.reshape(-1), moving_average))
    # 尋找局部極值時一併納入 MA5，讓所有價格特徵共用同一個垂直座標系。
    pmin = float(price_values.min())
    prange = max(float(price_values.max() - pmin), 1e-8)
    vmax = max(float(volume.max()), 0.0)
    image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    def row(value: float) -> int:
        normalized = max(0.0, min(1.0, (value - pmin) / prange))
        return int(round((1.0 - normalized) * (PRICE_ROWS - 1)))

    for day, (open_, high, low, close) in enumerate(adjusted_ohlc):
        start = day * 3
        open_row, high_row = row(open_), row(high)
        low_row, close_row = row(low), row(close)
        image[min(high_row, low_row) : max(high_row, low_row) + 1, start + 1] = 255
        image[open_row, start] = 255
        image[close_row, start + 2] = 255
        image[row(moving_average[day]), start + 1] = 255
        if vmax > 0:
            # 成交量只繪製在每個三欄交易日區塊的中間欄，
            # 並由影像底部向上填滿。
            height = max(1, int(round((volume[day] / vmax) * (IMG_HEIGHT - VOLUME_START))))
            image[IMG_HEIGHT - height : IMG_HEIGHT, start + 1] = 255
    return image


def _read_ohlcv(path: Path, source: str) -> pd.DataFrame:
    # S&P 檔案包含欄位標題；期貨檔案則是沒有標題的七欄文字檔。
    if source == "sp500":
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        rename = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
        frame = frame.rename(columns=rename)
    elif source == "futures":
        frame = pd.read_csv(
            path,
            header=None,
            names=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"],
        )
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    else:
        raise ValueError("source must be 'sp500' or 'futures'")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")


def build_i5_r5_files(
    source_dir: Path,
    output_dir: Path,
    source: str,
    start: str = "1993-01-01",
    end: str = "2019-12-31",
) -> dict[str, int]:
    """從 CSV/TXT 價格檔建立按年份儲存的 32x15 I5 影像與 R5 標籤。"""
    pattern = "*.csv" if source == "sp500" else "*.txt"
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {source_dir}")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    by_year: dict[int, list[np.ndarray]] = {}
    labels: dict[int, list[dict[str, object]]] = {}

    for path in files:
        if source == "sp500":
            header = pd.read_csv(path, nrows=0)
            required = {"Date", "Open", "High", "Low", "Close"}
            if not required.issubset(header.columns):
                continue
        frame = _read_ohlcv(path, source)
        frame = frame[frame["Date"].between(start_date, end_date)].reset_index(drop=True)
        if len(frame) < 10:
            continue
        symbol = path.stem
        prices = frame[["Open", "High", "Low", "Close"]].to_numpy(float)
        volume = frame["Volume"].fillna(0).clip(lower=0).to_numpy(float)
        raw_closes = frame["Close"].to_numpy(float)
        adjusted_closes = frame.get("Adj Close", frame["Close"])
        # Yahoo S&P 檔案可能包含 Adj Close；期貨資料使用自身的收盤價，
        # 因此沒有 Adj Close 時退回使用 Close。
        adjusted_closes = pd.to_numeric(adjusted_closes, errors="coerce").to_numpy(float)
        returns = pd.Series(adjusted_closes).pct_change().to_numpy(float).copy()
        returns[0] = 0.0
        dates = frame["Date"].to_numpy()
        for end_idx in range(4, len(frame) - 5):
            try:
                image = render_i5(
                    prices[end_idx - 4 : end_idx + 1],
                    volume[end_idx - 4 : end_idx + 1],
                    returns[end_idx - 4 : end_idx + 1],
                )
            except ValueError:
                continue
            base = adjusted_closes[end_idx]
            future = adjusted_closes[end_idx + 5]
            if not (math.isfinite(base) and math.isfinite(future) and base > 0 and future > 0):
                continue
            year = int(pd.Timestamp(dates[end_idx]).year)
            # 每年儲存成獨立檔案，先執行訓練/OOS 年份隔離，
            # 再進行任何隨機樣本切分。
            by_year.setdefault(year, []).append(image.reshape(-1))
            labels.setdefault(year, []).append(
                {
                    "Symbol": symbol,
                    "Date": pd.Timestamp(dates[end_idx]),
                    "Ret_5d": float(future / base - 1.0),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for year in sorted(by_year):
        images = np.stack(by_year[year]).astype(np.uint8)
        label_frame = pd.DataFrame(labels[year])
        prefix = f"i5_r5_{year}"
        images.tofile(output_dir / f"{prefix}_images.dat")
        label_frame.to_feather(output_dir / f"{prefix}_labels.feather")
        counts[str(year)] = len(label_frame)
    if not counts:
        raise RuntimeError("No I5/R5 samples were generated.")
    return counts
