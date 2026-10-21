from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

IMG_HEIGHT = 64
COLS_PER_DAY = 3
VOL_PANEL_START = 49
PRICE_PANEL_ROWS = 49


@dataclass
class CNWindowConfig:
    window_days: int = 20
    future_days: int = 20


def merge_excel_files(source_dir: Path) -> pd.DataFrame:
    excel_files = sorted(source_dir.glob("TRD_Dalyr*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No TRD_Dalyr*.xlsx files found under {source_dir}")
    frames: List[pd.DataFrame] = []
    for path in excel_files:
        frame = pd.read_excel(path, skiprows=[1, 2], dtype={"Stkcd": str})
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["Trddt"] = pd.to_datetime(df["Trddt"])
    numeric_cols = [
        "Opnprc",
        "Hiprc",
        "Loprc",
        "Clsprc",
        "Dnshrtrd",
        "Dnvaltrd",
        "Dsmvosd",
        "Dsmvtll",
        "Dretwd",
        "Dretnd",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Dnshrtrd"] = df["Dnshrtrd"].fillna(0.0)
    df = df.dropna(subset=["Stkcd", "Trddt"])
    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    return df


def compute_adjusted_fields(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    def _adj_close(series: pd.Series) -> pd.Series:
        returns = series.fillna(0.0) + 1.0
        cum = returns.cumprod()
        return cum / cum.iloc[0] if not cum.empty else cum

    df = df.copy()
    df["adj_close"] = df.groupby("Stkcd", group_keys=False)["Dretwd"].transform(_adj_close)
    for raw_col, adj_col in [
        ("Opnprc", "adj_open"),
        ("Hiprc", "adj_high"),
        ("Loprc", "adj_low"),
    ]:
        ratio = df[raw_col] / df["Clsprc"]
        df[adj_col] = df["adj_close"] * ratio
    df["adj_ma"] = (
        df.groupby("Stkcd", group_keys=False)["adj_close"]
        .transform(lambda s: s.rolling(window_days, min_periods=window_days).mean())
    )
    return df


def _price_to_row(value: float, pmin: float, prange: float) -> int:
    if not math.isfinite(value):
        return -1
    norm = (value - pmin) / prange if prange else 0.0
    norm = max(0.0, min(1.0, norm))
    return int(round((1.0 - norm) * (PRICE_PANEL_ROWS - 1)))


def _render_window(
    window_prices: np.ndarray, window_vol: np.ndarray, config: CNWindowConfig
) -> np.ndarray:
    valid_price_mask = np.all(np.isfinite(window_prices[:, :4]), axis=1)
    price_values = window_prices[valid_price_mask, :4].reshape(-1)
    ma_values = window_prices[valid_price_mask, 4]
    ma_values = ma_values[np.isfinite(ma_values)]
    if ma_values.size:
        price_values = np.concatenate([price_values, ma_values])
    if price_values.size == 0:
        raise ValueError("No valid price points for window.")

    pmin = price_values.min()
    pmax = price_values.max()
    prange = max(pmax - pmin, 1e-6)

    vol_valid = window_vol[window_vol > 0]
    vmax = vol_valid.max() if vol_valid.size else 0.0

    image = np.zeros((IMG_HEIGHT, config.window_days * COLS_PER_DAY), dtype=np.uint8)
    for day in range(config.window_days):
        col_start = day * COLS_PER_DAY
        col_end = col_start + COLS_PER_DAY
        if window_vol[day] <= 0:
            continue
        o, h, l, c, ma = window_prices[day]
        if not np.all(np.isfinite([o, h, l, c])):
            continue
        open_row = _price_to_row(o, pmin, prange)
        high_row = _price_to_row(h, pmin, prange)
        low_row = _price_to_row(l, pmin, prange)
        close_row = _price_to_row(c, pmin, prange)
        if min(open_row, high_row, low_row, close_row) < 0:
            continue
        top = min(high_row, low_row)
        bottom = max(high_row, low_row)
        image[top : bottom + 1, col_start + 1] = 255
        image[open_row, col_start] = 255
        image[close_row, col_end - 1] = 255
        if math.isfinite(ma):
            ma_row = _price_to_row(ma, pmin, prange)
            if ma_row >= 0:
                image[ma_row, col_start + 1] = 255

        if vmax > 0:
            vol_norm = max(0.0, min(1.0, window_vol[day] / vmax))
            vol_height = max(1, int(round(vol_norm * (IMG_HEIGHT - VOL_PANEL_START - 1))))
            start_row = IMG_HEIGHT - 1
            end_row = max(VOL_PANEL_START, start_row - vol_height + 1)
            image[end_row : start_row + 1, col_start:col_end] = 255
    return image


def build_cn_dataset(df: pd.DataFrame, config: CNWindowConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    images: List[np.ndarray] = []
    rows: List[Tuple[str, pd.Timestamp, float]] = []
    grouped = df.groupby("Stkcd")
    for stkcd, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) < config.window_days + config.future_days:
            continue
        price_matrix = group[
            ["adj_open", "adj_high", "adj_low", "adj_close", "adj_ma"]
        ].to_numpy(dtype=float)
        volume_vec = group["Dnshrtrd"].fillna(0.0).to_numpy(dtype=float)
        close_vec = group["adj_close"].to_numpy(dtype=float)
        dates = group["Trddt"].to_numpy()

        for end_idx in range(config.window_days - 1, len(group) - config.future_days):
            start_idx = end_idx - config.window_days + 1
            window_prices = price_matrix[start_idx : end_idx + 1]
            window_vol = volume_vec[start_idx : end_idx + 1]
            try:
                image = _render_window(window_prices, window_vol, config)
            except ValueError:
                continue

            future_close = close_vec[end_idx + config.future_days]
            base_close = close_vec[end_idx]
            if not (math.isfinite(future_close) and math.isfinite(base_close) and base_close != 0):
                continue
            future_return = future_close / base_close - 1.0
            images.append(image.reshape(-1))
            rows.append((stkcd, pd.Timestamp(dates[end_idx]), float(future_return)))

    if not images:
        raise RuntimeError("No images were generated. Check data availability.")

    image_array = np.stack(images).astype(np.uint8)
    label_frame = pd.DataFrame(rows, columns=["Stkcd", "Trddt", "Ret_20d"])
    return image_array, label_frame


def export_tabular_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_image_dataset(
    images: np.ndarray,
    labels: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_path = output_dir / f"{prefix}_images.dat"
    feather_path = output_dir / f"{prefix}_labels.feather"
    images.tofile(dat_path)
    labels.reset_index(drop=True).to_feather(feather_path)
    return dat_path, feather_path
