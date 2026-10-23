from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 60
IMAGE_DTYPE = np.uint8
BYTES_PER_SAMPLE = IMAGE_HEIGHT * IMAGE_WIDTH * np.dtype(IMAGE_DTYPE).itemsize


@dataclass(frozen=True)
class DataFilePair:
    """連結影像 .dat 檔與標籤 .feather 檔的資料容器。"""

    image_path: Path
    label_path: Path
    year: int | None = None


def discover_file_pairs(
    data_dir: Path,
    glob_pattern: str = "*_images.dat",
    label_suffix: str = "_labels_w_delay.feather",
) -> List[DataFilePair]:
    """
    Build file pairs by matching *_images.dat files with *_labels*.feather partners.

    Parameters
    ----------
    data_dir : Path
        Directory that contains the .dat/.feather files.
    glob_pattern : str
        Glob pattern to select image files.
    label_suffix : str
        Suffix used to map an image filename to the label filename.
    """

    image_files = sorted(data_dir.glob(glob_pattern))
    if not image_files:
        raise FileNotFoundError(f"No files matching '{glob_pattern}' found in {data_dir}")

    pairs: List[DataFilePair] = []
    for image_file in image_files:
        label_name = image_file.name.replace("_images.dat", label_suffix)
        label_path = image_file.with_name(label_name)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_file.name}")
        match = re.search(r"(\d{4})", image_file.stem)
        year = int(match.group(1)) if match else None
        pairs.append(DataFilePair(image_path=image_file, label_path=label_path, year=year))
    return pairs


class FinancialImageDataset(Dataset):
    """
    Dataset backed by numpy.memmap images and pandas label files.

    The dataset stores a sequence of (memmap, labels) tuples. Each memmap shares the
    same feature dimensions (HxN grayscale images). The labels array stores binary
    class IDs (0 for down, 1 for up). ``image_width`` is 15 for I5 and 60 for
    the existing 20-day datasets.
    """

    def __init__(
        self,
        file_pairs: Sequence[DataFilePair],
        label_column: str,
        max_samples_per_file: int | None = None,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> None:
        super().__init__()
        if not file_pairs:
            raise ValueError("file_pairs must not be empty")
        self.label_column = label_column
        self.image_width = image_width
        self.image_height = image_height
        self.normalization_mean = 0.0
        self.normalization_std = 255.0
        self._records: List[Tuple[np.memmap, np.ndarray]] = []
        self._metadata: List[pd.DataFrame] = []
        self._offsets: List[int] = []
        total = 0
        for pair in file_pairs:
            # 使用 memmap 將大型影像檔保留在磁碟，只把目前要求的樣本載入記憶體。
            img_shape = self._infer_memmap_shape(pair.image_path, image_height, image_width)
            if max_samples_per_file:
                img_shape = (min(max_samples_per_file, img_shape[0]),) + img_shape[1:]
            mm = np.memmap(pair.image_path, mode="r", dtype=IMAGE_DTYPE, shape=img_shape)
            labels_df = pd.read_feather(pair.label_path)
            if self.label_column not in labels_df.columns:
                raise KeyError(f"Column '{self.label_column}' not found in {pair.label_path}")
            labels = (labels_df[self.label_column].iloc[: img_shape[0]] > 0).astype(np.int64)
            self._records.append((mm, labels.to_numpy()))
            self._metadata.append(labels_df.iloc[: img_shape[0]].copy())
            total += img_shape[0]
            self._offsets.append(total)
        self._length = total

    @staticmethod
    def _infer_memmap_shape(
        image_path: Path, image_height: int, image_width: int
    ) -> Tuple[int, int, int]:
        bytes_on_disk = os.path.getsize(image_path)
        bytes_per_sample = image_height * image_width * np.dtype(IMAGE_DTYPE).itemsize
        if bytes_on_disk % bytes_per_sample != 0:
            raise ValueError(f"Unexpected file size for {image_path}")
        samples = bytes_on_disk // bytes_per_sample
        return (samples, image_height, image_width)

    def __len__(self) -> int:
        return self._length

    def _locate_index(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        file_idx = 0
        prev_offset = 0
        for offset in self._offsets:
            if index < offset:
                local_idx = index - prev_offset
                return file_idx, local_idx
            prev_offset = offset
            file_idx += 1
        raise IndexError(index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, local_idx = self._locate_index(index)
        images, labels = self._records[file_idx]
        image = torch.from_numpy(np.array(images[local_idx], dtype=np.float32)).unsqueeze(0)
        # 套用由訓練資料計算的 z-score；預設值只供未設定標準化的舊呼叫端相容使用。
        image = (image - self.normalization_mean) / self.normalization_std
        label = torch.tensor(labels[local_idx], dtype=torch.long)
        return image, label

    def set_normalization(self, mean: float, std: float) -> None:
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("normalization mean/std must be finite and std > 0")
        self.normalization_mean = float(mean)
        self.normalization_std = float(std)

    def pixel_statistics(self) -> Tuple[float, float]:
        # 只使用選定的訓練檔案計算整體像素平均值與標準差。
        total = 0
        value_sum = 0.0
        square_sum = 0.0
        for images, _ in self._records:
            values = np.asarray(images, dtype=np.float64)
            total += values.size
            value_sum += float(values.sum())
            square_sum += float(np.square(values).sum())
        mean = value_sum / max(total, 1)
        variance = max(square_sum / max(total, 1) - mean * mean, 1e-12)
        return mean, float(np.sqrt(variance))

    def metadata_frame(self) -> pd.DataFrame:
        """回傳與資料集順序對齊的合併後 metadata。"""
        if not self._metadata:
            raise RuntimeError("metadata not loaded")
        return pd.concat(self._metadata, axis=0, ignore_index=True)


def chronological_split(
    file_pairs: Sequence[DataFilePair],
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> Tuple[List[DataFilePair], List[DataFilePair], bool]:
    """在隨機切分 Train/Val 前，先依年份選取檔案。"""
    train_val = [p for p in file_pairs if p.year and train_start <= p.year <= train_end]
    test_pairs = [p for p in file_pairs if p.year and test_start <= p.year <= test_end]
    fallback = False
    if not train_val:
        # 期貨資料不涵蓋論文年份，因此只在延伸實驗使用時間順序 fallback，
        # 並將這個選擇回報給使用者。
        fallback = True
        sorted_pairs = sorted(file_pairs, key=lambda p: p.year or 0)
        cutoff = max(1, int(0.7 * len(sorted_pairs)))
        train_val = sorted_pairs[:cutoff]
        test_pairs = sorted_pairs[cutoff:]
    return train_val, test_pairs, fallback


def random_train_val_split(
    dataset: FinancialImageDataset, ratio: float, seed: int
) -> Tuple[Subset[FinancialImageDataset], Subset[FinancialImageDataset]]:
    """只對已選定的樣本內資料集進行隨機切分。"""
    if len(dataset) <= 1:
        indices = list(range(len(dataset)))
        subset = Subset(dataset, indices)
        return subset, subset
    train_len = max(1, int(len(dataset) * ratio))
    if train_len >= len(dataset):
        train_len = len(dataset) - 1
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)
    return train_subset, val_subset


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
