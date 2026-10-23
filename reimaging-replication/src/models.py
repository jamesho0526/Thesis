from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .data_utils import IMAGE_HEIGHT, IMAGE_WIDTH


class ConvBlock(nn.Module):
    """符合論文圖 7 結構的卷積區塊。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        # Padding 讓池化前的寬度與卷積後空間尺寸維持一致。
        padding = (
            ((5 - 1) * dilation[0]) // 2,
            ((3 - 1) * dilation[1]) // 2,
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(5, 3),
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TradingCNN(nn.Module):
    """復現程式共用的 CNN 模型。"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, IMAGE_HEIGHT, IMAGE_WIDTH),
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        # I5 使用 32x15 輸入與兩個卷積區塊；其他舊版設定仍保留
        # 可調整深度的架構，以維持相容性。
        if num_blocks < 1 or num_blocks > 4:
            raise ValueError("num_blocks must be between 1 and 4")
        channels = [64, 128, 256, 512][:num_blocks]
        layers: List[nn.Module] = []
        in_channels = input_shape[0]
        for idx, out_channels in enumerate(channels):
            if idx == 0 and not (num_blocks == 2 and input_shape[1] == 32):
                layers.append(
                    ConvBlock(
                        in_channels,
                        out_channels,
                        stride=(3, 1),
                        dilation=(2, 1),
                    )
                )
            else:
                # 嚴格 I5 路徑的兩個區塊都使用 stride=1、dilation=1，
                # 展平後會得到 15,360 維特徵。
                layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        feature_dim = self._infer_feature_dim(input_shape)
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(feature_dim, 2)
        self.apply(self._init_weights)

    def _infer_feature_dim(self, input_shape: Tuple[int, int, int]) -> int:
        # 根據實際卷積輸出推導分類器輸入維度，讓形狀錯誤及早發現，
        # 避免無聲地產生錯誤展平結果。
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # 使用 Xavier 初始化，以符合指定的 CNN 規格。
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
