from __future__ import annotations

import torch
from torch import nn


class MinesweeperCNN(nn.Module):
    """Small fully convolutional network.

    It outputs one logit per board cell. A larger logit means a higher
    estimated mine probability.
    """

    def __init__(self, in_channels: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
