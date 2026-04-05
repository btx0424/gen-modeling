from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class SmallConvNet(nn.Module):
    """
    Small image backbone for toy experiments.

    It preserves spatial structure and can be reused across:
    - EBM: followed by pooling + scalar head
    - EqM / Flow Matching: followed by a prediction head with image-shaped outputs
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int | None = None,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        out_channels = out_channels or hidden_channels
        widths = [hidden_channels] * max(num_blocks - 1, 0) + [out_channels]

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in widths:
            layers.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
