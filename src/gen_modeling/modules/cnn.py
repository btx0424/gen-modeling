from __future__ import annotations

import torch
import torch.nn as nn


def init_conv_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=0.02)


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class ConditionalResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(self.act(h)))
        return h + self.skip(x)


class Downsample2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PixelShuffleUpsample2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class ConvBlock(ResidualBlock2d):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        if stride != 1:
            raise ValueError("ConvBlock only supports stride=1. Use Downsample2d for resolution changes.")
        super().__init__(in_channels, out_channels)


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
            layers.append(ResidualBlock2d(c_in, c_out))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
        init_conv_modules(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
