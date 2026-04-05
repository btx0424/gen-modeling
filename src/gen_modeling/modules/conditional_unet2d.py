from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import (
    ConditionalResidualBlock2d,
    Downsample2d,
    PixelShuffleUpsample2d,
    init_conv_modules,
)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    DDPM-style Fourier features for scalar time (e.g. flow time in [0, 1]).
    t: (B,) or broadcastable to (B,). Returns (B, dim); dim must be even.
    """
    if dim % 2 != 0:
        raise ValueError(f"cond_dim must be even for time embedding, got {dim}")
    t_flat = t.reshape(-1).float()
    half = dim // 2
    device = t_flat.device
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = t_flat[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb.to(dtype=t.dtype)


class ConditionalUNet2D(nn.Module):
    """
    Small class-conditional U-Net backbone for image generation experiments.

    Optional continuous time ``t`` is embedded (sinusoidal + MLP) and added to
    the label embedding so every ``ConditionalResidualBlock2d`` gets FiLM from
    both class and time. Pass ``t=None`` for time-agnostic use (e.g. EqM).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_classes: int,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        self.null_class_idx = num_classes

        self.label_embedding = nn.Embedding(num_classes + 1, cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        widths = [base_channels * mult for mult in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_channels = base_channels
        for idx, width in enumerate(widths):
            self.down_blocks.append(ConditionalResidualBlock2d(cur_channels, width, cond_dim))
            cur_channels = width
            if idx < len(widths) - 1:
                self.downsamples.append(Downsample2d(cur_channels))

        self.mid_block1 = ConditionalResidualBlock2d(cur_channels, cur_channels, cond_dim)
        self.mid_block2 = ConditionalResidualBlock2d(cur_channels, cur_channels, cond_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(len(widths) - 1, -1, -1):
            skip_channels = widths[idx]
            self.up_blocks.append(ConditionalResidualBlock2d(cur_channels + skip_channels, skip_channels, cond_dim))
            cur_channels = skip_channels
            if idx > 0:
                self.upsamples.append(PixelShuffleUpsample2d(cur_channels))

        self.out_norm = nn.GroupNorm(min(8, cur_channels), cur_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1)

        init_conv_modules(self)

    def _encode_labels(self, y: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if y is None:
            y = torch.full((batch_size,), self.null_class_idx, device=device, dtype=torch.long)
        return self.label_embedding(y)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._encode_labels(y, x.shape[0], x.device)
        if t is not None:
            te = sinusoidal_time_embedding(t, self.cond_dim)
            cond = cond + self.time_mlp(te.to(dtype=cond.dtype))
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.down_blocks):
            h = block(h, cond)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        for idx, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)
            if idx < len(self.upsamples):
                h = self.upsamples[idx](h)

        return self.out_conv(self.out_act(self.out_norm(h)))
