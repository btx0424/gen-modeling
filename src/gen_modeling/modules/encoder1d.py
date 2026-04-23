"""1D convolutional encoders for trajectory segments ``(B, T, D)``."""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_conv1d_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv1d, nn.Linear)):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class Encoder1D(nn.Module):
    """
    Encodes a trajectory window with Conv1d along time.

    Expects ``x`` of shape ``(batch, seq_len, state_dim)`` (same layout as LAFAN1 batches).
    Internally uses layout ``(batch, state_dim, seq_len)`` so filters sweep the temporal axis.
    Returns Gaussian parameters ``mu``, ``logvar`` each of shape ``(batch, latent_dim)``.

    After the stem, each of ``num_downsample`` stages applies a **stride-2** temporal downsample
    then a **size-preserving** ``3×1`` Conv block with a residual add. If ``num_downsample`` is
    0, only the stem feeds the global pool and Gaussian head.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        latent_dim: int,
        hidden_channels: int = 128,
        stem_kernel: int = 7,
        num_downsample: int = 2,
    ) -> None:
        super().__init__()
        if stem_kernel % 2 == 0:
            raise ValueError("stem_kernel should be odd for same-length padding.")
        if num_downsample < 0:
            raise ValueError("num_downsample must be >= 0.")
        pad = stem_kernel // 2

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.hidden_channels = hidden_channels

        def preserving_block(in_ch: int):
            return nn.Sequential(
                nn.Conv1d(in_ch, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(2, hidden_channels),
                nn.SiLU(),
            )

        def downsample_block():
            return nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(2, hidden_channels),
                nn.SiLU(),
            )

        self.stem = nn.Sequential(
            nn.Conv1d(state_dim, hidden_channels, kernel_size=stem_kernel, padding=pad),
            nn.GroupNorm(2, hidden_channels),
            nn.SiLU(),
        )

        self.conv_blocks = nn.ModuleList([
            preserving_block(hidden_channels)
            for _ in range(num_downsample)
        ])
        self.down_blocks = nn.ModuleList([
            downsample_block()
            for _ in range(num_downsample)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_gaussian = nn.Linear(hidden_channels, 2 * latent_dim)
        _init_conv1d_linear(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x (B, T, D); got shape {tuple(x.shape)}")
        # (B, T, D) -> (B, D, T)
        h = x.transpose(1, 2)
        h = self.stem(h)
        for conv_block, down_block in zip(self.conv_blocks, self.down_blocks):
            h = down_block(h)
            h = conv_block(h) + h
        h = self.pool(h).flatten(1)
        params = self.to_gaussian(h)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar
