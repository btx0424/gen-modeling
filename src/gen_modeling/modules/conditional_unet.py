from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        groups_in = min(8, in_channels)
        groups_out = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet2D(nn.Module):
    """
    Small class-conditional U-Net backbone for image generation experiments.
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
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        widths = [base_channels * mult for mult in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_channels = base_channels
        for idx, width in enumerate(widths):
            self.down_blocks.append(ConditionalResBlock(cur_channels, width, cond_dim))
            cur_channels = width
            if idx < len(widths) - 1:
                self.downsamples.append(Downsample(cur_channels))

        self.mid_block1 = ConditionalResBlock(cur_channels, cur_channels, cond_dim)
        self.mid_block2 = ConditionalResBlock(cur_channels, cur_channels, cond_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(len(widths) - 1, -1, -1):
            skip_channels = widths[idx]
            self.up_blocks.append(ConditionalResBlock(cur_channels + skip_channels, skip_channels, cond_dim))
            cur_channels = skip_channels
            if idx > 0:
                self.upsamples.append(Upsample(cur_channels))

        self.out_norm = nn.GroupNorm(min(8, cur_channels), cur_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.label_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_labels(self, y: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if y is None:
            y = torch.full((batch_size,), self.null_class_idx, device=device, dtype=torch.long)
        return self.label_embedding(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        cond = self._encode_labels(y, x.shape[0], x.device)
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
