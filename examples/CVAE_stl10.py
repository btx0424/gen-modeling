"""
Conditional VAE on STL-10 with a bottlenecked convolutional encoder/decoder.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.images import STL10Dataset
from gen_modeling.modules.cnn import (
    Downsample2d,
    PixelShuffleUpsample2d,
    ResidualBlock2d,
    init_conv_modules,
)


@dataclass
class Config:
    data_root: str = "./data"
    split: str = "train"
    batch_size: int = 32
    base_channels: int = 64
    latent_dim: int = 256
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 30
    lr: float = 3e-4
    ema_decay: float = 0.999
    kl_beta: float = 1e-3
    num_plot_samples: int = 40


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock2d(in_channels, out_channels)
        self.down = Downsample2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.block(x))


class PixelShuffleUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels != out_channels:
            self.pre = ResidualBlock2d(in_channels, out_channels)
        else:
            self.pre = ResidualBlock2d(in_channels, in_channels)
        self.up = PixelShuffleUpsample2d(out_channels)
        self.out_block = ResidualBlock2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.up(x)
        return self.out_block(x)


class ConditionalEncoder(nn.Module):
    def __init__(self, base_channels: int, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        in_channels = STL10Dataset.info.channels + num_classes
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.mid = ResidualBlock2d(base_channels * 4, base_channels * 4)
        self.to_mu = nn.Linear(base_channels * 4, latent_dim)
        self.to_logvar = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_onehot = F.one_hot(y, num_classes=self.num_classes).to(dtype=x.dtype)
        y_map = y_onehot[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        h = self.stem(torch.cat([x, y_map], dim=1))
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        pooled = h.mean(dim=(2, 3))
        return self.to_mu(pooled), self.to_logvar(pooled)


class ConditionalDecoder(nn.Module):
    def __init__(self, base_channels: int, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.to_seed = nn.Linear(latent_dim * 2, base_channels * 4 * 12 * 12)
        self.up1 = PixelShuffleUpBlock(base_channels * 4, base_channels * 2)
        self.up2 = PixelShuffleUpBlock(base_channels * 2, base_channels)
        self.up3 = PixelShuffleUpBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, STL10Dataset.info.channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([z, self.label_embedding(y)], dim=-1)
        h = self.to_seed(cond).reshape(z.shape[0], -1, 12, 12)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return torch.tanh(self.out(h))


class CVAE(nn.Module):
    def __init__(self, base_channels: int, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = ConditionalEncoder(base_channels, latent_dim, num_classes)
        self.decoder = ConditionalDecoder(base_channels, latent_dim, num_classes)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sample_shape = (
            STL10Dataset.info.channels,
            STL10Dataset.info.height,
            STL10Dataset.info.width,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init_conv_modules(self)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, y)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, kl_beta: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon, mu, logvar = self(x, y)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
        loss = recon_loss + kl_beta * kl
        return loss, {
            "loss": loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl": kl.detach(),
        }

    @torch.inference_mode()
    def sample(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(labels.shape[0], self.latent_dim, device=device)
        return self.decode(z, labels)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


def make_eval_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_classes, device=device, dtype=torch.long)
    repeats = (num_samples + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:num_samples]


def plot_stl10_grid(
    samples: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    *,
    num_show: int,
    num_classes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = samples[:num_show].detach().cpu().clamp(-1, 1)
    samples = (samples + 1.0) * 0.5
    labels = labels[:num_show].detach().cpu()
    n_cols = num_classes
    n_rows = int(np.ceil(samples.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, (ax, image) in enumerate(zip(axes, samples, strict=False)):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(str(int(labels[idx].item())), fontsize=8, pad=1)
    for ax in axes[samples.shape[0]:]:
        ax.axis("off")
    plt.tight_layout(pad=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_and_save(
    model: CVAE,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, float]:
    labels = make_eval_labels(config.num_plot_samples, STL10Dataset.info.num_classes, device)
    samples = model.sample(labels, device)
    plot_stl10_grid(samples, labels, out_path, num_show=config.num_plot_samples, num_classes=STL10Dataset.info.num_classes)
    metrics = {
        "sample_mean": samples.mean().item(),
        "sample_std": samples.std().item(),
        "sample_min": samples.min().item(),
        "sample_max": samples.max().item(),
    }
    if metrics_path is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="STL-10 CVAE.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--split", type=str, default=Config.split)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--ema-decay", type=float, default=Config.ema_decay)
    parser.add_argument("--kl-beta", type=float, default=Config.kl_beta)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        kl_beta=args.kl_beta,
        num_plot_samples=args.num_plot_samples,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = STL10Dataset(config.data_root, split=config.split, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = CVAE(config.base_channels, config.latent_dim, STL10Dataset.info.num_classes).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "CVAE_stl10"

    for epoch in range(config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        last_metrics: dict[str, torch.Tensor] | None = None
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.compute_loss(x, y, config.kl_beta)
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)
            last_metrics = metrics
            pbar.set_postfix(
                loss=f"{metrics['loss'].item():.5f}",
                recon=f"{metrics['recon_loss'].item():.5f}",
                kl=f"{metrics['kl'].item():.5f}",
            )

        epoch_grid = out_dir / f"cvae_stl10_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"cvae_stl10_epoch_{epoch:03d}.json"
        metrics = sample_and_save(ema_model, config, device, epoch_grid, epoch_metrics)
        if last_metrics is not None:
            print(
                f"epoch {epoch}: "
                f"loss={last_metrics['loss'].item():.6f}, "
                f"recon={last_metrics['recon_loss'].item():.6f}, "
                f"kl={last_metrics['kl'].item():.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "cvae_stl10_samples.png"
    metrics_path = out_dir / "cvae_metrics.json"
    metrics = sample_and_save(ema_model, config, device, out, metrics_path)
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch STL-10 samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
