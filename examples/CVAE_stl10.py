"""
Conditional VAE on STL-10 with a bottlenecked convolutional encoder/decoder.
"""

from __future__ import annotations

import argparse
import json
import os
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

from gen_modeling.datasets.images import ImageDatasetInfo, STL10Dataset, tensor_batch_to_display
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
    batch_size: int = 64
    base_channels: int = 64
    latent_dim: int = 512
    num_threads: int = 1
    # DataLoader workers (0 = main process only). Default ~ half of CPUs, capped at 8.
    num_workers: int = min(8, max(0, (os.cpu_count() or 8) // 2))
    seed: int = 42
    train_epochs: int = 30
    lr: float = 3e-4
    kl_beta: float = 0.1
    num_plot_samples: int = 40


def _train_loader_kwargs(num_workers: int, *, pin_memory: bool) -> dict:
    """Extra DataLoader args for throughput. Tune `--num-workers` using wall time / GPU util."""
    kw: dict = {}
    if num_workers > 0:
        kw["num_workers"] = num_workers
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    if pin_memory:
        kw["pin_memory"] = True
    return kw


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
    def __init__(
        self, base_channels: int, latent_dim: int, num_classes: int, data_info: ImageDatasetInfo
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        in_channels = data_info.channels + num_classes
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.mid = ResidualBlock2d(base_channels * 4, base_channels * 4)
        # Three Downsample2d stages: spatial size is H/8 × W/8 (matches decoder seed reshape).
        h_sp = data_info.height // 8
        w_sp = data_info.width // 8
        flat_dim = base_channels * 4 * h_sp * w_sp
        self.to_parames = nn.Linear(flat_dim, latent_dim * 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_onehot = F.one_hot(y, num_classes=self.num_classes).to(dtype=x.dtype)
        y_map = y_onehot[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        h = self.stem(torch.cat([x, y_map], dim=1))
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        params = self.to_parames(h.flatten(1))
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(
        self, base_channels: int, latent_dim: int, num_classes: int, data_info: ImageDatasetInfo
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        h_sp = data_info.height // 8
        w_sp = data_info.width // 8
        self.to_seed = nn.Linear(latent_dim * 2, base_channels * 4 * h_sp * w_sp)
        self.up1 = PixelShuffleUpBlock(base_channels * 4, base_channels * 2)
        self.up2 = PixelShuffleUpBlock(base_channels * 2, base_channels)
        self.up3 = PixelShuffleUpBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, data_info.channels, kernel_size=3, padding=1)
        self._seed_h = h_sp
        self._seed_w = w_sp

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([z, self.label_embedding(y)], dim=-1)
        h = self.to_seed(cond).reshape(z.shape[0], -1, self._seed_h, self._seed_w)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return torch.tanh(self.out(h))


class CVAE(nn.Module):
    def __init__(self, base_channels: int, latent_dim: int, data_info: ImageDatasetInfo) -> None:
        super().__init__()
        nc = data_info.num_classes
        self.encoder = ConditionalEncoder(base_channels, latent_dim, nc, data_info)
        self.decoder = ConditionalDecoder(base_channels, latent_dim, nc, data_info)
        self.latent_dim = latent_dim
        self.num_classes = nc
        self.sample_shape = (data_info.channels, data_info.height, data_info.width)
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
    data_info: ImageDatasetInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = tensor_batch_to_display(samples[:num_show], data_info)
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


def plot_stl10_recon_grid(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    *,
    num_show: int,
    num_classes: int,
    data_info: ImageDatasetInfo,
) -> None:
    """Each cell: [original | reconstruction] side by side; title is the class label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs_vis = tensor_batch_to_display(inputs[:num_show], data_info)
    recons_vis = tensor_batch_to_display(recons[:num_show], data_info)
    labels = labels[:num_show].detach().cpu()
    pairs = torch.cat([inputs_vis, recons_vis], dim=-1)
    n_cols = num_classes
    n_rows = int(np.ceil(num_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 1.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, (ax, image) in enumerate(zip(axes, pairs, strict=False)):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(str(int(labels[idx].item())), fontsize=8, pad=1)
    for ax in axes[num_show:]:
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
    *,
    recon_path: Path | None = None,
    eval_x: torch.Tensor | None = None,
    eval_y: torch.Tensor | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    nc = data_info.num_classes
    labels = make_eval_labels(config.num_plot_samples, nc, device)
    samples = model.sample(labels, device)
    plot_stl10_grid(
        samples, labels, out_path, num_show=config.num_plot_samples, num_classes=nc, data_info=data_info
    )
    metrics: dict[str, float] = {
        "sample_mean": samples.mean().item(),
        "sample_std": samples.std().item(),
        "sample_min": samples.min().item(),
        "sample_max": samples.max().item(),
    }
    if recon_path is not None and eval_x is not None and eval_y is not None:
        x_e = eval_x.to(device)
        y_e = eval_y.to(device)
        mu, _ = model.encode(x_e, y_e)
        recons = model.decode(mu, y_e)
        n = min(config.num_plot_samples, x_e.shape[0])
        plot_stl10_recon_grid(
            x_e,
            recons,
            y_e,
            recon_path,
            num_show=n,
            num_classes=nc,
            data_info=data_info,
        )
        metrics["recon_mse_eval"] = F.mse_loss(recons, x_e).item()
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="DataLoader worker processes (default: from Config, ~CPU/2 capped at 8). Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--kl-beta", type=float, default=Config.kl_beta)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (debugging or environments with inductor issues).",
    )
    args = parser.parse_args()

    num_workers = Config.num_workers if args.num_workers is None else max(0, args.num_workers)
    config = Config(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_threads=args.num_threads,
        num_workers=num_workers,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        kl_beta=args.kl_beta,
        num_plot_samples=args.num_plot_samples,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = STL10Dataset(config.data_root, split=config.split, download=True, normalize=True)
    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        **_train_loader_kwargs(config.num_workers, pin_memory=pin),
    )
    # Fixed batch for reconstruction vis (same images every epoch); workers not worth it.
    vis_loader = DataLoader(dataset, batch_size=config.num_plot_samples, shuffle=False)
    eval_x, eval_y = next(iter(vis_loader))

    model = CVAE(config.base_channels, config.latent_dim, dataset.info).to(device)
    if not args.no_compile:
        model = torch.compile(model)
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
            last_metrics = metrics
            pbar.set_postfix(
                loss=f"{metrics['loss'].item():.5f}",
                recon=f"{metrics['recon_loss'].item():.5f}",
                kl=f"{metrics['kl'].item():.5f}",
            )

        epoch_grid = out_dir / f"cvae_stl10_epoch_{epoch:03d}.png"
        epoch_recon = out_dir / f"cvae_stl10_epoch_{epoch:03d}_recon.png"
        epoch_metrics = out_dir / f"cvae_stl10_epoch_{epoch:03d}.json"
        model.eval()
        metrics = sample_and_save(
            model,
            config,
            device,
            epoch_grid,
            epoch_metrics,
            recon_path=epoch_recon,
            eval_x=eval_x,
            eval_y=eval_y,
            data_info=dataset.info,
        )
        if last_metrics is not None:
            extra = (
                f", recon_mse_eval={metrics['recon_mse_eval']:.6f}"
                if "recon_mse_eval" in metrics
                else ""
            )
            print(
                f"epoch {epoch}: "
                f"loss={last_metrics['loss'].item():.6f}, "
                f"recon={last_metrics['recon_loss'].item():.6f}, "
                f"kl={last_metrics['kl'].item():.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
                f"{extra}"
            )

    out = out_dir / "cvae_stl10_samples.png"
    out_recon = out_dir / "cvae_stl10_recon.png"
    metrics_path = out_dir / "cvae_metrics.json"
    model.eval()
    metrics = sample_and_save(
        model,
        config,
        device,
        out,
        metrics_path,
        recon_path=out_recon,
        eval_x=eval_x,
        eval_y=eval_y,
        data_info=dataset.info,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch STL-10 samples, recon grids, and final metrics under examples/outputs")
    print(f"Saved plots to {out} and {out_recon}")


if __name__ == "__main__":
    main()
