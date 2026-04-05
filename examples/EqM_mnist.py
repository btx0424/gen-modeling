"""
Minimal Equilibrium Matching on MNIST.

This file is for image training only. Synthetic toy training lives in `EqM.py`.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.images import ImageDatasetInfo, MNISTDataset
from gen_modeling.modules import SmallConvNet


@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 128
    hidden_channels: int = 64
    num_blocks: int = 4
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 10
    lr: float = 5e-4
    ema_decay: float = 0.999
    sample_steps: int = 100
    sample_stepsize: float = 0.01
    sample_sampler: Literal["gd", "nag"] = "nag"
    sample_mu: float = 0.3
    num_plot_samples: int = 100


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    def func(t: torch.Tensor) -> torch.Tensor:
        return grad_scale * torch.where(t < a, 1.0, (1.0 - t) / (1.0 - a))
    return func


class ImageEqMBackbone(nn.Module):
    def __init__(self, hidden_channels: int = 64, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        self.in_channels = 1
        self.sample_shape = (1, 28, 28)
        self.num_classes = num_classes
        self.backbone = SmallConvNet(1 + num_classes, hidden_channels, hidden_channels, num_blocks)
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(dtype=x.dtype)
        y_map = y_onehot[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.out(self.backbone(torch.cat([x, y_map], dim=1)))


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


class EqM(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.grad_magnitude = eqm_ct()

    def compute_loss(self, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype).reshape(expand_shape)
        x0 = torch.randn_like(x1)
        xt = t * x1 + (1.0 - t) * x0
        target = (x1 - x0) * self.grad_magnitude(t)
        pred = self.network(xt, y)
        return ((pred - target) ** 2).mean()

    @torch.inference_mode()
    def sample_gd(self, labels: torch.Tensor, device: torch.device, num_steps: int, stepsize: float) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        num_samples = labels.shape[0]
        x = torch.randn((num_samples,) + self.network.sample_shape, device=device, dtype=dtype)
        for _ in range(num_steps):
            x = x + stepsize * self.network(x, labels)
        return x

    @torch.inference_mode()
    def sample_nag(self, labels: torch.Tensor, device: torch.device, num_steps: int, stepsize: float, mu: float) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        num_samples = labels.shape[0]
        x = torch.randn((num_samples,) + self.network.sample_shape, device=device, dtype=dtype)
        momentum = torch.zeros_like(x)
        for _ in range(num_steps):
            lookahead = x + stepsize * mu * momentum
            momentum = self.network(lookahead, labels)
            x = x + stepsize * momentum
        return x


def make_eval_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_classes, device=device, dtype=torch.long)
    repeats = (num_samples + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:num_samples]


def plot_mnist_grid(
    samples: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    *,
    num_show: int = 100,
    num_classes: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = samples[:num_show].detach().cpu().clamp(-1, 1)
    labels = labels[:num_show].detach().cpu()
    n_cols = num_classes
    n_rows = int(np.ceil(samples.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, (ax, image) in enumerate(zip(axes, samples, strict=False)):
        ax.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(str(int(labels[idx].item())), fontsize=8, pad=1)
    for ax in axes[samples.shape[0]:]:
        ax.axis("off")
    plt.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_and_save(
    model: EqM,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    sample_fn = model.sample_nag if config.sample_sampler == "nag" else model.sample_gd
    num_classes = data_info.num_classes
    labels = make_eval_labels(config.num_plot_samples, num_classes, device)
    samples = sample_fn(
        labels=labels,
        device=device,
        num_steps=config.sample_steps,
        stepsize=config.sample_stepsize,
        **({"mu": config.sample_mu} if config.sample_sampler == "nag" else {}),
    )
    plot_mnist_grid(samples, labels, out_path, num_show=config.num_plot_samples, num_classes=num_classes)
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
    parser = argparse.ArgumentParser(description="MNIST EqM.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-channels", type=int, default=Config.hidden_channels)
    parser.add_argument("--num-blocks", type=int, default=Config.num_blocks)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--sample-stepsize", type=float, default=Config.sample_stepsize)
    parser.add_argument("--sample-sampler", choices=["gd", "nag"], default=Config.sample_sampler)
    parser.add_argument("--sample-mu", type=float, default=Config.sample_mu)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        sample_steps=args.sample_steps,
        sample_stepsize=args.sample_stepsize,
        sample_sampler=args.sample_sampler,
        sample_mu=args.sample_mu,
        num_plot_samples=args.num_plot_samples,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MNISTDataset(config.data_root, train=True, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = EqM(
        ImageEqMBackbone(config.hidden_channels, config.num_blocks, dataset.info.num_classes)
    ).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "EqM_mnist"

    for epoch in range(config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        last_loss = None
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(x, y)
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)
            last_loss = loss
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        epoch_grid = out_dir / f"eqm_mnist_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"eqm_mnist_epoch_{epoch:03d}.json"
        metrics = sample_and_save(
            ema_model, config, device, epoch_grid, epoch_metrics, data_info=dataset.info
        )
        if last_loss is not None:
            print(
                f"epoch {epoch}: "
                f"loss={last_loss.item():.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "eqm_mnist_samples.png"
    metrics_path = out_dir / "eqm_metrics.json"
    metrics = sample_and_save(ema_model, config, device, out, metrics_path, data_info=dataset.info)
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch MNIST samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
