"""
Equilibrium Matching on STL-10 with a conditional U-Net backbone.
"""

from __future__ import annotations

import argparse
import json
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

from gen_modeling.datasets.images import ImageDatasetInfo, STL10Dataset, tensor_batch_to_display
from gen_modeling.modules import ConditionalUNet2D


@dataclass
class Config:
    data_root: str = "./data"
    split: str = "train"
    batch_size: int = 64
    base_channels: int = 64
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    sample_steps: int = 80
    sample_stepsize: float = 0.01
    sample_sampler: Literal["gd", "nag"] = "nag"
    sample_mu: float = 0.3
    num_plot_samples: int = 40


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    def func(t: torch.Tensor) -> torch.Tensor:
        return grad_scale * torch.where(t < a, 1.0, (1.0 - t) / (1.0 - a))

    return func


class EqM(nn.Module):
    def __init__(self, network: nn.Module, sample_shape: tuple[int, int, int]):
        super().__init__()
        self.network = network
        self.sample_shape = sample_shape
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
        x = torch.randn((labels.shape[0],) + self.network.sample_shape, device=device, dtype=dtype)
        for _ in range(num_steps):
            x = x + stepsize * self.network(x, labels)
        return x

    @torch.inference_mode()
    def sample_nag(
        self,
        labels: torch.Tensor,
        device: torch.device,
        num_steps: int,
        stepsize: float,
        mu: float,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        x = torch.randn((labels.shape[0],) + self.sample_shape, device=device, dtype=dtype)
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


def sample_and_save(
    model: EqM,
    num_classes: int,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    sample_fn = model.sample_nag if config.sample_sampler == "nag" else model.sample_gd
    labels = make_eval_labels(config.num_plot_samples, num_classes, device)
    samples = sample_fn(
        labels=labels,
        device=device,
        num_steps=config.sample_steps,
        stepsize=config.sample_stepsize,
        **({"mu": config.sample_mu} if config.sample_sampler == "nag" else {}),
    )
    plot_stl10_grid(
        samples,
        labels,
        out_path,
        num_show=config.num_plot_samples,
        num_classes=num_classes,
        data_info=data_info,
    )
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
    parser = argparse.ArgumentParser(description="STL-10 EqM.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--split", type=str, default=Config.split)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
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
        split=args.split,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
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

    dataset = STL10Dataset(config.data_root, split=config.split, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = EqM(
        network=ConditionalUNet2D(
            in_channels=dataset.info.channels,
            out_channels=dataset.info.channels,
            num_classes=dataset.info.num_classes,
            base_channels=config.base_channels,
            channel_mults=(1, 2, 4, 4),
        ),
        sample_shape=(dataset.info.channels, dataset.info.height, dataset.info.width),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "EqM_stl10"

    for epoch in range(config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses = []
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        epoch_grid = out_dir / f"eqm_stl10_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"eqm_stl10_epoch_{epoch:03d}.json"
        metrics = sample_and_save(
            model,
            dataset.info.num_classes,
            config,
            device,
            epoch_grid,
            epoch_metrics,
            data_info=dataset.info,
        )
        if losses:
            print(
                f"epoch {epoch}: "
                f"loss={np.mean(losses):.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "eqm_stl10_samples.png"
    metrics_path = out_dir / "eqm_metrics.json"
    metrics = sample_and_save(
        model,
        dataset.info.num_classes,
        config,
        device,
        out,
        metrics_path,
        data_info=dataset.info,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch STL-10 samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
