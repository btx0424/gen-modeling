"""
Equilibrium Matching on CelebA with an unconditional U-Net backbone.
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

from gen_modeling.datasets.images import CelebADataset, ImageDatasetInfo, tensor_batch_to_display
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

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype).reshape(expand_shape)
        x0 = torch.randn_like(x1)
        xt = t * x1 + (1.0 - t) * x0
        target = (x1 - x0) * self.grad_magnitude(t)
        pred = self.network(xt, cond=None)
        return ((pred - target) ** 2).mean()

    @torch.inference_mode()
    def sample_gd(self, num_samples: int, device: torch.device, num_steps: int, stepsize: float) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        x = torch.randn((num_samples,) + self.network.sample_shape, device=device, dtype=dtype)
        for _ in range(num_steps):
            x = x + stepsize * self.network(x, cond=None)
        return x

    @torch.inference_mode()
    def sample_nag(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int,
        stepsize: float,
        mu: float,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        x = torch.randn((num_samples,) + self.sample_shape, device=device, dtype=dtype)
        momentum = torch.zeros_like(x)
        for _ in range(num_steps):
            lookahead = x + stepsize * mu * momentum
            momentum = self.network(lookahead, cond=None)
            x = x + stepsize * momentum
        return x


def plot_image_grid(
    samples: torch.Tensor,
    path: Path,
    *,
    num_show: int,
    data_info: ImageDatasetInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = tensor_batch_to_display(samples[:num_show], data_info)
    n_cols = int(np.ceil(np.sqrt(samples.shape[0])))
    n_rows = int(np.ceil(samples.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, image in zip(axes, samples, strict=False):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.axis("off")
    for ax in axes[samples.shape[0]:]:
        ax.axis("off")
    plt.tight_layout(pad=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "training_checkpoint.pt"


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def try_resume_training(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> int:
    """Load checkpoint if present. Returns epoch index to start training from (0 = fresh)."""
    if not path.is_file():
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "torch_rng" in ckpt:
        tr = ckpt["torch_rng"]
        torch.set_rng_state(tr.cpu() if tr.device.type != "cpu" else tr)
    if "numpy_rng" in ckpt:
        np.random.set_state(ckpt["numpy_rng"])
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return int(ckpt["epoch"]) + 1


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
    samples = sample_fn(
        num_samples=config.num_plot_samples,
        device=device,
        num_steps=config.sample_steps,
        stepsize=config.sample_stepsize,
        **({"mu": config.sample_mu} if config.sample_sampler == "nag" else {}),
    )
    plot_image_grid(
        samples,
        out_path,
        num_show=config.num_plot_samples,
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
    parser = argparse.ArgumentParser(description="CelebA EqM.")
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load training_checkpoint.pt from the output directory even if it exists.",
    )
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

    dataset = CelebADataset(config.data_root, split=config.split, download=True, normalize=True)
    loader = DataLoader(dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)

    model = EqM(
        network=ConditionalUNet2D(
            in_channels=dataset.info.channels,
            out_channels=dataset.info.channels,
            num_classes=1,
            base_channels=config.base_channels,
            channel_mults=(1, 2, 4, 4),
        ),
        sample_shape=(dataset.info.channels, dataset.info.height, dataset.info.width),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "EqM_celeba"
    ckpt_path = _checkpoint_path(out_dir)
    start_epoch = 0
    if not args.no_resume:
        start_epoch = try_resume_training(
            ckpt_path, model=model, optimizer=optimizer, device=device
        )
        if start_epoch > 0:
            print(f"Resumed from {ckpt_path}; starting at epoch {start_epoch}")

    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses = []
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        epoch_grid = out_dir / f"eqm_celeba_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"eqm_celeba_epoch_{epoch:03d}.json"
        metrics = sample_and_save(
            model,
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
        save_training_checkpoint(
            ckpt_path, epoch=epoch, model=model, optimizer=optimizer
        )

    out = out_dir / "eqm_celeba_samples.png"
    metrics_path = out_dir / "eqm_celeba_metrics.json"
    metrics = sample_and_save(
        model,
        config,
        device,
        out,
        metrics_path,
        data_info=dataset.info,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch CelebA EqM samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
