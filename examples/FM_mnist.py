"""
Minimal Flow Matching on MNIST.

This file is for image training only. Synthetic toy training lives in `main.py`
and `flow_matching.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.images import MNISTDataset
from gen_modeling.flow_matching import (
    LinearFlow,
    ModelArch,
    PredictionType,
    LossType,
    prediction_wrapper_class,
)
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
    noise_scale: float = 1.0
    t_eps: float = 1e-2
    sample_steps: int = 100
    num_plot_samples: int = 64
    model_arch: ModelArch = "vanilla"
    pred_type: PredictionType = "v"
    loss_type: LossType = "v"


class ImageDenoisingCNN(nn.Module):
    def __init__(self, hidden_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.sample_shape = (1, 28, 28)
        self.backbone = SmallConvNet(2, hidden_channels, hidden_channels, num_blocks)
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = y
        t_map = t.reshape(-1, 1, 1, 1).expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
        features = self.backbone(torch.cat([x_t, t_map], dim=1))
        return self.out(features)


def build_model(config: Config) -> nn.Module:
    base_network = ImageDenoisingCNN(config.hidden_channels, config.num_blocks)
    wrapper_cls = prediction_wrapper_class(config.model_arch)
    return wrapper_cls(base_network, config.pred_type)


def plot_mnist_grid(samples: torch.Tensor, path: Path, *, num_show: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = samples[:num_show].detach().cpu().clamp(-1, 1)
    n = int(np.ceil(np.sqrt(samples.shape[0])))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    axes = np.asarray(axes).reshape(-1)
    for ax, image in zip(axes, samples, strict=False):
        ax.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    for ax in axes[samples.shape[0] :]:
        ax.axis("off")
    plt.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_and_save(
    flow: LinearFlow,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, float]:
    samples = flow.sample(config.num_plot_samples, device, config.sample_steps)
    plot_mnist_grid(samples, out_path)
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
    parser = argparse.ArgumentParser(description="MNIST Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-channels", type=int, default=Config.hidden_channels)
    parser.add_argument("--num-blocks", type=int, default=Config.num_blocks)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    parser.add_argument("--t-eps", type=float, default=Config.t_eps)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument(
        "--model-arch",
        choices=["vanilla", "global_residual", "corrected_residual1", "corrected_residual2"],
        default=Config.model_arch,
    )
    parser.add_argument("--pred-type", choices=["x", "eps", "v"], default=Config.pred_type)
    parser.add_argument("--loss-type", choices=["x", "eps", "v"], default=Config.loss_type)
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
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        sample_steps=args.sample_steps,
        num_plot_samples=args.num_plot_samples,
        model_arch=args.model_arch,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MNISTDataset(config.data_root, train=True, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = build_model(config).to(device)
    flow = LinearFlow(
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
        conditional=False,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "FM_mnist"

    for epoch in range(config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses = []
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow.compute_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        epoch_grid = out_dir / f"fm_mnist_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"fm_mnist_epoch_{epoch:03d}.json"
        model.eval()
        metrics = sample_and_save(flow, config, device, epoch_grid, epoch_metrics)
        if losses:
            print(
                f"epoch {epoch}: "
                f"loss={np.mean(losses):.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "fm_mnist_samples.png"
    metrics_path = out_dir / "fm_mnist_metrics.json"
    model.eval()
    metrics = sample_and_save(flow, config, device, out, metrics_path)
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch MNIST samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
