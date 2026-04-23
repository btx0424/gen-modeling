"""
Flow Matching on CelebA with an unconditional U-Net backbone.
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

from gen_modeling.datasets.images import CelebADataset, ImageDatasetInfo, tensor_batch_to_display
from gen_modeling.flow_matching import (
    LinearFlow,
    LossType,
    ModelArch,
    PredictionType,
    prediction_wrapper,
)
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
    noise_scale: float = 1.0
    t_eps: float = 1e-2
    sample_steps: int = 80
    num_plot_samples: int = 40
    model_arch: ModelArch = "vanilla"
    pred_type: PredictionType = "v"
    loss_type: LossType = "v"


class ImageFlowUNet(nn.Module):
    def __init__(self, base_channels: int, data_info: ImageDatasetInfo):
        super().__init__()
        self.sample_shape = (data_info.channels, data_info.height, data_info.width)
        self.backbone = ConditionalUNet2D(
            in_channels=data_info.channels,
            out_channels=data_info.channels,
            num_classes=1,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.backbone(x_t, t=t, cond=cond)


def build_model(config: Config, data_info: ImageDatasetInfo) -> nn.Module:
    base_network = ImageFlowUNet(config.base_channels, data_info)
    return prediction_wrapper(base_network, config.pred_type, config.model_arch)


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
    for ax in axes[samples.shape[0] :]:
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
    flow: LinearFlow,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    samples = flow.sample(config.num_plot_samples, device, config.sample_steps)
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
    parser = argparse.ArgumentParser(description="CelebA Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--split", type=str, default=Config.split)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
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
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (debugging or environments with inductor issues).",
    )
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

    dataset = CelebADataset(config.data_root, split=config.split, download=True, normalize=True)
    loader = DataLoader(dataset, num_workers=2, batch_size=config.batch_size, shuffle=True)
    data_info = dataset.info

    model = build_model(config, data_info).to(device)
    if not args.no_compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    flow = LinearFlow(
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
        conditional=False,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "FM_celeba"
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
        last_loss = None
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow.compute_loss(x)
            loss.backward()
            optimizer.step()
            last_loss = loss
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        epoch_grid = out_dir / f"fm_celeba_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"fm_celeba_epoch_{epoch:03d}.json"
        model.eval()
        metrics = sample_and_save(
            flow, config, device, epoch_grid, epoch_metrics, data_info=data_info
        )
        if last_loss is not None:
            print(
                f"epoch {epoch}: "
                f"loss={last_loss.item():.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )
        save_training_checkpoint(
            ckpt_path, epoch=epoch, model=model, optimizer=optimizer
        )

    out = out_dir / "fm_celeba_samples.png"
    metrics_path = out_dir / "fm_celeba_metrics.json"
    model.eval()
    metrics = sample_and_save(flow, config, device, out, metrics_path, data_info=data_info)
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch CelebA FM samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
