"""
Equilibrium Matching on LAFAN1-style robot trajectories.

This is a scaffold example using the shared 1D conditional U-Net over sequences.
Inputs and outputs have shape (B, T, C), matching ``LAFAN1Dataset`` windows.
Conditioning is done by pinning the first ``k`` timesteps (``--cond-steps``); the model
is trained to match the flow field on later steps while keeping the prefix fixed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from jaxtyping import Float

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.robotics import LAFAN1Dataset, RobotName
from gen_modeling.modules import ConditionalUNet1D


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    seq_len: int = 32
    cond_steps: int = 4
    stride: int = 1
    batch_size: int = 128
    base_channels: int = 128
    cond_dim: int = 256
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    sample_steps: int = 80
    sample_stepsize: float = 0.01
    sample_sampler: Literal["gd", "nag"] = "nag"
    sample_mu: float = 0.3
    num_plot_samples: int = 16
    num_plot_dims: int = 6
    download: bool = True


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    def func(t: torch.Tensor) -> torch.Tensor:
        return grad_scale * torch.where(t < a, 1.0, (1.0 - t) / (1.0 - a))

    return func


class TrajectoryEqMBackbone(nn.Module):
    def __init__(self, input_dim: int, base_channels: int, cond_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.unet = ConditionalUNet1D(
            input_dim=input_dim,
            output_dim=input_dim,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
            cond_dim=cond_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = torch.zeros(x.shape[0], self.cond_dim, device=x.device, dtype=x.dtype)
        return self.unet(x, cond, t=None)


class EqM(nn.Module):
    def __init__(self, network: nn.Module, *, cond_steps: int = 4):
        super().__init__()
        if cond_steps < 1:
            raise ValueError("cond_steps must be >= 1")
        self.network = network
        self.cond_steps = cond_steps
        self.grad_magnitude = eqm_ct()

    def compute_loss(self, x1: Float[torch.Tensor, "N seq_len D"]) -> torch.Tensor:
        k = self.cond_steps
        if x1.shape[1] < k:
            raise ValueError(
                f"sequence length {x1.shape[1]} is shorter than cond_steps={k}"
            )
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype).reshape(expand_shape)
        x0 = torch.randn_like(x1)
        xt = t * x1 + (1.0 - t) * x0
        prefix = x1[:, :k]
        xt[:, :k] = prefix
        target = (x1 - x0) * self.grad_magnitude(t)
        target[:, :k] = 0.0
        pred = self.network(xt)
        pred[:, :k] = 0.0
        return ((pred - target) ** 2).mean()

    @torch.inference_mode()
    def sample_gd(
        self,
        cond_prefix: torch.Tensor,
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int,
        stepsize: float,
    ) -> torch.Tensor:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        num_samples = cond_prefix.shape[0]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x = torch.randn(num_samples, seq_len, self.network.input_dim, device=device, dtype=dtype)
        x[:, :k] = cond_prefix
        for _ in range(num_steps):
            update = self.network(x)
            update[:, :k] = 0.0
            x = x + stepsize * update
            x[:, :k] = cond_prefix
        return x

    @torch.inference_mode()
    def sample_nag(
        self,
        cond_prefix: torch.Tensor,
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int,
        stepsize: float,
        mu: float,
    ) -> torch.Tensor:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        num_samples = cond_prefix.shape[0]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x = torch.randn(num_samples, seq_len, self.network.input_dim, device=device, dtype=dtype)
        x[:, :k] = cond_prefix
        momentum = torch.zeros_like(x)
        for _ in range(num_steps):
            lookahead = x + stepsize * mu * momentum
            lookahead[:, :k] = cond_prefix
            momentum = self.network(lookahead)
            momentum[:, :k] = 0.0
            x = x + stepsize * momentum
            x[:, :k] = cond_prefix
        return x


def plot_trajectory_grid(
    data_seq: torch.Tensor,
    sample_seq: torch.Tensor,
    path: Path,
    *,
    num_dims: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_seq = data_seq.detach().cpu()
    sample_seq = sample_seq.detach().cpu()
    num_dims = min(num_dims, data_seq.shape[-1], sample_seq.shape[-1])
    fig, axes = plt.subplots(num_dims, 2, figsize=(10, 1.8 * num_dims), sharex=True)
    axes = np.atleast_2d(axes)
    time = np.arange(data_seq.shape[0])
    for dim in range(num_dims):
        ax_l = axes[dim, 0]
        ax_r = axes[dim, 1]
        ax_l.plot(time, data_seq[:, dim].numpy(), lw=1.2)
        ax_r.plot(time, sample_seq[:, dim].numpy(), lw=1.2, color="C1")
        ax_l.set_ylabel(f"q[{dim}]")
        if dim == 0:
            ax_l.set_title("Dataset window")
            ax_r.set_title("EqM sample")
    axes[-1, 0].set_xlabel("timestep")
    axes[-1, 1].set_xlabel("timestep")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_and_save(
    model: EqM,
    config: Config,
    device: torch.device,
    reference_batch: torch.Tensor,
    dataset: LAFAN1Dataset,
    out_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, float]:
    sample_fn = model.sample_nag if config.sample_sampler == "nag" else model.sample_gd
    k = config.cond_steps
    cond_prefix = reference_batch[: config.num_plot_samples, :k].to(device)
    samples = sample_fn(
        cond_prefix=cond_prefix,
        seq_len=config.seq_len,
        device=device,
        num_steps=config.sample_steps,
        stepsize=config.sample_stepsize,
        **({"mu": config.sample_mu} if config.sample_sampler == "nag" else {}),
    )
    samples_denorm = dataset.denormalize(samples.detach().cpu())
    ref_row_denorm = dataset.denormalize(reference_batch[0].detach().cpu())
    plot_trajectory_grid(
        ref_row_denorm,
        samples_denorm[0],
        out_path,
        num_dims=config.num_plot_dims,
    )
    metrics = {
        "sample_mean": samples_denorm.mean().item(),
        "sample_std": samples_denorm.std().item(),
        "sample_min": samples_denorm.min().item(),
        "sample_max": samples_denorm.max().item(),
    }
    if metrics_path is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Config,
    reference_batch: torch.Tensor | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "reference_batch": reference_batch,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[int, torch.Tensor | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].contiguous().cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    epoch = int(payload["epoch"])
    ref = payload.get("reference_batch")
    if ref is not None:
        ref = ref.detach().cpu()
    return epoch, ref


@torch.no_grad()
def _first_normalized_batch(
    loader: DataLoader,
    dataset: LAFAN1Dataset,
    device: torch.device,
) -> torch.Tensor:
    x, _ = next(iter(loader))
    x = dataset.make_relative(x.to(device))
    x = dataset.normalize(x)
    return x.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 EqM scaffold.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument(
        "--cond-steps",
        type=int,
        default=Config.cond_steps,
        help="Pin the first k timesteps of every window for training and sampling.",
    )
    parser.add_argument("--stride", type=int, default=Config.stride)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--sample-stepsize", type=float, default=Config.sample_stepsize)
    parser.add_argument("--sample-sampler", choices=["gd", "nag"], default=Config.sample_sampler)
    parser.add_argument("--sample-mu", type=float, default=Config.sample_mu)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--num-plot-dims", type=int, default=Config.num_plot_dims)
    parser.add_argument("--no-download", action="store_true", help="Require local CSV clips; do not fetch from HF.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path (default: examples/outputs/EqM_lafan1/checkpoint.pt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load weights, optimizer, RNG, and reference batch from --checkpoint and continue.",
    )
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        robot=args.robot,
        seq_len=args.seq_len,
        cond_steps=args.cond_steps,
        stride=args.stride,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        sample_steps=args.sample_steps,
        sample_stepsize=args.sample_stepsize,
        sample_sampler=args.sample_sampler,
        sample_mu=args.sample_mu,
        num_plot_samples=args.num_plot_samples,
        num_plot_dims=args.num_plot_dims,
        download=not args.no_download,
    )
    if not (1 <= config.cond_steps <= config.seq_len):
        raise ValueError(f"Need 1 <= cond-steps <= seq-len; got cond_steps={config.cond_steps}, seq_len={config.seq_len}")

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.seq_len,
        stride=config.stride,
        download=config.download,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    state_dim = dataset.state_dim

    model = EqM(
        network=TrajectoryEqMBackbone(
            input_dim=state_dim,
            base_channels=config.base_channels,
            cond_dim=config.cond_dim,
        ),
        cond_steps=config.cond_steps,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "EqM_lafan1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    reference_batch: torch.Tensor | None = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        start_epoch, reference_batch = _load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []
        for x, _meta in pbar:
            x = dataset.make_relative(x.to(device))
            x = dataset.normalize(x)
            if reference_batch is None:
                reference_batch = x.detach().cpu()
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        epoch_plot = out_dir / f"eqm_lafan1_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"eqm_lafan1_epoch_{epoch:03d}.json"
        if reference_batch is None:
            reference_batch = _first_normalized_batch(loader, dataset, device)
        metrics = sample_and_save(
            model,
            config,
            device,
            reference_batch,
            dataset,
            epoch_plot,
            epoch_metrics,
        )
        _save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            reference_batch=reference_batch,
        )
        if losses:
            print(
                f"epoch {epoch}: "
                f"loss={np.mean(losses):.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "eqm_lafan1_samples.png"
    metrics_path = out_dir / "eqm_metrics.json"
    ref_final = (
        reference_batch
        if reference_batch is not None
        else _first_normalized_batch(loader, dataset, device)
    )
    metrics = sample_and_save(
        model,
        config,
        device,
        ref_final,
        dataset,
        out,
        metrics_path,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch LAFAN1 samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
