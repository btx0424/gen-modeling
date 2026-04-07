"""
Equilibrium Matching on LAFAN1-style robot trajectories.

This is a scaffold example using the shared 1D conditional U-Net over sequences.
Inputs and outputs have shape (B, T, C), matching ``LAFAN1Dataset`` windows.
Conditioning is done by pinning the first ``k`` timesteps (``--cond-steps``); the model
is trained to match the flow field on later steps while keeping the prefix fixed.
Each epoch, sliding-window rollouts are denormalized and saved as CSV under ``outputs/.../validation/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.robotics import LAFAN1Dataset, POSE_BASE_DIM, RobotName
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
    val_total_len: int = 128
    val_window_stride: int | None = None
    download: bool = True


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    def func(t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
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

    def forward(self, x: Float[Tensor, "batch time dim"]) -> Float[Tensor, "batch time dim"]:
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

    def compute_loss(self, x1: Float[Tensor, "batch seq dim"]) -> Float[Tensor, ""]:
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
        cond_prefix: Float[Tensor, "batch cond dim"],
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int,
        stepsize: float,
    ) -> Float[Tensor, "batch seq dim"]:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        num_samples = cond_prefix.shape[0]
        feat_dim = cond_prefix.shape[-1]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x = torch.randn(num_samples, seq_len, feat_dim, device=device, dtype=dtype)
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
        cond_prefix: Float[Tensor, "batch cond dim"],
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int,
        stepsize: float,
        mu: float,
    ) -> Float[Tensor, "batch seq dim"]:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        num_samples = cond_prefix.shape[0]
        feat_dim = cond_prefix.shape[-1]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x = torch.randn(num_samples, seq_len, feat_dim, device=device, dtype=dtype)
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


def _resolve_val_window_stride(config: Config) -> int:
    k = config.cond_steps
    max_stride = config.seq_len - k
    if max_stride < 1:
        raise ValueError(
            f"Need seq_len ({config.seq_len}) > cond_steps ({k}) for sliding-window validation."
        )
    if config.val_window_stride is None:
        return max_stride
    if config.val_window_stride < 1 or config.val_window_stride > max_stride:
        raise ValueError(
            f"val_window_stride must be in [1, {max_stride}], got {config.val_window_stride}"
        )
    return config.val_window_stride


@torch.no_grad()
def save_validation_rollouts_csv(
    model: EqM,
    config: Config,
    device: torch.device,
    reference_batch: Float[Tensor, "batch seq dim"],
    dataset: LAFAN1Dataset,
    out_dir: Path,
    epoch: int,
) -> dict[str, float | int | str]:
    """
    Sliding-window rollouts in physical trajectory space, then write one CSV per sample
    in retargeting **qpos** layout (``xyzw`` quaternion + ``jpos``; no ``jvel``).
    """
    model.eval()
    k = config.cond_steps
    n = min(config.num_plot_samples, reference_batch.shape[0])
    cond_prefix = reference_batch[:n, :k].to(device)
    stride = _resolve_val_window_stride(config)
    total_len = config.val_total_len
    traj = sliding_window_generate(
        model,
        dataset,
        config,
        device,
        cond_prefix,
        total_len,
        stride,
    )
    traj_denorm = traj.detach().cpu()

    val_dir = out_dir / "validation" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        csv_qpos = dataset.trajectory_to_lafan1_csv_qpos(traj_denorm[i])
        arr = csv_qpos.numpy()
        np.savetxt(val_dir / f"rollout_{i:03d}.csv", arr, delimiter=",", fmt="%.8f")

    meta = {
        "epoch": epoch,
        "val_total_len": total_len,
        "val_window_stride": stride,
        "num_rollouts": n,
        "csv_dir": str(val_dir),
        "sample_mean": float(traj_denorm.mean().item()),
        "sample_std": float(traj_denorm.std().item()),
        "sample_min": float(traj_denorm.min().item()),
        "sample_max": float(traj_denorm.max().item()),
    }
    metrics_path = out_dir / f"eqm_lafan1_epoch_{epoch:03d}.json"
    metrics_path.write_text(json.dumps(meta, indent=2))
    return meta


@torch.no_grad()
def sliding_window_generate(
    model: EqM,
    dataset: LAFAN1Dataset,
    config: Config,
    device: torch.device,
    cond_prefix: Float[Tensor, "batch cond dim"],
    total_len: int,
    window_stride: int,
) -> Float[Tensor, "batch total dim"]:
    """
    Generate a trajectory longer than ``seq_len`` by repeatedly sampling a length-``seq_len``
    window and advancing the window start by ``window_stride`` frames.

    The stitched ``traj`` is stored in physical trajectory space (unnormalized). Normalization
    is applied only at the model boundary: right before calling ``sample_*`` with
    ``cond_prefix``, and right after sampling to map generated chunks back to physical space.

    Require ``window_stride <= seq_len - cond_steps`` so the next prefix always lies in
    frames already filled by the previous chunk (standard non-gap overlap condition).

    Parameters
    ----------
    cond_prefix
        Shape ``(N, cond_steps, D)`` in physical trajectory space (unnormalized).
    total_len
        Target number of timesteps ``T`` (must be at least ``cond_steps``).
    window_stride
        Increment of the window start ``ws`` after each sample (smaller => more overlap).
    """
    k = model.cond_steps
    seq_len = config.seq_len
    if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
        raise ValueError(
            f"cond_prefix must be (N, {k}, D), got {tuple(cond_prefix.shape)}"
        )
    if total_len < k:
        raise ValueError(f"total_len ({total_len}) must be >= cond_steps ({k})")
    if window_stride < 1:
        raise ValueError("window_stride must be >= 1")
    max_stride = seq_len - k
    if max_stride < 1:
        raise ValueError(
            f"Need seq_len ({seq_len}) > cond_steps ({k}) to slide the window; "
            "got no room to generate beyond the pinned prefix."
        )
    if window_stride > max_stride:
        raise ValueError(
            f"window_stride ({window_stride}) must be <= seq_len - cond_steps ({max_stride}) "
            "so the next prefix stays inside the previous generated segment."
        )

    sample_fn = model.sample_nag if config.sample_sampler == "nag" else model.sample_gd
    kwargs = (
        {"mu": config.sample_mu}
        if config.sample_sampler == "nag"
        else {}
    )
    dtype = next(model.parameters()).dtype
    n, _, dim = cond_prefix.shape
    traj = torch.zeros(n, total_len, dim, device=device, dtype=dtype)
    cond0 = dataset.make_relative(cond_prefix.to(device=device, dtype=dtype))
    traj[:, :k] = cond0

    ws = 0
    while True:
        if ws + k > total_len:
            break
        # traj and cond0 stay in physical space; only model inputs/outputs are normalized.
        traj_ws = traj[:, ws : ws + k]
        cond_local = dataset.normalize(dataset.make_relative(traj_ws))
        root_pos_ref = traj[:, ws, :3]
        root_rot6d_ref = traj[:, ws, 3:POSE_BASE_DIM]

        chunk = sample_fn(
            cond_prefix=cond_local,
            seq_len=seq_len,
            device=device,
            num_steps=config.sample_steps,
            stepsize=config.sample_stepsize,
            **kwargs,
        )
        n_write = min(seq_len, total_len - ws)
        chunk_phys = dataset.denormalize(chunk[:, :n_write])
        chunk_merged_phys = dataset.accumulate_chunk_in_root_frame(
            chunk_phys,
            root_pos_ref,
            root_rot6d_ref,
        )
        traj[:, ws : ws + n_write] = chunk_merged_phys
        if ws + n_write >= total_len:
            break
        ws += window_stride
        if ws >= total_len:
            break
    return traj


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].contiguous().cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    return int(payload["epoch"])


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
    parser.add_argument(
        "--val-total-len",
        type=int,
        default=Config.val_total_len,
        help="Sliding-window validation rollout length (timesteps); CSV rows per rollout.",
    )
    parser.add_argument(
        "--val-window-stride",
        type=int,
        default=None,
        help="Stride between windows in validation (default: seq_len - cond_steps).",
    )
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
        help="Load weights, optimizer, and RNG from --checkpoint and continue.",
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
        val_total_len=args.val_total_len,
        val_window_stride=args.val_window_stride,
        download=not args.no_download,
    )
    if not (1 <= config.cond_steps <= config.seq_len):
        raise ValueError(f"Need 1 <= cond-steps <= seq-len; got cond_steps={config.cond_steps}, seq_len={config.seq_len}")
    if config.val_total_len < config.cond_steps:
        raise ValueError(
            f"val_total_len ({config.val_total_len}) must be >= cond_steps ({config.cond_steps})"
        )

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
        start_epoch = _load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []
        reference_batch = None

        for batch, _meta in pbar:
            if reference_batch is None:
                reference_batch = batch.cpu()
            batch_norm = dataset.normalize(dataset.make_relative(batch.to(device)))
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch_norm)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        metrics = save_validation_rollouts_csv(
            model,
            config,
            device,
            reference_batch,
            dataset,
            out_dir,
            epoch,
        )
        _save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
        )
        if losses:
            print(
                f"epoch {epoch}: "
                f"loss={np.mean(losses):.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    ref_final = reference_batch
    final_meta = save_validation_rollouts_csv(
        model,
        config,
        device,
        ref_final,
        dataset,
        out_dir,
        config.train_epochs,
    )
    (out_dir / "eqm_metrics.json").write_text(json.dumps(final_meta, indent=2))
    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'eqm_metrics.json'}")


if __name__ == "__main__":
    main()
