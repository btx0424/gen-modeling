"""
Equilibrium Matching on LAFAN1-style robot trajectories.

This is a scaffold example using the shared 1D conditional U-Net over sequences.
Inputs and outputs have shape (B, T, C), matching ``LAFAN1Dataset`` windows.
Conditioning is done by pinning the first ``k`` timesteps (see ``lafan1_config.SlidingWindowConfig``); the model
is trained to match the flow field on later steps while keeping the prefix fixed.
Each epoch, sliding-window rollouts are denormalized and saved as CSV under ``outputs/.../validation/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.robotics import LAFAN1Dataset, RobotName
from gen_modeling.modules import ConditionalUNet1D
from gen_modeling.utils.checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_config,
    save_training_checkpoint,
)
from gen_modeling.utils.optim import MuonAdamWWrapper

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from lafan1_config import SlidingWindowConfig, save_validation_rollouts_csv


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    batch_size: int = 128
    base_channels: int = 128
    cond_dim: int = 256
    time_conditioning: bool = False
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    use_muon_adamw: bool = False
    sample_steps: int = 80
    sample_stepsize: float = 0.01
    sample_sampler: Literal["gd", "nag"] = "nag"
    sample_mu: float = 0.3
    # Noise level schedule for time-conditional sampling (matches flow ``t_eps`` convention).
    sample_t_eps: float = 1e-2
    num_plot_samples: int = 16
    num_plot_dims: int = 6
    use_wandb: bool = True


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    def func(t: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
        return grad_scale * torch.where(t < a, 1.0, (1.0 - t) / (1.0 - a))

    return func


class TrajectoryEqMBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        base_channels: int,
        cond_dim: int,
        time_conditioning: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_conditioning = time_conditioning
        self.unet = ConditionalUNet1D(
            input_dim=input_dim,
            output_dim=input_dim,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x_t: Float[Tensor, "batch time dim"],
        t: Float[Tensor, "batch"] | None = None,
        cond: Tensor | None = None,
    ) -> Float[Tensor, "batch time dim"]:
        """U-Net on the trajectory; ``cond`` is unused (prefix is pinned outside)."""
        _ = cond
        if self.time_conditioning:
            if t is None:
                raise ValueError("TrajectoryEqMBackbone: time_conditioning requires ``t``.")
            return self.unet(x_t, cond=None, t=t)
        return self.unet(x_t, cond=None, t=None)


class EqM(nn.Module):
    def __init__(self, network: nn.Module, *, cond_steps: int = 4):
        super().__init__()
        if cond_steps < 1:
            raise ValueError("cond_steps must be >= 1")
        self.network = network
        self.time_conditioning = network.time_conditioning
        self.cond_steps = cond_steps
        self.grad_magnitude = eqm_ct()

    def compute_loss(self, x1: Float[Tensor, "batch seq dim"]) -> Float[Tensor, ""]:
        k = self.cond_steps
        if x1.shape[1] < k:
            raise ValueError(
                f"sequence length {x1.shape[1]} is shorter than cond_steps={k}"
            )
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t_view = t.view(-1, 1, 1)
        x0 = torch.randn_like(x1)
        xt = t_view * x1 + (1.0 - t_view) * x0
        prefix = x1[:, :k]
        xt[:, :k] = prefix
        target = (x1 - x0) * self.grad_magnitude(t).view(-1, 1, 1)
        target[:, :k] = 0.0
        pred = self.network(xt, t=t, cond=None)
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
        t_eps: float = 1e-2,
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
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        x = torch.randn(num_samples, seq_len, feat_dim, device=device, dtype=dtype)
        x[:, :k] = cond_prefix
        ts = torch.linspace(t_eps, 1.0 - t_eps, num_steps, device=device, dtype=dtype)
        for t_scalar in ts:
            t_batch = t_scalar.expand(num_samples,)
            update = self.network(x, t=t_batch, cond=None)
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
        t_eps: float = 1e-2,
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
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        x = torch.randn(num_samples, seq_len, feat_dim, device=device, dtype=dtype)
        x[:, :k] = cond_prefix
        momentum = torch.zeros_like(x)
        ts = torch.linspace(t_eps, 1.0 - t_eps, num_steps, device=device, dtype=dtype)
        for t_scalar in ts:
            t_batch = t_scalar.expand(num_samples,)
            lookahead = x + stepsize * mu * momentum
            lookahead[:, :k] = cond_prefix
            momentum = self.network(lookahead, t=t_batch, cond=None)
            momentum[:, :k] = 0.0
            x = x + stepsize * momentum
            x[:, :k] = cond_prefix
        return x


def _eqm_validation_sample_chunk(
    eqm: EqM,
    config: Config,
    device: torch.device,
) -> Callable[[Tensor], Tensor]:
    sample_fn = eqm.sample_nag if config.sample_sampler == "nag" else eqm.sample_gd
    kwargs: dict[str, float] = {"t_eps": config.sample_t_eps}
    if config.sample_sampler == "nag":
        kwargs["mu"] = config.sample_mu
    seq_len = config.sliding.seq_len

    def sample_chunk(cond_local: Tensor) -> Tensor:
        return sample_fn(
            cond_prefix=cond_local,
            seq_len=seq_len,
            device=device,
            num_steps=config.sample_steps,
            stepsize=config.sample_stepsize,
            **kwargs,
        )

    return sample_chunk


def _assert_resume_compatible(checkpoint_path: Path, config: Config) -> None:
    ckpt_config = read_training_checkpoint_config(checkpoint_path)
    ckpt_time = ckpt_config.get("time_conditioning")
    if ckpt_time is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing `time_conditioning`; refusing to resume "
            "because the time-conditioning path materially changes model behavior."
        )
    if bool(ckpt_time) != config.time_conditioning:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with time_conditioning={ckpt_time}, "
            f"but current config requests time_conditioning={config.time_conditioning}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 EqM scaffold.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument(
        "--time-conditioning",
        action=argparse.BooleanOptionalAction,
        default=Config.time_conditioning,
        help="Enable scalar time conditioning in the trajectory U-Net.",
    )
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument(
        "--use-muon-adamw",
        action="store_true",
        help="Use MuonAdamWWrapper instead of plain AdamW.",
    )
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--sample-stepsize", type=float, default=Config.sample_stepsize)
    parser.add_argument("--sample-sampler", choices=["gd", "nag"], default=Config.sample_sampler)
    parser.add_argument("--sample-mu", type=float, default=Config.sample_mu)
    parser.add_argument(
        "--sample-t-eps",
        type=float,
        default=Config.sample_t_eps,
        help="Endpoints for the sampling-time schedule [t_eps, 1-t_eps] when time_conditioning is True.",
    )
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--num-plot-dims", type=int, default=Config.num_plot_dims)
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
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        time_conditioning=args.time_conditioning,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        use_muon_adamw=args.use_muon_adamw,
        sample_steps=args.sample_steps,
        sample_stepsize=args.sample_stepsize,
        sample_sampler=args.sample_sampler,
        sample_mu=args.sample_mu,
        sample_t_eps=args.sample_t_eps,
        num_plot_samples=args.num_plot_samples,
        num_plot_dims=args.num_plot_dims,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.sliding.seq_len,
        stride=config.sliding.stride,
        download=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    state_dim = dataset.state_dim

    model = EqM(
        network=TrajectoryEqMBackbone(
            input_dim=state_dim,
            base_channels=config.base_channels,
            cond_dim=config.cond_dim,
            time_conditioning=config.time_conditioning,
        ),
        cond_steps=config.sliding.cond_steps,
    ).to(device)
    if config.use_muon_adamw:
        optimizer = MuonAdamWWrapper([model], lr=config.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    wandb_run = None
    if config.use_wandb:
        wandb_run = wandb.init(
            project="gen-modeling",
            name=f"EqM_lafan1",
            config=dataclasses.asdict(config),
        )
    out_dir = Path(__file__).resolve().parent / "outputs" / "EqM_lafan1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    reference_batch: torch.Tensor | None = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        _assert_resume_compatible(checkpoint_path, config)
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    val_dtype = next(model.parameters()).dtype
    val_sample_chunk = _eqm_validation_sample_chunk(model, config, device)
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

        assert reference_batch is not None
        metrics = save_validation_rollouts_csv(
            eval_module=model,
            sliding=config.sliding,
            num_plot_samples=config.num_plot_samples,
            device=device,
            reference_batch=reference_batch,
            dataset=dataset,
            out_dir=out_dir,
            epoch=epoch,
            metrics_name_prefix="eqm_lafan1",
            sample_chunk=val_sample_chunk,
            dtype=val_dtype,
        )
        save_training_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
        )
        if losses:
            avg_loss = float(np.mean(losses))
            print(
                f"epoch {epoch}: "
                f"loss={avg_loss:.6f}, "
                f"sample_std={metrics['sample_std']:.6f}, "
                f"root_vel_fd_mse={metrics['root_vel_fd_mse']:.6f}, "
                f"joint_vel_fd_mse={metrics['joint_vel_fd_mse']:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "val/sample_mean": float(metrics["sample_mean"]),
                        "val/sample_std": float(metrics["sample_std"]),
                        "val/sample_min": float(metrics["sample_min"]),
                        "val/sample_max": float(metrics["sample_max"]),
                        "val/root_vel_fd_mse": float(metrics["root_vel_fd_mse"]),
                        "val/joint_vel_fd_mse": float(metrics["joint_vel_fd_mse"]),
                    },
                    step=epoch,
                )

    ref_final = reference_batch
    assert ref_final is not None
    final_meta = save_validation_rollouts_csv(
        eval_module=model,
        sliding=config.sliding,
        num_plot_samples=config.num_plot_samples,
        device=device,
        reference_batch=ref_final,
        dataset=dataset,
        out_dir=out_dir,
        epoch=config.train_epochs,
        metrics_name_prefix="eqm_lafan1",
        sample_chunk=val_sample_chunk,
        dtype=val_dtype,
    )
    (out_dir / "eqm_metrics.json").write_text(json.dumps(final_meta, indent=2))
    if wandb_run is not None:
        wandb_run.log(
            {
                "final/sample_mean": float(final_meta["sample_mean"]),
                "final/sample_std": float(final_meta["sample_std"]),
                "final/sample_min": float(final_meta["sample_min"]),
                "final/sample_max": float(final_meta["sample_max"]),
                "final/root_vel_fd_mse": float(final_meta["root_vel_fd_mse"]),
                "final/joint_vel_fd_mse": float(final_meta["joint_vel_fd_mse"]),
            },
            step=config.train_epochs,
        )
        if checkpoint_path.is_file():
            artifact = wandb.Artifact(
                name=f"eqm_lafan1_checkpoint_{wandb_run.id}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'eqm_metrics.json'}")


if __name__ == "__main__":
    main()
