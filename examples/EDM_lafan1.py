"""
Classic denoising diffusion on LAFAN1-style robot trajectories.

The script mirrors ``EqM_lafan1.py`` / ``FM_lafan1.py`` structure:
- same sliding-window config and rollout stitching
- same checkpoint / resume flow
- same per-epoch validation CSV export under ``outputs/.../validation/``

Conditioning is done by pinning the first ``cond_steps`` frames in each sampled window.

Sampling supports:
- DDPM: stochastic ancestral updates (injects fresh Gaussian noise each reverse step).
- DDIM: implicit updates with optional ``eta`` (``eta=0`` is deterministic and often faster
  at low step counts, while DDPM tends to preserve more stochasticity/diversity).
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
    time_conditioning: bool = True
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    use_muon_adamw: bool = False
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    pred_type: Literal["eps", "x"] = "eps"
    sample_sampler: Literal["ddpm", "ddim"] = "ddpm"
    sample_eta: float = 0.0
    sample_steps: int | None = None
    num_plot_samples: int = 16
    num_plot_dims: int = 6
    use_wandb: bool = True


class TrajectoryDiffusionBackbone(nn.Module):
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
                raise ValueError("TrajectoryDiffusionBackbone: time_conditioning requires ``t``.")
            return self.unet(x_t, cond=None, t=t)
        return self.unet(x_t, cond=None, t=None)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        *,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        pred_type: Literal["eps", "x"] = "eps",
        cond_steps: int = 4,
    ):
        super().__init__()
        if cond_steps < 1:
            raise ValueError("cond_steps must be >= 1")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        if pred_type not in ("eps", "x"):
            raise ValueError(f"Unsupported pred_type={pred_type}; expected 'eps' or 'x'")

        self.network = network
        self.time_conditioning = network.time_conditioning
        self.cond_steps = cond_steps
        self.diffusion_steps = diffusion_steps
        self.pred_type = pred_type

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            torch.clamp(posterior_variance, min=1e-20),
        )

    def _extract(
        self,
        values: Float[Tensor, "steps"],
        t: Tensor,
        x_shape: torch.Size,
    ) -> Tensor:
        out = values.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _time_input(self, t: Tensor) -> Tensor:
        return t.float() / float(self.diffusion_steps - 1)

    def _predict_eps(
        self,
        x_t: Float[Tensor, "batch seq dim"],
        t: Tensor,
        cond_prefix: Float[Tensor, "batch cond dim"],
    ) -> tuple[Float[Tensor, "batch seq dim"], Float[Tensor, "batch seq dim"]]:
        model_t = self._time_input(t) if self.time_conditioning else None
        pred = self.network(x_t, t=model_t, cond=None)
        k = self.cond_steps
        sqrt_ab_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omb_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        if self.pred_type == "eps":
            eps_pred = pred
            eps_pred[:, :k] = 0.0
            x0_pred = (x_t - sqrt_omb_t * eps_pred) / torch.clamp(sqrt_ab_t, min=1e-12)
            x0_pred[:, :k] = cond_prefix
        else:
            x0_pred = pred
            x0_pred[:, :k] = cond_prefix
            eps_pred = (x_t - sqrt_ab_t * x0_pred) / torch.clamp(sqrt_omb_t, min=1e-12)
            eps_pred[:, :k] = 0.0
        return eps_pred, x0_pred

    def compute_loss(self, x0: Float[Tensor, "batch seq dim"]) -> Float[Tensor, ""]:
        k = self.cond_steps
        if x0.shape[1] < k:
            raise ValueError(
                f"sequence length {x0.shape[1]} is shorter than cond_steps={k}"
            )

        batch_size = x0.shape[0]
        t = torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(batch_size,),
            device=x0.device,
        )
        noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        x_t[:, :k] = x0[:, :k]

        model_t = self._time_input(t) if self.time_conditioning else None
        pred = self.network(x_t, t=model_t, cond=None)
        if self.pred_type == "eps":
            pred[:, :k] = 0.0
            noise[:, :k] = 0.0
            return ((pred - noise) ** 2).mean()

        pred[:, :k] = x0[:, :k]
        return ((pred - x0) ** 2).mean()

    @torch.inference_mode()
    def sample_ddpm(
        self,
        cond_prefix: Float[Tensor, "batch cond dim"],
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int | None = None,
    ) -> Float[Tensor, "batch seq dim"]:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        n, _, feat_dim = cond_prefix.shape

        if num_steps is None:
            num_steps = self.diffusion_steps
        if not (1 <= num_steps <= self.diffusion_steps):
            raise ValueError(
                f"num_steps must be in [1, {self.diffusion_steps}], got {num_steps}"
            )

        x_t = torch.randn(n, seq_len, feat_dim, device=device, dtype=dtype)
        x_t[:, :k] = cond_prefix

        # Use a strided subset of timesteps for faster ancestral sampling when requested.
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=num_steps,
            device=device,
            dtype=torch.long,
        )

        for t_scalar in timesteps:
            t_int = int(t_scalar.item())
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            eps_pred, _x0_pred = self._predict_eps(x_t, t, cond_prefix)

            beta_t = self._extract(self.betas, t, x_t.shape)
            sqrt_omb_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

            model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_omb_t) * eps_pred)
            if t_int > 0:
                posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
                noise = torch.randn_like(x_t)
                noise[:, :k] = 0.0
                x_t = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x_t = model_mean
            x_t[:, :k] = cond_prefix

        return x_t

    @torch.inference_mode()
    def sample_ddim(
        self,
        cond_prefix: Float[Tensor, "batch cond dim"],
        seq_len: int,
        device: torch.device,
        *,
        num_steps: int | None = None,
        eta: float = 0.0,
    ) -> Float[Tensor, "batch seq dim"]:
        k = self.cond_steps
        dtype = next(self.parameters()).dtype
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
            raise ValueError(
                f"cond_prefix must have shape (N, {k}, D), got {tuple(cond_prefix.shape)}"
            )
        if eta < 0.0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        n, _, feat_dim = cond_prefix.shape

        if num_steps is None:
            num_steps = self.diffusion_steps
        if not (1 <= num_steps <= self.diffusion_steps):
            raise ValueError(
                f"num_steps must be in [1, {self.diffusion_steps}], got {num_steps}"
            )

        x_t = torch.randn(n, seq_len, feat_dim, device=device, dtype=dtype)
        x_t[:, :k] = cond_prefix
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            steps=num_steps,
            device=device,
            dtype=torch.long,
        )

        for idx, t_scalar in enumerate(timesteps):
            t_int = int(t_scalar.item())
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            eps_pred, x0_pred = self._predict_eps(x_t, t, cond_prefix)

            if idx + 1 < len(timesteps):
                prev_int = int(timesteps[idx + 1].item())
            else:
                prev_int = -1

            if prev_int >= 0:
                prev_t = torch.full((n,), prev_int, device=device, dtype=torch.long)
                alpha_bar_prev = self._extract(self.alphas_cumprod, prev_t, x_t.shape)
            else:
                alpha_bar_prev = torch.ones_like(x_t)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)

            sigma_t = (
                eta
                * torch.sqrt(torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0))
                * torch.sqrt(
                    torch.clamp(
                        1.0 - (alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-12)),
                        min=0.0,
                    )
                )
            )
            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0))
            if prev_int >= 0:
                noise = torch.randn_like(x_t)
                noise[:, :k] = 0.0
                x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_coeff * eps_pred + sigma_t * noise
            else:
                x_t = x0_pred
            x_t[:, :k] = cond_prefix

        return x_t


def _edm_validation_sample_chunk(
    diffusion: GaussianDiffusion,
    config: Config,
    device: torch.device,
) -> Callable[[Tensor], Tensor]:
    seq_len = config.sliding.seq_len

    def sample_chunk(cond_local: Tensor) -> Tensor:
        if config.sample_sampler == "ddim":
            return diffusion.sample_ddim(
                cond_prefix=cond_local,
                seq_len=seq_len,
                device=device,
                num_steps=config.sample_steps,
                eta=config.sample_eta,
            )
        return diffusion.sample_ddpm(
            cond_prefix=cond_local,
            seq_len=seq_len,
            device=device,
            num_steps=config.sample_steps,
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
    ckpt_steps = ckpt_config.get("diffusion_steps")
    if ckpt_steps is not None and int(ckpt_steps) != config.diffusion_steps:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with diffusion_steps={ckpt_steps}, "
            f"but current config requests diffusion_steps={config.diffusion_steps}."
        )
    ckpt_pred_type = ckpt_config.get("pred_type")
    if ckpt_pred_type is not None and ckpt_pred_type != config.pred_type:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with pred_type={ckpt_pred_type}, "
            f"but current config requests pred_type={config.pred_type}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 classic diffusion (DDPM/DDIM).")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument(
        "--time-conditioning",
        action=argparse.BooleanOptionalAction,
        default=Config.time_conditioning,
        help="Enable scalar normalized time conditioning in the trajectory U-Net.",
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
    parser.add_argument("--diffusion-steps", type=int, default=Config.diffusion_steps)
    parser.add_argument("--beta-start", type=float, default=Config.beta_start)
    parser.add_argument("--beta-end", type=float, default=Config.beta_end)
    parser.add_argument(
        "--pred-type",
        choices=["eps", "x"],
        default=Config.pred_type,
        help="Network prediction target: epsilon noise (`eps`) or clean sample (`x`).",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=Config.sample_steps,
        help="Number of reverse steps for sampling; defaults to diffusion_steps.",
    )
    parser.add_argument(
        "--sample-sampler",
        choices=["ddpm", "ddim"],
        default=Config.sample_sampler,
        help="Sampling algorithm: `ddpm` (stochastic ancestral) or `ddim` (implicit).",
    )
    parser.add_argument(
        "--sample-eta",
        type=float,
        default=Config.sample_eta,
        help="DDIM stochasticity. 0.0 = deterministic DDIM; >0 adds noise like generalized DDIM.",
    )
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--num-plot-dims", type=int, default=Config.num_plot_dims)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path (default: examples/outputs/EDM_lafan1/checkpoint.pt).",
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
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        pred_type=args.pred_type,
        sample_steps=args.sample_steps,
        sample_sampler=args.sample_sampler,
        sample_eta=args.sample_eta,
        num_plot_samples=args.num_plot_samples,
        num_plot_dims=args.num_plot_dims,
    )

    if config.sample_steps is None:
        config.sample_steps = config.diffusion_steps

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

    model = GaussianDiffusion(
        network=TrajectoryDiffusionBackbone(
            input_dim=state_dim,
            base_channels=config.base_channels,
            cond_dim=config.cond_dim,
            time_conditioning=config.time_conditioning,
        ),
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        pred_type=config.pred_type,
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
            name="EDM_lafan1",
            config=dataclasses.asdict(config),
        )
    out_dir = Path(__file__).resolve().parent / "outputs" / "EDM_lafan1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        _assert_resume_compatible(checkpoint_path, config)
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    ref_final: torch.Tensor | None = None
    val_dtype = next(model.parameters()).dtype
    val_sample_chunk = _edm_validation_sample_chunk(model, config, device)
    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []
        reference_batch: torch.Tensor | None = None

        for batch, _meta in pbar:
            if reference_batch is None:
                reference_batch = batch.cpu()
            x = dataset.normalize(dataset.make_relative(batch.to(device)))
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(x)
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
            metrics_name_prefix="edm_lafan1",
            sample_chunk=val_sample_chunk,
            dtype=val_dtype,
        )
        save_training_checkpoint(
            checkpoint_path, epoch=epoch, model=model, optimizer=optimizer, config=config
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

    if ref_final is None:
        ref_final = next(iter(loader))[0].cpu()
    final_meta = save_validation_rollouts_csv(
        eval_module=model,
        sliding=config.sliding,
        num_plot_samples=config.num_plot_samples,
        device=device,
        reference_batch=ref_final,
        dataset=dataset,
        out_dir=out_dir,
        epoch=config.train_epochs,
        metrics_name_prefix="edm_lafan1",
        sample_chunk=val_sample_chunk,
        dtype=val_dtype,
    )
    (out_dir / "edm_metrics.json").write_text(json.dumps(final_meta, indent=2))
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
                name=f"edm_lafan1_checkpoint_{wandb_run.id}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'edm_metrics.json'}")


if __name__ == "__main__":
    main()
