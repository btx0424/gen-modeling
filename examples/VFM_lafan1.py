"""
Variational Flow Matching on LAFAN1-style robot trajectories.

Sliding-window validation CSVs mirror ``FM_lafan1.py`` / ``EqM_lafan1.py`` (see
``lafan1_config.SlidingWindowConfig``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
from lafan1_config import SlidingWindowConfig, save_validation_rollouts_csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.robotics import LAFAN1Dataset, RobotName
from gen_modeling.flow_matching import (
    LossType,
    ModelArch,
    PredictionType,
    VariationalFlow,
    prediction_wrapper,
)
from gen_modeling.modules import ConditionalUNet1D, Encoder1D
from gen_modeling.utils.checkpoint import read_training_checkpoint_config, save_training_checkpoint
from gen_modeling.utils.optim import MuonAdamWWrapper


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    batch_size: int = 128
    base_channels: int = 128
    cond_dim: int = 256
    encoder_num_downsample: int = 2
    time_conditioning: bool = True
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    use_muon_adamw: bool = False
    noise_scale: float = 1.0
    t_eps: float = 1e-2
    sample_steps: int = 80
    num_plot_samples: int = 16
    model_arch: ModelArch = "vanilla"
    pred_type: PredictionType = "v"
    loss_type: LossType = "v"
    use_wandb: bool = True


class TrajectoryVariationalBackbone(nn.Module):
    """Conditional U-Net trajectory denoiser with latent ``cond`` (encoder output)."""

    def __init__(
        self,
        input_dim: int,
        base_channels: int,
        cond_dim: int,
        time_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.sample_shape = (None, input_dim)
        self.cond_dim = cond_dim
        self.time_conditioning = time_conditioning
        self.unet = ConditionalUNet1D(
            input_dim=input_dim,
            output_dim=input_dim,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
            cond_dim=cond_dim,
        )

    def forward(self, x_t: torch.Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        if cond is None:
            raise ValueError("TrajectoryVariationalBackbone requires cond (latent z).")
        if self.time_conditioning:
            return self.unet(x_t, cond=cond, t=t)
        return self.unet(x_t, cond=cond, t=None)


def build_model(config: Config, state_dim: int, seq_len: int) -> nn.Module:
    base = TrajectoryVariationalBackbone(
        input_dim=state_dim,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_conditioning=config.time_conditioning,
    )
    base.sample_shape = (seq_len, state_dim)
    return prediction_wrapper(base, config.pred_type, config.model_arch)


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


def load_vfm_checkpoint(
    path: Path,
    encoder: nn.Module,
    model: nn.Module,
    optimizer: optim.Optimizer,
    *,
    map_location: str | torch.device | None = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "encoder_state_dict" not in payload:
        raise ValueError(f"Checkpoint {path} is not a VFM checkpoint (missing encoder_state_dict).")
    encoder.load_state_dict(payload["encoder_state_dict"])
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].contiguous().cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    return int(payload["epoch"])


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 Variational Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument(
        "--encoder-num-downsample",
        type=int,
        default=Config.encoder_num_downsample,
        help="Encoder1D num_downsample (temporal stride-2 stages).",
    )
    parser.add_argument(
        "--time-conditioning",
        action=argparse.BooleanOptionalAction,
        default=Config.time_conditioning,
        help="Pass scalar t into the trajectory U-Net.",
    )
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--use-muon-adamw", action="store_true", help="Use MuonAdamWWrapper.")
    parser.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    parser.add_argument("--t-eps", type=float, default=Config.t_eps)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument(
        "--model-arch",
        choices=["vanilla", "global_residual", "corrected_residual1", "corrected_residual2"],
        default=Config.model_arch,
    )
    parser.add_argument("--pred-type", choices=["v"], default=Config.pred_type)
    parser.add_argument("--loss-type", choices=["v"], default=Config.loss_type)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path (default: examples/outputs/VFM_lafan1/checkpoint.pt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load encoder, flow model, optimizer, and RNG from --checkpoint and continue.",
    )
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        robot=args.robot,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        encoder_num_downsample=args.encoder_num_downsample,
        time_conditioning=args.time_conditioning,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        use_muon_adamw=args.use_muon_adamw,
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

    dataset = LAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.sliding.seq_len,
        stride=config.sliding.stride,
        download=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    state_dim = dataset.state_dim

    encoder = Encoder1D(
        state_dim,
        latent_dim=config.cond_dim,
        hidden_channels=config.base_channels,
        num_downsample=config.encoder_num_downsample,
    ).to(device)
    model = build_model(config, state_dim, config.sliding.seq_len).to(device)

    if config.use_muon_adamw:
        optimizer = MuonAdamWWrapper([encoder, model], lr=config.lr)
    else:
        optimizer = optim.AdamW(
            list(encoder.parameters()) + list(model.parameters()),
            lr=config.lr,
        )

    flow = VariationalFlow(
        encoder,
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
    )

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="gen-modeling",
            name="VFM_lafan1",
            config=dataclasses.asdict(config),
        )

    out_dir = Path(__file__).resolve().parent / "outputs" / "VFM_lafan1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        _assert_resume_compatible(checkpoint_path, config)
        start_epoch = load_vfm_checkpoint(checkpoint_path, encoder, model, optimizer, map_location=device)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    ref_final: torch.Tensor | None = None
    for epoch in range(start_epoch, config.train_epochs):
        encoder.train()
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []
        fm_losses: list[float] = []
        kl_losses: list[float] = []
        reference_batch: torch.Tensor | None = None

        for batch, _meta in pbar:
            if reference_batch is None:
                reference_batch = batch.cpu()
            x = dataset.normalize(dataset.make_relative(batch.to(device)))
            optimizer.zero_grad(set_to_none=True)
            loss, fm_loss, kl_loss = flow.compute_loss(x, cond_steps=config.sliding.cond_steps)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            fm_losses.append(fm_loss.item())
            kl_losses.append(kl_loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}", fm_loss=f"{fm_loss.item():.5f}", kl_loss=f"{kl_loss.item():.5f}")

        assert reference_batch is not None
        metrics = save_validation_rollouts_csv(
            eval_module=flow,
            sliding=config.sliding,
            num_plot_samples=config.num_plot_samples,
            device=device,
            reference_batch=reference_batch,
            dataset=dataset,
            out_dir=out_dir,
            epoch=epoch,
            metrics_name_prefix="vfm_lafan1",
            sample_chunk=lambda c: flow.sample_cond_prefix(c, device, config.sample_steps),
            dtype=next(flow.model.parameters()).dtype,
        )
        save_training_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
            extra={"encoder_state_dict": encoder.state_dict()},
        )
        if losses:
            avg_loss = float(np.mean(losses))
            avg_fm_loss = float(np.mean(fm_losses))
            avg_kl_loss = float(np.mean(kl_losses))
            print(
                f"epoch {epoch}: "
                f"loss={avg_loss:.6f}, "
                f"fm_loss={avg_fm_loss:.5f}, "
                f"kl_loss={avg_kl_loss:.5f}, "
                f"sample_std={metrics['sample_std']:.6f}, "
                f"root_vel_fd_mse={metrics['root_vel_fd_mse']:.6f}, "
                f"joint_vel_fd_mse={metrics['joint_vel_fd_mse']:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/fm_loss": avg_fm_loss,
                        "train/kl_loss": avg_kl_loss,
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
        eval_module=flow,
        sliding=config.sliding,
        num_plot_samples=config.num_plot_samples,
        device=device,
        reference_batch=ref_final,
        dataset=dataset,
        out_dir=out_dir,
        epoch=config.train_epochs,
        metrics_name_prefix="vfm_lafan1",
        sample_chunk=lambda c: flow.sample_cond_prefix(c, device, config.sample_steps),
        dtype=next(flow.model.parameters()).dtype,
    )
    (out_dir / "vfm_metrics.json").write_text(json.dumps(final_meta, indent=2))
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
                name=f"vfm_lafan1_checkpoint_{wandb_run.id}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'vfm_metrics.json'}")


if __name__ == "__main__":
    main()
