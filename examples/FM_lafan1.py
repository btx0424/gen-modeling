"""
Flow Matching on LAFAN1-style robot trajectories.

Each epoch, sliding-window rollouts are built from shared ``lafan1_config.SlidingWindowConfig``
(``seq_len``, ``cond_steps``, ``stride``, ``val_total_len``, ``val_window_stride``), matching
``EqM_lafan1.py``. CSV rollouts are written under ``outputs/.../validation/``.
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
from lafan1_config import SlidingWindowConfig

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.robotics import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
    RobotName,
)
from gen_modeling.flow_matching import (
    LinearFlow,
    LossType,
    ModelArch,
    PredictionType,
    prediction_wrapper_class,
)
from gen_modeling.modules import ConditionalUNet1D
from gen_modeling.utils.optim import MuonAdamWWrapper


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    batch_size: int = 128
    base_channels: int = 128
    cond_dim: int = 256
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


class TrajectoryFlowBackbone(nn.Module):
    def __init__(self, input_dim: int, base_channels: int, cond_dim: int):
        super().__init__()
        self.sample_shape = (None, input_dim)
        self.cond_dim = cond_dim
        self.unet = ConditionalUNet1D(
            input_dim=input_dim,
            output_dim=input_dim,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
            cond_dim=cond_dim,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = torch.zeros(x_t.shape[0], self.cond_dim, device=x_t.device, dtype=x_t.dtype)
        return self.unet(x_t, cond, t)


def build_model(config: Config, state_dim: int, seq_len: int) -> nn.Module:
    base_network = TrajectoryFlowBackbone(state_dim, config.base_channels, config.cond_dim)
    base_network.sample_shape = (seq_len, state_dim)
    wrapper_cls = prediction_wrapper_class(config.model_arch)
    return wrapper_cls(base_network, config.pred_type)


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


@torch.no_grad()
def sliding_window_generate(
    flow: LinearFlow,
    dataset: LAFAN1Dataset,
    config: Config,
    device: torch.device,
    cond_prefix: Float[Tensor, "batch cond dim"],
    total_len: int,
    window_stride: int,
) -> Float[Tensor, "batch total dim"]:
    """
    Same contract as ``EqM_lafan1.sliding_window_generate``: stitch long trajectories in
    physical (unnormalized) trajectory space using normalized windows and root-frame merge.
    """
    k = config.sliding.cond_steps
    seq_len = config.sliding.seq_len
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

    dtype = next(flow.model.parameters()).dtype
    n, _, dim = cond_prefix.shape
    traj = torch.zeros(n, total_len, dim, device=device, dtype=dtype)
    cond0 = dataset.make_relative(cond_prefix.to(device=device, dtype=dtype))
    traj[:, :k] = cond0

    ws = 0
    while True:
        if ws + k > total_len:
            break
        traj_ws = traj[:, ws : ws + k]
        cond_local = dataset.normalize(dataset.make_relative(traj_ws))
        root_pos_ref = traj[:, ws, :3]
        root_rot6d_ref = traj[:, ws, ROOT_ROT_OFFSET:POSE_BASE_DIM]

        chunk = flow.sample_cond_prefix(cond_local, device, config.sample_steps)
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


@torch.no_grad()
def save_validation_rollouts_csv(
    flow: LinearFlow,
    config: Config,
    device: torch.device,
    reference_batch: Float[Tensor, "batch seq dim"],
    dataset: LAFAN1Dataset,
    out_dir: Path,
    epoch: int,
) -> dict[str, float | int | str]:
    flow.model.eval()
    k = config.sliding.cond_steps
    n = min(config.num_plot_samples, reference_batch.shape[0])
    cond_prefix = reference_batch[:n, :k].to(device)
    stride = config.sliding.val_window_stride
    total_len = config.sliding.val_total_len
    traj = sliding_window_generate(
        flow,
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

    meta: dict[str, float | int | str] = {
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
    meta.update(dataset.compute_metrics(traj_denorm))
    (out_dir / f"fm_lafan1_epoch_{epoch:03d}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--use-muon-adamw", action="store_true", help="Use MuonAdamWWrapper.")
    parser.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    parser.add_argument("--t-eps", type=float, default=Config.t_eps)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--model-arch", choices=["vanilla", "global_residual", "corrected_residual1", "corrected_residual2"], default=Config.model_arch)
    parser.add_argument("--pred-type", choices=["x", "eps", "v"], default=Config.pred_type)
    parser.add_argument("--loss-type", choices=["x", "eps", "v"], default=Config.loss_type)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path (default: examples/outputs/FM_lafan1/checkpoint.pt).",
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

    model = build_model(config, state_dim, config.sliding.seq_len).to(device)
    if config.use_muon_adamw:
        optimizer = MuonAdamWWrapper([model], lr=config.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    flow = LinearFlow(
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
        conditional=False,
    )

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="gen-modeling",
            name="FM_lafan1",
            config=dataclasses.asdict(config),
        )

    out_dir = Path(__file__).resolve().parent / "outputs" / "FM_lafan1"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        start_epoch = _load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    ref_final: torch.Tensor | None = None
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
            loss = flow.compute_loss(x, cond_steps=config.sliding.cond_steps)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        assert reference_batch is not None
        metrics = save_validation_rollouts_csv(
            flow,
            config,
            device,
            reference_batch,
            dataset,
            out_dir,
            epoch,
        )
        _save_checkpoint(
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
        flow,
        config,
        device,
        ref_final,
        dataset,
        out_dir,
        config.train_epochs,
    )
    (out_dir / "fm_metrics.json").write_text(json.dumps(final_meta, indent=2))
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
                name=f"fm_lafan1_checkpoint_{wandb_run.id}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'fm_metrics.json'}")


if __name__ == "__main__":
    main()
