"""Shared sliding-window settings and validation rollout helpers for LAFAN1 examples."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from gen_modeling.datasets.robotics import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
)


@dataclass
class SlidingWindowConfig:
    seq_len: int = 32
    cond_steps: int = 4
    stride: int = 1
    val_total_len: int = 128
    val_window_stride: int | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.cond_steps <= self.seq_len):
            raise ValueError(
                f"Need 1 <= cond-steps <= seq-len; got cond_steps={self.cond_steps}, seq_len={self.seq_len}"
            )
        if self.val_total_len < self.cond_steps:
            raise ValueError(
                f"val_total_len ({self.val_total_len}) must be >= cond_steps ({self.cond_steps})"
            )
        if self.val_window_stride is None:
            self.val_window_stride = self.seq_len - self.cond_steps


@torch.no_grad()
def sliding_window_generate(
    dataset: LAFAN1Dataset,
    sliding: SlidingWindowConfig,
    device: torch.device,
    cond_prefix: Float[Tensor, "batch cond dim"],
    total_len: int,
    window_stride: int,
    *,
    sample_chunk: Callable[[Tensor], Tensor],
    dtype: torch.dtype,
) -> Float[Tensor, "batch total dim"]:
    """
    Stitch long trajectories in physical (unnormalized) space using normalized windows
    and root-frame merge.

    ``sample_chunk`` maps a normalized prefix ``(N, k, D)`` to a full normalized window
    ``(N, seq_len, D)`` (flow ODE, diffusion reverse process, EqM sampler, etc.).
    """
    k = sliding.cond_steps
    seq_len = sliding.seq_len
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

        chunk = sample_chunk(cond_local)
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
    *,
    eval_module: nn.Module | None = None,
    sliding: SlidingWindowConfig,
    num_plot_samples: int,
    device: torch.device,
    reference_batch: Float[Tensor, "batch seq dim"],
    dataset: LAFAN1Dataset,
    out_dir: Path,
    epoch: int,
    metrics_name_prefix: str,
    sample_chunk: Callable[[Tensor], Tensor],
    dtype: torch.dtype,
) -> dict[str, float | int | str]:
    """
    Generate stitched rollouts from ``reference_batch`` prefixes, write CSVs under
    ``out_dir/validation/epoch_*/``, and save per-epoch metrics JSON
    ``{metrics_name_prefix}_epoch_{epoch:03d}.json``.
    """
    if eval_module is not None:
        eval_module.eval()
    k = sliding.cond_steps
    n = min(num_plot_samples, reference_batch.shape[0])
    cond_prefix = reference_batch[:n, :k].to(device)
    stride = sliding.val_window_stride
    assert stride is not None
    total_len = sliding.val_total_len
    traj = sliding_window_generate(
        dataset,
        sliding,
        device,
        cond_prefix,
        total_len,
        stride,
        sample_chunk=sample_chunk,
        dtype=dtype,
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
    (out_dir / f"{metrics_name_prefix}_epoch_{epoch:03d}.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta
