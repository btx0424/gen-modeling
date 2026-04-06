"""
Robot motion datasets (e.g. LAFAN1 retargeted mocap).

CSV rows are qpos only: ``[x, y, z, qw, qx, qy, qz, ...jpos]`` (free flyer + actuated joints),
as in the LAFAN1 retargeting layout ``<root>/<robot>/*.csv``. After loading, joint velocities
``jvel`` are appended per frame as finite differences of ``jpos`` times ``fps`` (same layout as
many sim stacks: ``[qpos | jvel]``).
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from gen_modeling.utils.math import quat_conjugate, quat_mul

RobotName = Literal["g1", "h1", "h1_2"]

LAFAN1_HF_REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
# Same name as the Hugging Face / git clone top-level folder (``g1/``, ``h1/``, … live under it).
LAFAN1_REPO_DIRNAME = "LAFAN1_Retargeting_Dataset"

# Base qpos layout: world position (3) + quaternion wxyz (4); remaining columns are jpos.
QPOS_BASE_DIM = 7


def _lafan1_base_with_clips(root: Path, robot: RobotName) -> Path | None:
    """Return ``base`` if ``base/<robot>/*.csv`` exists; try flat layout then clone/hub layout."""
    for base in (root, root / LAFAN1_REPO_DIRNAME):
        clip = base / robot
        if clip.is_dir() and any(clip.glob("*.csv")):
            return base
    return None


def _ensure_lafan1_robot_files(root: Path, robot: RobotName, *, download: bool) -> None:
    if _lafan1_base_with_clips(root, robot) is not None:
        return
    clip_dir = root / robot
    if not download:
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"Missing robot folder: {clip_dir}")
        raise ValueError(f"No CSV clips found in {clip_dir}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "LAFAN1Dataset with download=True requires `huggingface_hub`. "
            "Install it with: pip install huggingface_hub"
        ) from e
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=LAFAN1_HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=[f"{robot}/*.csv"],
        local_dir_use_symlinks=False,
    )
    if not clip_dir.is_dir() or not any(clip_dir.glob("*.csv")):
        raise RuntimeError(
            f"Download finished but no CSV files were found under {clip_dir}"
        )


class LAFAN1Dataset(Dataset):
    """
    Sliding-window access to LAFAN1-style retargeted trajectories for one robot.
    Can be downloaded from https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

    All clips are loaded into memory at construction (``float32``) to avoid disk I/O
    during training. Per-joint mean and population standard deviation are then computed
    over all frames in those clips for jpos and jvel; :meth:`normalize` / :meth:`denormalize`
    apply them while leaving root position and quaternion unchanged.

    Parameters
    ----------
    root
        Directory that either contains ``<robot>/*.csv`` directly **or** is the parent of a
        git-cloned / hub-style tree ``LAFAN1_Retargeting_Dataset/<robot>/*.csv``.
    robot
        Which robot / DoF layout to load.
    seq_len
        Number of consecutive frames per sample, shape ``(seq_len, state_dim)`` where
        ``state_dim = qpos_dim + n_joints`` (qpos from disk plus appended jvel).
    stride
        Frame stride between consecutive windows within the same clip.
    fps
        Sampling rate used to turn jpos finite differences into jvel (rad/s or deg/s per
        your CSV convention). LAFAN1 mocap is commonly 30 Hz.
    dtype
        Output tensor dtype (default ``float32``).
    download
        If ``True`` (default), fetch missing ``{robot}/*.csv`` from Hugging Face
        (``lvhaidong/LAFAN1_Retargeting_Dataset``) into ``root``. If ``False``,
        ``root/<robot>`` must already contain CSV clips.
    """

    QPOS_DIM: dict[RobotName, int] = {
        "h1": 26,
        "h1_2": 34,
        "g1": 36,
    }

    def __init__(
        self,
        root: str | Path,
        robot: RobotName = "g1",
        seq_len: int = 32,
        stride: int = 1,
        *,
        fps: float = 30.0,
        dtype: torch.dtype = torch.float32,
        download: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.root = Path(root).expanduser().resolve()
        self.robot: RobotName = robot
        self.seq_len = seq_len
        self.stride = stride
        self.fps = float(fps)
        self.dtype = dtype

        qpos_dim = self.QPOS_DIM[robot]
        n_joints = qpos_dim - QPOS_BASE_DIM
        if n_joints < 1:
            raise ValueError(f"robot {robot!r}: expected at least one joint column in qpos")
        self.qpos_dim = qpos_dim
        self.n_joints = n_joints
        self.state_dim = qpos_dim + n_joints

        resolved = _lafan1_base_with_clips(self.root, robot)
        if resolved is not None:
            self.root = resolved
        else:
            _ensure_lafan1_robot_files(self.root, robot, download=download)

        clip_dir = self.root / robot
        paths = sorted(clip_dir.glob("*.csv"))

        self._clips: list[torch.Tensor] = []
        self._clip_names: list[str] = []
        windows_per_clip: list[int] = []

        for p in paths:
            data = np.loadtxt(p, delimiter=",")
            n = data.shape[0]
            nw = _num_windows(n, seq_len, stride)
            if nw == 0:
                continue
            self._clips.append(self.process_data(data, self.fps))
            self._clip_names.append(p.name)
            windows_per_clip.append(nw)

        if not self._clips:
            raise ValueError(
                f"No windows of length {seq_len} in {clip_dir}; all clips are too short."
            )

        lo = QPOS_BASE_DIM
        mid = lo + n_joints
        all_frames = torch.cat(self._clips, dim=0)
        jpos_block = all_frames[:, lo:mid]
        jvel_block = all_frames[:, mid:]
        eps = 1e-6
        self._jpos_mean = jpos_block.mean(dim=0)
        self._jpos_std = jpos_block.std(dim=0, correction=0).clamp_min(eps)
        self._jvel_mean = jvel_block.mean(dim=0)
        self._jvel_std = jvel_block.std(dim=0, correction=0).clamp_min(eps)
        print(f"jpos_mean: {self._jpos_mean}")
        print(f"jpos_std: {self._jpos_std}")
        print(f"jvel_mean: {self._jvel_mean}")
        print(f"jvel_std: {self._jvel_std}")

        self._window_offsets: list[int] = [0]
        for w in windows_per_clip:
            self._window_offsets.append(self._window_offsets[-1] + w)

        self._total_windows = self._window_offsets[-1]

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int | str]]:
        if index < 0 or index >= self._total_windows:
            raise IndexError(index)

        clip_idx = bisect.bisect_right(self._window_offsets, index) - 1
        win_in_clip = index - self._window_offsets[clip_idx]
        start = win_in_clip * self.stride
        end = start + self.seq_len

        clip = self._clips[clip_idx]
        chunk = clip[start:end]
        meta = {
            "clip_index": clip_idx,
            "frame_start": start,
        }
        return chunk, meta
    
    @staticmethod
    def process_data(qpos: np.ndarray, fps: float) -> torch.Tensor:
        """
        Append per-frame jvel from jpos: backward difference in time, ``(jpos[t]-jpos[t-1])*fps``,
        with the first frame using the same segment as frame 1 (forward difference from t=0).
        """
        root_pos = qpos[:, :3]
        root_quat_xyzw = qpos[:, 3:7]
        root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
        jpos = qpos[:, QPOS_BASE_DIM:]
        t, j = jpos.shape
        jvel = np.empty((t, j))
        if t < 2:
            jvel.fill(0.0)
        else:
            delta0 = (jpos[1] - jpos[0]) * fps
            jvel[0] = delta0
            jvel[1:] = (jpos[1:] - jpos[:-1]) * fps
        processed = np.concatenate([root_pos, root_quat_wxyz, jpos, jvel], axis=1, dtype=np.float32)
        return torch.from_numpy(processed)
    
    @staticmethod
    def make_relative(trajectory: torch.Tensor) -> torch.Tensor:
        root_pos = trajectory[..., :3]
        root_quat_wxyz = trajectory[..., 3:7]
        root_pos = root_pos - root_pos[..., 0:1, :]
        root_quat_wxyz = quat_mul(
            quat_conjugate(root_quat_wxyz[..., 0:1, :]).expand_as(root_quat_wxyz),
            root_quat_wxyz
        )
        jpos_jvel = trajectory[..., QPOS_BASE_DIM:]
        return torch.cat([root_pos, root_quat_wxyz, jpos_jvel], dim=-1)
    
    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Scale jpos and jvel to roughly zero mean / unit variance; root pos and quat unchanged."""
        lo = QPOS_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = (out[..., lo:mid] - jpos_mean) / jpos_std.clamp_min(1e-6)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., mid:] = (out[..., mid:] - jvel_mean) / jvel_std.clamp_min(1e-6)
        return out

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` for jpos and jvel blocks."""
        lo = QPOS_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = out[..., lo:mid] * jpos_std + jpos_mean
        out[..., mid:] = out[..., mid:] * jvel_std + jvel_mean
        return out


def _num_windows(n_rows: int, seq_len: int, stride: int) -> int:
    if n_rows < seq_len:
        return 0
    return (n_rows - seq_len) // stride + 1
