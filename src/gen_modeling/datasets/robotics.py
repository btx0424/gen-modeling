"""
Robot motion datasets (e.g. LAFAN1 retargeted mocap).

CSV rows are qpos only: ``[x, y, z, qw, qx, qy, qz, ...jpos]`` (free flyer + actuated joints),
as in the LAFAN1 retargeting layout ``<root>/<robot>/*.csv``. The root quaternion is converted
to **6D rotation** (Zhou et al.: first two columns of ``R``, flattened). Joint velocities
``jvel`` are appended per frame from ``jpos`` differences times ``fps``. Layout per frame:
``[x,y,z, rot6d(6), jpos..., jvel...]``.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from gen_modeling.utils.math import (
    yaw_quat,
    yaw_matrix,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_rot6d,
    quat_wxyz_to_xyzw,
    rot6d_from_matrix,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
)

RobotName = Literal["g1", "h1", "h1_2"]

LAFAN1_HF_REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
# Same name as the Hugging Face / git clone top-level folder (``g1/``, ``h1/``, … live under it).
LAFAN1_REPO_DIRNAME = "LAFAN1_Retargeting_Dataset"

# Columns in CSV before joint positions: position (3) + quaternion (xyzw, 4).
CSV_QPOS_BASE_DIM = 7
# Processed trajectory layout per frame: position (3) + root rot6d (6) + jpos + jvel.
ROOT_ROT6D_DIM = 6
POSE_BASE_DIM = 3 + ROOT_ROT6D_DIM


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
    during training. Per-joint mean and population standard deviation are computed over
    all frames for jpos and jvel. Root position mean/std are computed over **relative**
    root translations from every training window (same ``seq_len`` / ``stride`` as
    :meth:`__getitem__`, via :meth:`make_relative`), matching the tensor layout seen after
    ``make_relative`` then ``normalize`` in typical training code. Root orientation (rot6d
    or quaternion) is not z-scored.
    Note that the original data's quaternions are in xyzw order. We convert them to wxyz order
    to align with MuJoCo's convention.

    Parameters
    ----------
    root
        Directory that either contains ``<robot>/*.csv`` directly **or** is the parent of a
        git-cloned / hub-style tree ``LAFAN1_Retargeting_Dataset/<robot>/*.csv``.
    robot
        Which robot / DoF layout to load.
    seq_len
        Number of consecutive frames per sample, shape ``(seq_len, state_dim)`` with
        ``state_dim = 9 + 2 * n_joints`` (pos + rot6d + jpos + jvel; ``n_joints`` from CSV).
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
        rot6d: bool = True
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
        self.rot6d = rot6d

        qpos_dim = self.QPOS_DIM[robot]
        n_joints = qpos_dim - CSV_QPOS_BASE_DIM
        if n_joints < 1:
            raise ValueError(f"robot {robot!r}: expected at least one joint column in qpos")
        self.qpos_dim = qpos_dim
        self.n_joints = n_joints
        self.state_dim = POSE_BASE_DIM + 2 * n_joints

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

        lo = POSE_BASE_DIM
        mid = lo + n_joints
        all_frames = torch.cat(self._clips, dim=0)
        jpos_block = all_frames[:, lo:mid]
        jvel_block = all_frames[:, mid:]
        eps = 1e-6
        self._jpos_mean = jpos_block.mean(dim=0)
        self._jpos_std = jpos_block.std(dim=0, correction=0).clamp_min(eps)
        self._jvel_mean = jvel_block.mean(dim=0)
        self._jvel_std = jvel_block.std(dim=0, correction=0).clamp_min(eps)

        rel_root_rows: list[torch.Tensor] = []
        for clip in self._clips:
            t_rows = clip.shape[0]
            for start in range(0, t_rows - seq_len + 1, stride):
                chunk = clip[start : start + seq_len]
                rel = self.make_relative(chunk)
                rel_root_rows.append(rel[:, :3].reshape(-1, 3))
        all_rel_root = torch.cat(rel_root_rows, dim=0)
        self._root_pos_mean = all_rel_root.mean(dim=0)
        self._root_pos_std = all_rel_root.std(dim=0, correction=0).clamp_min(eps)

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
    
    def process_data(self, qpos: np.ndarray, fps: float) -> torch.Tensor:
        """
        Root orientation as rot6d (from normalized quaternion wxyz); append jvel from jpos.
        """
        x = torch.as_tensor(qpos, dtype=torch.float32)
        root_pos = x[:, :3]
        xyzw = x[:, 3:7]
        wxyz = xyzw[:, [3, 0, 1, 2]]
        wxyz = wxyz / wxyz.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if self.rot6d:
            root_rot = quat_to_rot6d(wxyz)
        else:
            root_rot = wxyz
        jpos = x[:, CSV_QPOS_BASE_DIM:]
        t_rows, j = jpos.shape
        jvel = torch.zeros((t_rows, j), dtype=x.dtype, device=x.device)
        if t_rows >= 2:
            fp = float(fps)
            jvel[0] = (jpos[1] - jpos[0]) * fp
            jvel[1:] = (jpos[1:] - jpos[:-1]) * fp
        return torch.cat([root_pos, root_rot, jpos, jvel], dim=1)
    
    def make_relative(
        self,
        trajectory: torch.Tensor,
        xy_only: bool = True,
        yaw_only: bool = False,
    ) -> torch.Tensor:
        """
        Express a trajectory in the first-frame root coordinate system.

        Root translation is offset by frame 0 and then rotated by the inverse of the
        frame-0 root orientation. If ``xy_only`` is ``True`` (default), only ``x/y`` are
        offset in world space before this rotation (``z`` keeps its absolute value);
        if ``xy_only`` is ``False``, all ``x/y/z`` components are offset.

        Defaulting to ``xy_only=True`` keeps the global height anchor and tends to reduce
        long-horizon rollout drift/accumulated vertical bias when windows are stitched
        autoregressively.
        If ``yaw_only=True``, the frame-0 reference orientation is projected to yaw-only
        before local-frame conversion, which avoids chaining pitch/roll in the stitched
        reference frame and can reduce unnatural long-horizon tilting.

        This function defines the local-frame convention used by training targets and
        by rollout composition utilities.
        """
        device = trajectory.device
        root_pos = trajectory[..., :3]
        if xy_only:
            root_pos_rel = root_pos - root_pos[..., 0:1, :] * torch.tensor([1.0, 1.0, 0.0], device=device)
        else:
            root_pos_rel = root_pos - root_pos[..., 0:1, :]
        if self.rot6d:
            jstates = trajectory[..., 9:]
            root_rot6d = trajectory[..., 3:9]
            R = rot6d_to_matrix(root_rot6d)
            if yaw_only:
                R0 = yaw_matrix(R[..., 0:1, :, :])
            else:
                R0 = R[..., 0:1, :, :]
            R0_inv = R0.transpose(-1, -2)
            R_rel = torch.matmul(R0_inv, R)
            root_rot6d_rel = rot6d_from_matrix(R_rel)
            root_pos_rel = torch.matmul(R0_inv, root_pos_rel.unsqueeze(-1)).squeeze(-1)
            result = torch.cat([root_pos_rel, root_rot6d_rel, jstates], dim=-1)
        else:
            jstates = trajectory[..., 7:]
            root_wxyz = trajectory[..., 3:7]
            root_wxyz_0 = root_wxyz[..., 0:1, :]
            if yaw_only:
                root_wxyz_0 = yaw_quat(root_wxyz_0)
            root_wxyz_0_inv = quat_conjugate(root_wxyz_0).expand_as(root_wxyz)
            root_wxyz_rel = quat_mul(root_wxyz_0_inv, root_wxyz)
            root_pos_rel = quat_rotate(root_wxyz_0_inv, root_pos_rel)
            result = torch.cat([root_pos_rel, root_wxyz_rel, jstates], dim=-1)
        return result

    @staticmethod
    def accumulate_chunk_in_root_frame(
        chunk_local: torch.Tensor,
        root_pos_ref: torch.Tensor,
        root_rot6d_ref: torch.Tensor,
        xy_only: bool = True,
        yaw_only: bool = False,
    ) -> torch.Tensor:
        """
        Map a chunk expressed **relative to its first frame** (same convention as
        :meth:`make_relative` output) into the **rollout root frame** used by ``traj``:
        positions and rotations are relative to timestep 0 of the full trajectory.

        Joint blocks (jpos, jvel) are copied unchanged.

        Parameters
        ----------
        chunk_local
            ``(..., T, D)`` with root pos/rot6d relative to frame 0 of this chunk.
        root_pos_ref, root_rot6d_ref
            Global-encoding root pose at that chunk's frame 0, ``(..., 3)`` and ``(..., 6)``.
        xy_only
            Must match ``xy_only`` used by :meth:`make_relative`. With ``xy_only=True`` (default),
            only x/y translation is anchored to frame 0.
        yaw_only
            Must match ``yaw_only`` used by :meth:`make_relative`. If ``True``, compose with
            the yaw-only component of ``root_rot6d_ref``.
        """
        pos_l = chunk_local[..., :3]
        rot6d_l = chunk_local[..., 3:POSE_BASE_DIM]
        tail = chunk_local[..., POSE_BASE_DIM:]
        if yaw_only:
            R_ref = yaw_matrix(rot6d_to_matrix(root_rot6d_ref))
        else:
            R_ref = rot6d_to_matrix(root_rot6d_ref)
        R_l = rot6d_to_matrix(rot6d_l)
        root_pos_anchor = root_pos_ref.unsqueeze(-2)
        if xy_only:
            root_pos_anchor = root_pos_anchor * torch.tensor(
                [1.0, 1.0, 0.0],
                device=root_pos_ref.device,
                dtype=root_pos_ref.dtype,
            )
        pos_g = (
            torch.matmul(R_ref.unsqueeze(-3), pos_l.unsqueeze(-1)).squeeze(-1)
            + root_pos_anchor
        )
        R_g = torch.matmul(R_ref.unsqueeze(-3), R_l)
        rot6d_g = rot6d_from_matrix(R_g)
        return torch.cat([pos_g, rot6d_g, tail], dim=-1)

    def trajectory_to_lafan1_csv_qpos(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Convert a processed trajectory (root pos, rot6d, jpos, jvel) to the retargeting CSV
        **qpos** row layout: ``[x, y, z, qx, qy, qz, qw]`` (translation + quaternion **xyzw** +
        ``jpos``). **jvel is dropped** — original clips do not store it.
        """
        if self.rot6d:
            expected = 9 + 2 * self.n_joints
        else:
            expected = 7 + 2 * self.n_joints
        if traj.shape[-1] != expected:
            raise ValueError(
                f"trajectory last dim must be {expected} (state_dim), got {traj.shape[-1]}"
            )
        pos = traj[..., :3]
        if self.rot6d:
            rot6d = traj[..., 3:9]
            quat_wxyz = rot6d_to_quat_wxyz(rot6d)
            quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
            jpos = traj[..., 9:9 + self.n_joints]
        else:
            quat_wxyz = traj[..., 3:7]
            quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
            jpos = traj[..., 7:7 + self.n_joints]
        return torch.cat([pos, quat_xyzw, jpos], dim=-1)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Z-score root position (relative space), jpos, and jvel; root orientation unchanged."""
        lo = POSE_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        rp_mean = self._root_pos_mean.to(device=device, dtype=dtype)
        rp_std = self._root_pos_std.to(device=device, dtype=dtype)
        out[..., :3] = (out[..., :3] - rp_mean) / rp_std.clamp_min(1e-6)
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = (out[..., lo:mid] - jpos_mean) / jpos_std.clamp_min(1e-6)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., mid:] = (out[..., mid:] - jvel_mean) / jvel_std.clamp_min(1e-6)
        return out

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` for root position, jpos, and jvel."""
        lo = POSE_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        rp_mean = self._root_pos_mean.to(device=device, dtype=dtype)
        rp_std = self._root_pos_std.to(device=device, dtype=dtype)
        out[..., :3] = out[..., :3] * rp_std + rp_mean
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = out[..., lo:mid] * jpos_std + jpos_mean
        out[..., mid:] = out[..., mid:] * jvel_std + jvel_mean
        return out


def _num_windows(n_rows: int, seq_len: int, stride: int) -> int:
    if n_rows < seq_len:
        return 0
    return (n_rows - seq_len) // stride + 1
