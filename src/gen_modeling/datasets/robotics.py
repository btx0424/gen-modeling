"""
Robot motion datasets (e.g. LAFAN1 retargeted mocap).

Each CSV row is one configuration vector: ``[x, y, z, qw, qx, qy, qz, ...joints]`` suitable
for ``pin.RobotWrapper`` (free-flyer + actuated joints), as used in the LAFAN1 retargeting
release layout: ``<root>/<robot>/*.csv``.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

RobotName = Literal["g1", "h1", "h1_2"]


class LAFAN1Dataset(Dataset):
    """
    Sliding-window access to LAFAN1-style retargeted trajectories for one robot.

    All clips are loaded into memory at construction (``float32``) to avoid disk I/O
    during training.

    Parameters
    ----------
    root
        Dataset root directory containing subfolders ``g1/``, ``h1/``, ``h1_2/`` with
        ``*.csv`` files (no header; one float row per frame).
    robot
        Which robot / DoF layout to load.
    seq_len
        Number of consecutive frames per sample, shape ``(seq_len, state_dim)``.
    stride
        Frame stride between consecutive windows within the same clip.
    dtype
        Output tensor dtype (default ``float32``).
    """

    STATE_DIM: dict[RobotName, int] = {
        "h1": 26,
        "h1_2": 34,
        "g1": 36,
    }

    def __init__(
        self,
        root: str | Path,
        robot: RobotName = "h1",
        seq_len: int = 30,
        stride: int = 1,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.root = Path(root).expanduser().resolve()
        self.robot: RobotName = robot
        self.seq_len = seq_len
        self.stride = stride
        self.dtype = dtype

        self.state_dim = self.STATE_DIM[robot]
        clip_dir = self.root / robot
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"Missing robot folder: {clip_dir}")

        paths = sorted(clip_dir.glob("*.csv"))

        self._clips: list[np.ndarray] = []
        self._clip_names: list[str] = []
        windows_per_clip: list[int] = []

        for p in paths:
            data = np.loadtxt(p, delimiter=",", dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != self.state_dim:
                raise ValueError(
                    f"{p.name}: expected {self.state_dim} columns for {self.robot}, "
                    f"got {data.shape[1]}"
                )
            n = data.shape[0]
            nw = _num_windows(n, seq_len, stride)
            if nw == 0:
                continue
            self._clips.append(data)
            self._clip_names.append(p.name)
            windows_per_clip.append(nw)

        if not self._clips:
            raise ValueError(
                f"No windows of length {seq_len} in {clip_dir}; all clips are too short."
            )

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

        arr = self._clips[clip_idx]
        chunk = arr[start:end]
        tensor = torch.as_tensor(chunk, dtype=self.dtype)
        meta = {
            "clip_index": clip_idx,
            "frame_start": start,
            "robot": self.robot,
            "path": self._clip_names[clip_idx],
        }
        return tensor, meta


def _num_windows(n_rows: int, seq_len: int, stride: int) -> int:
    if n_rows < seq_len:
        return 0
    return (n_rows - seq_len) // stride + 1
