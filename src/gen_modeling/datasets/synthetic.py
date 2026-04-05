import math
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_checkerboard, make_moons, make_swiss_roll


class SyntheticAmbientDataset(Dataset):
    """
    Low-dimensional distribution isometrically embedded in R^{ambient_dim} with
    orthonormal columns Q: x = z @ Q.T, z in R^{intrinsic_dim}.

    Use ``unproject`` to recover intrinsic coordinates z = x @ Q from ambient x.
    """

    intrinsic_dim: ClassVar[int]
    Q: torch.Tensor
    data: torch.Tensor
    ambient_dim: int

    def unproject(self, x_ambient: torch.Tensor) -> torch.Tensor:
        """Map ambient points to intrinsic coordinates: z = x @ Q."""
        return x_ambient @ self.Q

    def embed(self, z_intrinsic: torch.Tensor) -> torch.Tensor:
        """Map intrinsic points to ambient: x = z @ Q.T."""
        return z_intrinsic @ self.Q.T


class SwissRollDataset(SyntheticAmbientDataset):
    """
    Swiss roll from sklearn (points in R^3) embedded in R^{ambient_dim} via a fixed random
    matrix with orthonormal columns: x = z @ Q.T with z in R^3.
    """

    intrinsic_dim: ClassVar[int] = 3

    def __init__(
        self,
        ambient_dim: int,
        n_samples: int = 10_000,
        noise: float = 0.05,
        hole: bool = False,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if ambient_dim < self.intrinsic_dim:
            raise ValueError(
                f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({self.intrinsic_dim})"
            )
        self.ambient_dim = ambient_dim
        z_np, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
            hole=hole,
        )
        z = torch.as_tensor(z_np, dtype=torch.float32) / 10.0
        self.Q = torch.randn(ambient_dim, self.intrinsic_dim, dtype=torch.float32)
        self.Q, _ = torch.linalg.qr(self.Q)
        self.Q = self.Q.to(device)
        z = z.to(device)
        self.data = self.embed(z)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)


class MoonsDataset(SyntheticAmbientDataset):
    """
    Two moons in R^2 embedded in R^{ambient_dim} with the same orthonormal-column construction.
    """

    intrinsic_dim: ClassVar[int] = 2

    def __init__(
        self,
        ambient_dim: int,
        n_samples: int = 10_000,
        noise: float = 0.05,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if ambient_dim < self.intrinsic_dim:
            raise ValueError(
                f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({self.intrinsic_dim})"
            )
        self.ambient_dim = ambient_dim
        z_np, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        z = torch.as_tensor(z_np, dtype=torch.float32)
        self.Q = torch.randn(ambient_dim, self.intrinsic_dim, dtype=torch.float32)
        self.Q, _ = torch.linalg.qr(self.Q)
        self.Q = self.Q.to(device)
        z = z.to(device)
        self.data = self.embed(z)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)


class GaussianMixtureDataset(SyntheticAmbientDataset):
    """
    Non-uniform Gaussian mixture in R^2 (isotropic components with means on a circle),
    embedded in R^{ambient_dim} via a fixed random matrix with orthonormal columns:
    x = z @ Q.T with z in R^2. Mixture weights are random (softmax of normals), not uniform.
    Each cluster uses its own isotropic scale, drawn uniformly from ``scale_range``.
    """

    intrinsic_dim: ClassVar[int] = 2

    def __init__(
        self,
        ambient_dim: int,
        n_samples: int = 10_000,
        n_components: int = 5,
        radius: float = 1.0,
        scale_range: tuple[float, float] = (0.04, 0.15),
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if ambient_dim < self.intrinsic_dim:
            raise ValueError(
                f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({self.intrinsic_dim})"
            )
        low, high = scale_range
        if not (0 < low <= high):
            raise ValueError(
                f"scale_range must satisfy 0 < low <= high; got {(low, high)}"
            )
        self.ambient_dim = ambient_dim

        gen = torch.Generator(device="cpu")
        if random_state is not None:
            gen.manual_seed(random_state)

        logits = torch.randn(n_components, generator=gen, dtype=torch.float32)
        weights = torch.softmax(logits, dim=0)

        comp_scales = (
            torch.rand(n_components, generator=gen, dtype=torch.float32) * (high - low) + low
        )

        angles = 2 * math.pi * torch.arange(n_components, dtype=torch.float32) / n_components
        means = torch.stack(
            [radius * torch.cos(angles), radius * torch.sin(angles)],
            dim=1,
        )

        assign = torch.multinomial(weights, n_samples, replacement=True, generator=gen)
        eps = torch.randn(
            n_samples, self.intrinsic_dim, generator=gen, dtype=torch.float32
        )
        z = means[assign] + comp_scales[assign, None] * eps

        self.Q = torch.randn(
            ambient_dim, self.intrinsic_dim, generator=gen, dtype=torch.float32
        )
        self.Q, _ = torch.linalg.qr(self.Q)
        self.Q = self.Q.to(device)
        z = z.to(device)
        self.data = self.embed(z)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)


class CheckerboardDataset(SyntheticAmbientDataset):
    """
    2D point cloud from `sklearn.datasets.make_checkerboard`: build a checkerboard matrix,
    take all cells below the median value as one "color", sample ``n_samples`` cells uniformly
    with jitter inside each cell, optionally add small extrinsic jitter, then embed with
    ``x = z @ Q.T``. Use ``shuffle=False`` (default) so matrix indices match spatial layout.
    """

    intrinsic_dim: ClassVar[int] = 2

    def __init__(
        self,
        ambient_dim: int,
        n_samples: int = 10_000,
        grid_cells: int = 32,
        cell_size: float = 0.125,
        n_clusters: int = 8,
        noise: float = 2.0,
        minval: float = 0.0,
        maxval: float = 100.0,
        shuffle: bool = False,
        jitter: float = 0.0,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if ambient_dim < self.intrinsic_dim:
            raise ValueError(
                f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({self.intrinsic_dim})"
            )
        if grid_cells < 2:
            raise ValueError("grid_cells must be >= 2")
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        self.ambient_dim = ambient_dim

        gen = torch.Generator(device="cpu")
        if random_state is not None:
            gen.manual_seed(random_state)

        X, _, _ = make_checkerboard(
            shape=(grid_cells, grid_cells),
            n_clusters=n_clusters,
            noise=noise,
            minval=minval,
            maxval=maxval,
            shuffle=shuffle,
            random_state=random_state,
        )
        median = float(np.median(X))
        rows, cols = np.nonzero(X < median)
        pool = np.stack([rows, cols], axis=1)
        if pool.shape[0] == 0:
            raise ValueError("make_checkerboard produced an empty low-mask; try other parameters")

        rng = np.random.default_rng(random_state)
        pick = rng.integers(0, pool.shape[0], size=n_samples, endpoint=False)
        sel = pool[pick]
        ux = rng.random(n_samples).astype(np.float32)
        uy = rng.random(n_samples).astype(np.float32)
        half = grid_cells * cell_size / 2.0
        z_np = np.stack(
            [
                -half + (sel[:, 0].astype(np.float32) + ux) * cell_size,
                -half + (sel[:, 1].astype(np.float32) + uy) * cell_size,
            ],
            axis=1,
        )
        if jitter > 0:
            z_np = z_np + jitter * rng.standard_normal((n_samples, 2)).astype(np.float32)
        z = torch.as_tensor(z_np, dtype=torch.float32)

        self.Q = torch.randn(
            ambient_dim, self.intrinsic_dim, generator=gen, dtype=torch.float32
        )
        self.Q, _ = torch.linalg.qr(self.Q)
        self.Q = self.Q.to(device)
        z = z.to(device)
        self.data = self.embed(z)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)