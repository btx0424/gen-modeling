import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll, make_moons

class SwissRollDataset(Dataset):
    """
    Swiss roll from sklearn (points in R^3) embedded in R^{ambient_dim} via a fixed random
    matrix with orthonormal columns: x = z @ Q.T with z in R^3.
    """

    intrinsic_dim: int = 3

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
        z = torch.as_tensor(z_np, dtype=torch.float32)
        self.Q = torch.randn(ambient_dim, self.intrinsic_dim, dtype=torch.float32)
        self.Q, _ = torch.linalg.qr(self.Q)
        self.Q = self.Q.to(device)
        z = z.to(device)
        self.data = z @ self.Q.T

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)


class MoonsDataset(Dataset):
    """
    Two moons in R^2 embedded in R^{ambient_dim} with the same orthonormal-column construction.
    """

    intrinsic_dim: int = 2

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
        self.data = z @ self.Q.T

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        return x, torch.zeros((), dtype=torch.long, device=x.device)