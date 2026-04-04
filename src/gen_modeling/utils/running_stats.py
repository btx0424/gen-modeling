import torch


class RunningNormalizationStats:
    """
    Online per-feature mean and (unbiased) variance using Chan's parallel merge on batches.
    First dimension is the sample axis; remaining dimensions are flattened to features.

    Use ``update`` while streaming data (or one shot on the training set), then
    ``normalize`` / ``unnormalize`` with the same statistics. Internal accumulation
    uses float64 for numerical stability; outputs follow ``x``'s device/dtype in
    ``normalize`` / ``unnormalize``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._count: int = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def reset(self) -> None:
        self._count = 0
        self._mean = None
        self._m2 = None

    def update(self, x: torch.Tensor) -> None:
        """Incorporate a batch ``x`` with shape ``(N, *features)``, ``N >= 1``."""
        if x.ndim < 1:
            raise ValueError("x must have shape (N, ...) with N >= 1")
        b = x.detach().reshape(x.shape[0], -1).to(dtype=torch.float64)
        n_b, d = b.shape
        mean_b = b.mean(dim=0)
        m2_b = ((b - mean_b) ** 2).sum(dim=0)

        if self._mean is None:
            self._mean = mean_b.clone()
            self._m2 = m2_b.clone()
            self._count = n_b
            return
        if d != self._mean.numel():
            raise ValueError(
                f"feature dimension {d} does not match existing {self._mean.numel()}"
            )

        n_a = self._count
        n_ab = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / n_ab)
        self._m2 = self._m2 + m2_b + delta.square() * n_a * n_b / n_ab
        self._count = n_ab

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> torch.Tensor:
        if self._mean is None or self._count == 0:
            raise RuntimeError("No samples; call update() first")
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        if self._mean is None or self._count == 0:
            raise RuntimeError("No samples; call update() first")
        if self._count < 2:
            return torch.zeros_like(self._mean)
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> torch.Tensor:
        if self._mean is None or self._count == 0:
            raise RuntimeError("No samples; call update() first")
        if self._count < 2:
            return torch.ones_like(self._mean)
        v = self._m2 / (self._count - 1)
        return torch.sqrt(v.clamp_min(self.eps * self.eps))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self.std.to(device=x.device, dtype=x.dtype)
        return (x - m) / s.clamp_min(self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self.std.to(device=x.device, dtype=x.dtype)
        return x * s + m

