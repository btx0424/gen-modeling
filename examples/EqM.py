"""
Minimal Equilibrium Matching (EqM) toy demo on synthetic ambient datasets.

Training follows the same linear bridge as flow matching, but the target vector field
is scaled by ``grad_magnitude(t)`` (default ``1 - t``) and the backbone predicts a
time-independent field ``f(x_t)`` — only ``x_t`` is given as input.

Sampling mirrors the reference implementation's gradient-descent sampler: starting
from Gaussian noise, iterate ``x <- x + stepsize * f(x)`` for a fixed number of steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.synthetic import (
    CheckerboardDataset,
    GaussianMixtureDataset,
    MoonsDataset,
    SwissRollDataset,
    SyntheticAmbientDataset,
)


@dataclass
class Config:
    data_type: Literal[
        "swiss_roll", "moons", "gaussian_mixture", "checkerboard"
    ] = "moons"
    ambient_dim: int = 32
    n_points: int = 2**16
    batch_size: int = 1024
    hidden_dim: int = 512
    seed: int = 42
    train_steps: int = 10_000
    lr: float = 5e-4
    sample_steps: int = 100
    sample_stepsize: float = 0.01
    sample_sampler: Literal["gd", "nag"] = "nag"
    sample_mu: float = 0.3
    num_plot_samples: int = 4096


@dataclass(frozen=True)
class DataBundle:
    data_ambient: torch.Tensor
    dataloader: DataLoader
    dataset: SyntheticAmbientDataset


def prepare_data(config: Config, device: torch.device) -> DataBundle:
    common = dict(
        ambient_dim=config.ambient_dim,
        n_samples=config.n_points,
        device=str(device),
        random_state=config.seed,
    )
    if config.data_type == "swiss_roll":
        dataset = SwissRollDataset(noise=0.1, **common)
    elif config.data_type == "moons":
        dataset = MoonsDataset(noise=0.1, **common)
    elif config.data_type == "gaussian_mixture":
        dataset = GaussianMixtureDataset(scale_range=(0.04, 0.12), **common)
    elif config.data_type == "checkerboard":
        dataset = CheckerboardDataset(noise=2.0, jitter=0.03, **common)
    else:
        raise ValueError(f"unknown data_type: {config.data_type}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return DataBundle(
        data_ambient=dataset.data,
        dataloader=dataloader,
        dataset=dataset,
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.input_dim = input_dim
        act = activation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), act(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), act(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), act(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), act(), 
            nn.Linear(hidden_dim, input_dim),
        )
        # Apply orthogonal initialization to all Linear layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def eqm_ct(a: float = 0.8, grad_scale: float = 4.0):
    """Official EqM-style truncated decay with an overall gradient multiplier."""
    def func(t: torch.Tensor) -> torch.Tensor:
        return grad_scale * torch.where(t < a, 1.0, (1.0 - t) / (1.0 - a))
    return func


@torch.no_grad()
def compute_subspace_metrics(
    model: nn.Module,
    x1: torch.Tensor,
    q: torch.Tensor,
    grad_magnitude: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    expand_shape = (-1,) + (x1.ndim - 1) * (1,)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    t = t.reshape(expand_shape)
    x0 = torch.randn_like(x1)
    xt = t * x1 + (1.0 - t) * x0
    target = (x1 - x0) * grad_magnitude(t)
    pred = model(xt)

    err = pred - target
    q = q.to(device=x1.device, dtype=x1.dtype)
    intrinsic = (err @ q) @ q.T
    orthogonal = err - intrinsic
    return {
        "loss": err.square().mean().item(),
        "intrinsic_mse": intrinsic.square().mean().item(),
        "orthogonal_mse": orthogonal.square().mean().item(),
    }


class EqM(nn.Module):
    """
    Time-independent vector field trained to match a scaled flow-matching target.

    At each step, sample ``t``, ``x0``, build ``x_t = t x1 + (1-t) x0``, and regress
    ``network(x_t)`` toward ``(x1 - x0) * grad_magnitude(t)``.
    """

    def __init__(
        self,
        network: nn.Module,
        grad_magnitude: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.network = network
        self.grad_magnitude = grad_magnitude or eqm_ct()

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t = t.reshape(expand_shape)
        x0 = torch.randn_like(x1)
        xt = t * x1 + (1.0 - t) * x0
        target = (x1 - x0) * self.grad_magnitude(t)
        pred = self.network(xt)
        return ((pred - target) ** 2).mean()

    @torch.inference_mode()
    def sample_gd(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int = 500,
        stepsize: float = 0.02,
    ) -> torch.Tensor:
        """Gradient-descent style sampling: ``x <- x + stepsize * f(x)`` from noise."""
        self.eval()
        dtype = next(self.parameters()).dtype
        x = torch.randn(num_samples, self.network.input_dim, device=device, dtype=dtype)
        for _ in range(num_steps):
            x = x + stepsize * self.network(x)
        return x
    
    @torch.inference_mode()
    def sample_nag(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int = 500,
        stepsize: float = 0.02,
        mu: float = 0.3,
    ) -> torch.Tensor:
        self.eval()
        dtype = next(self.parameters()).dtype
        x = torch.randn(num_samples, self.network.input_dim, device=device, dtype=dtype)
        momentum = torch.zeros_like(x)
        for _ in range(num_steps):
            lookahead = x + stepsize * mu * momentum
            momentum = self.network(lookahead)
            x = x + stepsize * momentum
        return x


def _plot_intrinsic_comparison(
    data_z: torch.Tensor,
    samples_z: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d_intr = data_z.shape[1]
    if d_intr == 2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(
            data_z[:, 0].cpu(),
            data_z[:, 1].cpu(),
            s=2,
            alpha=0.25,
            c="C0",
        )
        axes[0].set_title("Data")
        axes[1].scatter(
            samples_z[:, 0].cpu(),
            samples_z[:, 1].cpu(),
            s=2,
            alpha=0.25,
            c="C1",
        )
        axes[1].set_title("EqM samples")
        for ax in axes:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(r"$z_1$")
            ax.set_ylabel(r"$z_2$")
    elif d_intr == 3:
        # Swiss roll in R^3, same layout as main.py plot_intrinsic_scatter (3D)
        d_np = data_z.detach().cpu().numpy()
        s_np = samples_z.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 5.5))
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        ax0.scatter(
            d_np[:, 0],
            d_np[:, 1],
            d_np[:, 2],
            s=4,
            c="black",
            alpha=0.15,
            depthshade=False,
        )
        ax0.set_title("Data")
        ax0.set_xlabel(r"$z_1$")
        ax0.set_ylabel(r"$z_2$")
        ax0.set_zlabel(r"$z_3$")
        ax1.scatter(
            s_np[:, 0],
            s_np[:, 1],
            s_np[:, 2],
            s=4,
            alpha=0.35,
            c="C1",
            depthshade=False,
        )
        ax1.set_title("EqM samples")
        ax1.set_xlabel(r"$z_1$")
        ax1.set_ylabel(r"$z_2$")
        ax1.set_zlabel(r"$z_3$")
    else:
        raise ValueError(f"Plotting supports intrinsic dim 2 or 3, got {d_intr}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    bundle = prepare_data(config, device)

    backbone = MLP(config.ambient_dim, config.hidden_dim).to(device)
    model = EqM(backbone, eqm_ct()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    pbar = tqdm(range(config.train_steps), desc="train")
    data_iter = cycle(bundle.dataloader)
    for step in pbar:
        x1, _ = next(data_iter)
        x1 = x1.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(x1)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == config.train_steps - 1:
            metrics = compute_subspace_metrics(
                model.network,
                x1,
                bundle.dataset.Q,
                model.grad_magnitude,
            )
            pbar.set_postfix(
                loss=f"{metrics['loss']:.5f}",
                intr=f"{metrics['intrinsic_mse']:.5f}",
                orth=f"{metrics['orthogonal_mse']:.5f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.5f}")

    n_plot = min(config.num_plot_samples, config.n_points)
    sample_fn = model.sample_nag if config.sample_sampler == "nag" else model.sample_gd
    samples = sample_fn(
        num_samples=n_plot,
        device=device,
        num_steps=config.sample_steps,
        stepsize=config.sample_stepsize,
        **({"mu": config.sample_mu} if config.sample_sampler == "nag" else {}),
    )
    data_z = bundle.dataset.unproject(bundle.data_ambient[:n_plot])
    samples_z = bundle.dataset.unproject(samples)
    out = Path(__file__).resolve().parent / "outputs" / "eqm_intrinsic.png"
    _plot_intrinsic_comparison(data_z, samples_z, out)
    final_metrics = compute_subspace_metrics(
        model.network,
        bundle.data_ambient[: min(config.batch_size, config.n_points)],
        bundle.dataset.Q,
        model.grad_magnitude,
    )
    print(
        "Final metrics: "
        f"loss={final_metrics['loss']:.6f}, "
        f"intrinsic_mse={final_metrics['intrinsic_mse']:.6f}, "
        f"orthogonal_mse={final_metrics['orthogonal_mse']:.6f}"
    )
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
