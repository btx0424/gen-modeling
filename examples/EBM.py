"""
Minimal Energy-Based Model toy demo inspired by EBT.

This file is for synthetic toy datasets only. MNIST training lives in `EBM_mnist.py`.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.synthetic import (
    GaussianMixtureDataset,
    MoonsDataset,
    SwissRollDataset,
    SyntheticAmbientDataset,
)
from gen_modeling.utils.running_stats import RunningNormalizationStats


@dataclass
class Config:
    data_type: Literal["swiss_roll", "moons", "gaussian_mixture"] = "gaussian_mixture"
    ambient_dim: int = 32
    n_points: int = 2**16
    batch_size: int = 1024
    hidden_dim: int = 512
    num_threads: int = 1
    seed: int = 42
    train_steps: int = 4000
    lr: float = 5e-4
    ema_decay: float = 0.999
    normalize: bool = True
    noise_std: float = 0.75
    mcmc_steps: int = 4
    mcmc_step_size: float = 0.15
    langevin_noise_std: float = 0.0
    truncate_mcmc: bool = False
    sample_steps: int = 200
    sample_step_size: float = 0.02
    sample_langevin_noise_std: float = 0.0
    sample_init_std: float = 1.0
    num_plot_samples: int = 2000
    metrics_num_points: int = 2000
    metrics_num_projections: int = 128
    metrics_k: int = 5


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
        dataset = SwissRollDataset(noise=0.05, **common)
    elif config.data_type == "moons":
        dataset = MoonsDataset(noise=0.05, **common)
    elif config.data_type == "gaussian_mixture":
        dataset = GaussianMixtureDataset(scale_range=(0.04, 0.12), **common)
    else:
        raise ValueError(f"unknown data_type: {config.data_type}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return DataBundle(dataset.data, dataloader, dataset)


class EnergyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.sample_shape = (input_dim,)
        act = activation
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
        )
        self.energy_head = nn.Linear(hidden_dim, 1, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.energy_head.weight, mean=0.0, std=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_head(self.backbone(x)).squeeze(-1)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


@torch.no_grad()
def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).unsqueeze(0)
    return torch.clamp(x2 + y2 - 2 * (x @ y.T), min=0.0)


@torch.no_grad()
def intrinsic_metrics(
    data_z: torch.Tensor,
    samples_z: torch.Tensor,
    *,
    num_projections: int,
    k: int,
) -> dict[str, float]:
    d_ss = pairwise_sq_dists(samples_z, samples_z)
    d_dd = pairwise_sq_dists(data_z, data_z)
    d_sd = pairwise_sq_dists(samples_z, data_z)
    d_ds = d_sd.T
    d_ss.diagonal().fill_(float("inf"))
    d_dd.diagonal().fill_(float("inf"))

    k_eff_s = min(k, max(d_ss.shape[1] - 1, 1))
    k_eff_d = min(k, max(d_dd.shape[1] - 1, 1))
    sample_radius = d_ss.topk(k_eff_s, largest=False).values[:, -1].sqrt()
    data_radius = d_dd.topk(k_eff_d, largest=False).values[:, -1].sqrt()
    sample_to_data = d_sd.min(dim=1).values.sqrt()
    data_to_sample = d_ds.min(dim=1).values.sqrt()

    proj = torch.randn(num_projections, data_z.shape[1], device=data_z.device, dtype=data_z.dtype)
    proj = proj / torch.linalg.norm(proj, dim=1, keepdim=True).clamp_min(1e-12)
    data_proj = torch.sort(data_z @ proj.T, dim=0).values
    samp_proj = torch.sort(samples_z @ proj.T, dim=0).values

    return {
        "precision": (sample_to_data <= data_radius.mean()).float().mean().item(),
        "coverage": (data_to_sample <= sample_radius.mean()).float().mean().item(),
        "chamfer_l2": (sample_to_data.mean() + data_to_sample.mean()).item(),
        "swd": (data_proj - samp_proj).abs().mean().item(),
    }


class ToyEBM(nn.Module):
    def __init__(self, energy_model: nn.Module):
        super().__init__()
        self.energy_model = energy_model

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_model(x)

    def refine(
        self,
        x_init: torch.Tensor,
        *,
        num_steps: int,
        step_size: float,
        langevin_noise_std: float,
        learning: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        xs: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        x = x_init.clone().detach()
        for _ in range(num_steps):
            x = x.detach().requires_grad_(True)
            if langevin_noise_std > 0:
                x = x + torch.randn_like(x) * langevin_noise_std
            energy = self.energy(x)
            grad = torch.autograd.grad(energy.sum(), x, create_graph=learning)[0]
            x = x - step_size * grad
            xs.append(x)
            energies.append(energy)
        return xs, energies

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        *,
        noise_std: float,
        num_steps: int,
        step_size: float,
        langevin_noise_std: float,
        truncate: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_noisy = x_clean + torch.randn_like(x_clean) * noise_std
        refined, energies = self.refine(
            x_noisy,
            num_steps=num_steps,
            step_size=step_size,
            langevin_noise_std=langevin_noise_std,
            learning=True,
        )
        loss = F.smooth_l1_loss(refined[-1], x_clean) if truncate else sum(
            F.smooth_l1_loss(x_step, x_clean) for x_step in refined
        ) / len(refined)
        metrics = {
            "loss": loss.detach(),
            "initial_rec": F.smooth_l1_loss(refined[0], x_clean).detach(),
            "final_rec": F.smooth_l1_loss(refined[-1], x_clean).detach(),
            "initial_energy": energies[0].mean().detach(),
            "final_energy": energies[-1].mean().detach(),
        }
        return loss, metrics

    @torch.inference_mode(False)
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        *,
        num_steps: int,
        step_size: float,
        langevin_noise_std: float,
        init_std: float,
    ) -> torch.Tensor:
        self.eval()
        dtype = next(self.parameters()).dtype
        x = torch.randn((num_samples,) + self.energy_model.sample_shape, device=device, dtype=dtype) * init_std
        refined, _ = self.refine(
            x,
            num_steps=num_steps,
            step_size=step_size,
            langevin_noise_std=langevin_noise_std,
            learning=False,
        )
        return refined[-1].detach()


def _plot_intrinsic_comparison(data_z: torch.Tensor, samples_z: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d_intr = data_z.shape[1]
    if d_intr == 2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(data_z[:, 0].cpu(), data_z[:, 1].cpu(), s=2, alpha=0.25, c="C0")
        axes[1].scatter(samples_z[:, 0].cpu(), samples_z[:, 1].cpu(), s=2, alpha=0.25, c="C1")
        axes[0].set_title("Data")
        axes[1].set_title("EBM samples")
        for ax in axes:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(r"$z_1$")
            ax.set_ylabel(r"$z_2$")
    elif d_intr == 3:
        d_np = data_z.detach().cpu().numpy()
        s_np = samples_z.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 5.5))
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        ax0.scatter(d_np[:, 0], d_np[:, 1], d_np[:, 2], s=4, c="black", alpha=0.15, depthshade=False)
        ax1.scatter(s_np[:, 0], s_np[:, 1], s_np[:, 2], s=4, c="C1", alpha=0.35, depthshade=False)
        ax0.set_title("Data")
        ax1.set_title("EBM samples")
    else:
        raise ValueError(f"plotting supports intrinsic dim 2 or 3, got {d_intr}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy EBM for synthetic datasets.")
    parser.add_argument("--data-type", choices=["swiss_roll", "moons", "gaussian_mixture"], default=Config.data_type)
    parser.add_argument("--ambient-dim", type=int, default=Config.ambient_dim)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--noise-std", type=float, default=Config.noise_std)
    parser.add_argument("--mcmc-steps", type=int, default=Config.mcmc_steps)
    parser.add_argument("--mcmc-step-size", type=float, default=Config.mcmc_step_size)
    parser.add_argument("--langevin-noise-std", type=float, default=Config.langevin_noise_std)
    parser.add_argument("--truncate-mcmc", action="store_true", default=Config.truncate_mcmc)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--sample-step-size", type=float, default=Config.sample_step_size)
    parser.add_argument("--sample-langevin-noise-std", type=float, default=Config.sample_langevin_noise_std)
    parser.add_argument("--sample-init-std", type=float, default=Config.sample_init_std)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--metrics-num-points", type=int, default=Config.metrics_num_points)
    parser.add_argument("--metrics-num-projections", type=int, default=Config.metrics_num_projections)
    parser.add_argument("--metrics-k", type=int, default=Config.metrics_k)
    parser.add_argument("--no-normalize", action="store_true", default=not Config.normalize)
    args = parser.parse_args()

    config = Config(
        data_type=args.data_type,
        ambient_dim=args.ambient_dim,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_threads=args.num_threads,
        seed=args.seed,
        train_steps=args.train_steps,
        lr=args.lr,
        normalize=not args.no_normalize,
        noise_std=args.noise_std,
        mcmc_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        langevin_noise_std=args.langevin_noise_std,
        truncate_mcmc=args.truncate_mcmc,
        sample_steps=args.sample_steps,
        sample_step_size=args.sample_step_size,
        sample_langevin_noise_std=args.sample_langevin_noise_std,
        sample_init_std=args.sample_init_std,
        num_plot_samples=args.num_plot_samples,
        metrics_num_points=args.metrics_num_points,
        metrics_num_projections=args.metrics_num_projections,
        metrics_k=args.metrics_k,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    bundle = prepare_data(config, device)
    stats: RunningNormalizationStats | None = None
    if config.normalize:
        stats = RunningNormalizationStats()
        stats.update(bundle.data_ambient.cpu())

    model = ToyEBM(EnergyMLP(config.ambient_dim, config.hidden_dim)).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    pbar = tqdm(range(config.train_steps), desc="train")
    data_iter = cycle(bundle.dataloader)
    for step in pbar:
        x, _ = next(data_iter)
        x = x.to(device)
        x_model = stats.normalize(x) if stats is not None else x
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.compute_loss(
            x_model,
            noise_std=config.noise_std,
            num_steps=config.mcmc_steps,
            step_size=config.mcmc_step_size,
            langevin_noise_std=config.langevin_noise_std,
            truncate=config.truncate_mcmc,
        )
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model, config.ema_decay)
        if step % 100 == 0 or step == config.train_steps - 1:
            pbar.set_postfix(
                loss=f"{metrics['loss'].item():.5f}",
                init=f"{metrics['initial_rec'].item():.5f}",
                final=f"{metrics['final_rec'].item():.5f}",
                dE=f"{(metrics['initial_energy'] - metrics['final_energy']).item():.5f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.5f}")

    n_plot = min(config.num_plot_samples, config.n_points)
    samples = ema_model.sample(
        num_samples=n_plot,
        device=device,
        num_steps=config.sample_steps,
        step_size=config.sample_step_size,
        langevin_noise_std=config.sample_langevin_noise_std,
        init_std=config.sample_init_std,
    )
    if stats is not None:
        samples = stats.unnormalize(samples)
    data_z = bundle.dataset.unproject(bundle.data_ambient[:n_plot])
    samples_z = bundle.dataset.unproject(samples)
    out = Path(__file__).resolve().parent / "outputs" / "ebm_intrinsic.png"
    _plot_intrinsic_comparison(data_z, samples_z, out)

    n_metrics = min(config.metrics_num_points, config.n_points, samples.shape[0])
    metrics = intrinsic_metrics(
        bundle.dataset.unproject(bundle.data_ambient[:n_metrics]),
        bundle.dataset.unproject(samples[:n_metrics]),
        num_projections=config.metrics_num_projections,
        k=config.metrics_k,
    )
    metrics_path = Path(__file__).resolve().parent / "outputs" / "ebm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
