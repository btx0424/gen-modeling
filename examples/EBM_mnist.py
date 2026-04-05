"""
Minimal Energy-Based Model on MNIST.

This file is for image training only. Synthetic toy training lives in `EBM.py`.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.images import ImageDatasetInfo, MNISTDataset
from gen_modeling.modules import SmallConvNet


@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 128
    hidden_channels: int = 64
    num_blocks: int = 4
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 10
    lr: float = 5e-4
    ema_decay: float = 0.999
    noise_std: float = 0.75
    mcmc_steps: int = 4
    mcmc_step_size: float = 0.15
    langevin_noise_std: float = 0.0
    truncate_mcmc: bool = False
    sample_steps: int = 200
    sample_step_size: float = 0.02
    sample_langevin_noise_std: float = 0.0
    sample_init_std: float = 1.0
    num_plot_samples: int = 100


class EnergyCNN(nn.Module):
    def __init__(self, hidden_channels: int = 64, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        self.sample_shape = (1, 28, 28)
        self.backbone = SmallConvNet(1, hidden_channels, hidden_channels, num_blocks)
        self.label_embedding = nn.Embedding(num_classes, hidden_channels)
        self.energy_head = nn.Linear(hidden_channels, 1, bias=False)
        nn.init.normal_(self.label_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.energy_head.weight, mean=0.0, std=1e-2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = feats.mean(dim=(2, 3))
        cond = pooled + self.label_embedding(y)
        return self.energy_head(cond).squeeze(-1)


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


class EBM(nn.Module):
    def __init__(self, energy_model: nn.Module):
        super().__init__()
        self.energy_model = energy_model

    def refine(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
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
            energy = self.energy_model(x, y)
            grad = torch.autograd.grad(energy.sum(), x, create_graph=learning)[0]
            x = x - step_size * grad
            xs.append(x)
            energies.append(energy)
        return xs, energies

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        y: torch.Tensor,
        config: Config,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_noisy = x_clean + torch.randn_like(x_clean) * config.noise_std
        refined, energies = self.refine(
            x_noisy,
            y,
            num_steps=config.mcmc_steps,
            step_size=config.mcmc_step_size,
            langevin_noise_std=config.langevin_noise_std,
            learning=True,
        )
        loss = F.smooth_l1_loss(refined[-1], x_clean) if config.truncate_mcmc else sum(
            F.smooth_l1_loss(x_step, x_clean) for x_step in refined
        ) / len(refined)
        return loss, {
            "loss": loss.detach(),
            "initial_rec": F.smooth_l1_loss(refined[0], x_clean).detach(),
            "final_rec": F.smooth_l1_loss(refined[-1], x_clean).detach(),
            "initial_energy": energies[0].mean().detach(),
            "final_energy": energies[-1].mean().detach(),
        }

    @torch.inference_mode(False)
    def sample(self, labels: torch.Tensor, device: torch.device, config: Config) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        num_samples = labels.shape[0]
        x = torch.randn((num_samples,) + self.energy_model.sample_shape, device=device, dtype=dtype)
        x = x * config.sample_init_std
        refined, _ = self.refine(
            x,
            labels,
            num_steps=config.sample_steps,
            step_size=config.sample_step_size,
            langevin_noise_std=config.sample_langevin_noise_std,
            learning=False,
        )
        return refined[-1].detach()


def make_eval_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_classes, device=device, dtype=torch.long)
    repeats = (num_samples + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:num_samples]


def plot_mnist_grid(
    samples: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    *,
    num_show: int = 100,
        num_classes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = samples[:num_show].detach().cpu().clamp(-1, 1)
    labels = labels[:num_show].detach().cpu()
    n_cols = num_classes
    n_rows = int(np.ceil(samples.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, (ax, image) in enumerate(zip(axes, samples, strict=False)):
        ax.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(str(int(labels[idx].item())), fontsize=8, pad=1)
    for ax in axes[samples.shape[0]:]:
        ax.axis("off")
    plt.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_and_save(
    model: EBM,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    num_classes = data_info.num_classes
    labels = make_eval_labels(config.num_plot_samples, num_classes, device)
    samples = model.sample(labels, device, config)
    plot_mnist_grid(samples, labels, out_path, num_show=config.num_plot_samples, num_classes=num_classes)
    metrics = {
        "sample_mean": samples.mean().item(),
        "sample_std": samples.std().item(),
        "sample_min": samples.min().item(),
        "sample_max": samples.max().item(),
    }
    if metrics_path is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST EBM.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-channels", type=int, default=Config.hidden_channels)
    parser.add_argument("--num-blocks", type=int, default=Config.num_blocks)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
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
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
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
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = MNISTDataset(config.data_root, train=True, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = EBM(EnergyCNN(config.hidden_channels, config.num_blocks, dataset.info.num_classes)).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    out_dir = Path(__file__).resolve().parent / "outputs" / "EBM_mnist"

    for epoch in range(config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        last_metrics: dict[str, torch.Tensor] | None = None
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.compute_loss(x, y, config)
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)
            last_metrics = metrics
            pbar.set_postfix(
                loss=f"{metrics['loss'].item():.5f}",
                init=f"{metrics['initial_rec'].item():.5f}",
                final=f"{metrics['final_rec'].item():.5f}",
                dE=f"{(metrics['initial_energy'] - metrics['final_energy']).item():.5f}",
            )
        epoch_grid = out_dir / f"ebm_mnist_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"ebm_mnist_epoch_{epoch:03d}.json"
        metrics = sample_and_save(
            ema_model, config, device, epoch_grid, epoch_metrics, data_info=dataset.info
        )
        if last_metrics is not None:
            print(
                f"epoch {epoch}: "
                f"loss={last_metrics['loss'].item():.6f}, "
                f"final_rec={last_metrics['final_rec'].item():.6f}, "
                f"sample_std={metrics['sample_std']:.6f}"
            )

    out = out_dir / "ebm_mnist_samples.png"
    metrics_path = out_dir / "ebm_metrics.json"
    metrics = sample_and_save(ema_model, config, device, out, metrics_path, data_info=dataset.info)
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch MNIST samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
