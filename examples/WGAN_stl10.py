"""
Class-conditional WGAN-GP on STL-10.
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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_modeling.datasets.images import ImageDatasetInfo, STL10Dataset, tensor_batch_to_display
from gen_modeling.modules.cnn import (
    Downsample2d,
    PixelShuffleUpsample2d,
    ResidualBlock2d,
    init_conv_modules,
)


@dataclass
class Config:
    data_root: str = "./data"
    split: str = "train"
    batch_size: int = 64
    base_channels: int = 64
    latent_dim: int = 256
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.99
    ema_decay: float = 0.999
    critic_steps: int = 5
    gp_lambda: float = 10.0
    num_plot_samples: int = 40


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pre = ResidualBlock2d(in_channels, out_channels)
        self.up = PixelShuffleUpsample2d(out_channels)
        self.post = ResidualBlock2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.up(x)
        return self.post(x)


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        base_channels: int,
        num_classes: int,
        *,
        data_info: ImageDatasetInfo,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        upsample_factor = 2 ** 3
        self._seed_h = data_info.height // upsample_factor
        self._seed_w = data_info.width // upsample_factor
        self.to_seed = nn.Linear(latent_dim * 2, base_channels * 4 * self._seed_h * self._seed_w)
        self.up1 = GeneratorBlock(base_channels * 4, base_channels * 2)
        self.up2 = GeneratorBlock(base_channels * 2, base_channels)
        self.up3 = GeneratorBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, data_info.channels, kernel_size=3, padding=1)
        self.latent_dim = latent_dim
        self.sample_shape = (data_info.channels, data_info.height, data_info.width)
        init_conv_modules(self)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([z, self.label_embedding(y)], dim=-1)
        h = self.to_seed(cond).reshape(z.shape[0], -1, self._seed_h, self._seed_w)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return torch.tanh(self.out(h))


class CriticBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock2d(in_channels, out_channels)
        self.down = Downsample2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.block(x))


class ProjectionCritic(nn.Module):
    def __init__(self, base_channels: int, num_classes: int, *, data_info: ImageDatasetInfo) -> None:
        super().__init__()
        in_channels = data_info.channels
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = CriticBlock(base_channels, base_channels)
        self.down2 = CriticBlock(base_channels, base_channels * 2)
        self.down3 = CriticBlock(base_channels * 2, base_channels * 4)
        self.mid = ResidualBlock2d(base_channels * 4, base_channels * 4)
        self.score_head = nn.Linear(base_channels * 4, 1)
        self.class_embed = nn.Embedding(num_classes, base_channels * 4)
        init_conv_modules(self)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        return h.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        proj = (feat * self.class_embed(y)).sum(dim=-1, keepdim=True)
        return self.score_head(feat) + proj


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


def gradient_penalty(
    critic: ProjectionCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=real.dtype)
    assert real.shape == fake.shape, f"Real shape: {real.shape}, fake shape: {fake.shape}"
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(interpolated, labels)
    grad = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(grad.shape[0], -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def make_eval_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_classes, device=device, dtype=torch.long)
    repeats = (num_samples + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:num_samples]


def plot_stl10_grid(
    samples: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    *,
    num_show: int,
    num_classes: int,
    data_info: ImageDatasetInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = tensor_batch_to_display(samples[:num_show], data_info)
    labels = labels[:num_show].detach().cpu()
    n_cols = num_classes
    n_rows = int(np.ceil(samples.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, (ax, image) in enumerate(zip(axes, samples, strict=False)):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(str(int(labels[idx].item())), fontsize=8, pad=1)
    for ax in axes[samples.shape[0]:]:
        ax.axis("off")
    plt.tight_layout(pad=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def sample_and_save(
    generator: ConditionalGenerator,
    num_classes: int,
    config: Config,
    device: torch.device,
    out_path: Path,
    metrics_path: Path | None = None,
    *,
    data_info: ImageDatasetInfo,
) -> dict[str, float]:
    labels = make_eval_labels(config.num_plot_samples, num_classes, device)
    z = torch.randn(labels.shape[0], config.latent_dim, device=device)
    samples = generator(z, labels)
    plot_stl10_grid(
        samples,
        labels,
        out_path,
        num_show=config.num_plot_samples,
        num_classes=num_classes,
        data_info=data_info,
    )
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
    parser = argparse.ArgumentParser(description="Conditional WGAN-GP on STL-10.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--split", type=str, default=Config.split)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--beta1", type=float, default=Config.beta1)
    parser.add_argument("--beta2", type=float, default=Config.beta2)
    parser.add_argument("--ema-decay", type=float, default=Config.ema_decay)
    parser.add_argument("--critic-steps", type=int, default=Config.critic_steps)
    parser.add_argument("--gp-lambda", type=float, default=Config.gp_lambda)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        ema_decay=args.ema_decay,
        critic_steps=args.critic_steps,
        gp_lambda=args.gp_lambda,
        num_plot_samples=args.num_plot_samples,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = STL10Dataset(config.data_root, split=config.split, download=True, normalize=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    generator = ConditionalGenerator(
        config.latent_dim,
        config.base_channels,
        dataset.info.num_classes,
        data_info=dataset.info,
    ).to(device)
    generator_ema = deepcopy(generator).to(device)
    generator_ema.eval()
    critic = ProjectionCritic(
        config.base_channels,
        dataset.info.num_classes,
        data_info=dataset.info,
    ).to(device)

    g_opt = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    d_opt = optim.Adam(critic.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    out_dir = Path(__file__).resolve().parent / "outputs" / "WGAN_stl10"

    for epoch in range(config.train_epochs):
        generator.train()
        critic.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        last_g_loss = None
        last_d_loss = None
        last_wdist = None
        last_gp = None
        for step, (real, labels) in enumerate(pbar):
            real = real.to(device)
            labels = labels.to(device)

            z = torch.randn(real.shape[0], config.latent_dim, device=device)
            fake = generator(z, labels)

            d_opt.zero_grad(set_to_none=True)
            real_score = critic(real, labels)
            fake_score = critic(fake.detach(), labels)
            gp = gradient_penalty(critic, real, fake.detach(), labels)
            d_loss = fake_score.mean() - real_score.mean() + config.gp_lambda * gp
            d_loss.backward()
            d_opt.step()

            last_d_loss = d_loss.detach()
            last_wdist = (real_score.mean() - fake_score.mean()).detach()
            last_gp = gp.detach()

            if (step + 1) % config.critic_steps == 0:
                g_opt.zero_grad(set_to_none=True)
                z = torch.randn(real.shape[0], config.latent_dim, device=device)
                fake = generator(z, labels)
                g_loss = -critic(fake, labels).mean()
                g_loss.backward()
                g_opt.step()
                update_ema(generator_ema, generator, config.ema_decay)
                last_g_loss = g_loss.detach()

            pbar.set_postfix(
                d_loss=f"{last_d_loss.item():.5f}" if last_d_loss is not None else "nan",
                wdist=f"{last_wdist.item():.5f}" if last_wdist is not None else "nan",
                gp=f"{last_gp.item():.5f}" if last_gp is not None else "nan",
                g_loss=f"{last_g_loss.item():.5f}" if last_g_loss is not None else "nan",
            )

        epoch_grid = out_dir / f"wgan_stl10_epoch_{epoch:03d}.png"
        epoch_metrics = out_dir / f"wgan_stl10_epoch_{epoch:03d}.json"
        metrics = sample_and_save(
            generator_ema,
            dataset.info.num_classes,
            config,
            device,
            epoch_grid,
            epoch_metrics,
            data_info=dataset.info,
        )
        print(
            f"epoch {epoch}: "
            f"d_loss={last_d_loss.item() if last_d_loss is not None else float('nan'):.6f}, "
            f"wdist={last_wdist.item() if last_wdist is not None else float('nan'):.6f}, "
            f"gp={last_gp.item() if last_gp is not None else float('nan'):.6f}, "
            f"g_loss={last_g_loss.item() if last_g_loss is not None else float('nan'):.6f}, "
            f"sample_std={metrics['sample_std']:.6f}"
        )

    out = out_dir / "wgan_stl10_samples.png"
    metrics_path = out_dir / "wgan_metrics.json"
    metrics = sample_and_save(
        generator_ema,
        dataset.info.num_classes,
        config,
        device,
        out,
        metrics_path,
        data_info=dataset.info,
    )
    print(json.dumps(metrics, indent=2))
    print("Saved per-epoch STL-10 samples and final metrics under examples/outputs")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
