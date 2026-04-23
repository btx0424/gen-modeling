from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
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
from gen_modeling.flow_matching import LossType, ModelArch, PredictionType, PredictionWrapper

torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    ambient_dim: int = 32
    n_points: int = 8192
    batch_size: int = 1024
    data_type: Literal[
        "swiss_roll", "moons", "gaussian_mixture", "checkerboard"
    ] = "gaussian_mixture"
    loss_type: LossType = "x"
    train_steps: int = 500
    hidden_dim: int = 256
    t_eps: float = 1e-2
    sample_steps: int = 50
    output_path: str = "assets/residual.png"
    experiment_output_dir: str = "assets/residual_experiments"


@dataclass(frozen=True)
class DataBundle:
    data_ambient: torch.Tensor
    dataloader: DataLoader
    dataset: SyntheticAmbientDataset


@dataclass
class ExperimentResult:
    model: nn.Module
    losses: list[float]
    samples_ambient: torch.Tensor


DEFAULT_EXPERIMENTS: list[tuple[ModelArch, PredictionType]] = [
    ("vanilla", "x"),
    ("vanilla", "eps"),
    ("vanilla", "v"),
    ("global_residual", "x"),
    ("global_residual", "eps"),
    ("global_residual", "v"),
    ("corrected_residual2", "x"),
    ("corrected_residual2", "eps"),
    ("corrected_residual2", "v"),
]


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
    elif config.data_type == "checkerboard":
        dataset = CheckerboardDataset(noise=2.0, jitter=0.03, **common)
    else:
        raise ValueError(f"unknown data_type: {config.data_type}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    return DataBundle(
        data_ambient=dataset.data,
        dataloader=dataloader,
        dataset=dataset,
    )


class DenoisingMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        self.sample_shape = (dim,)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 20),
            activation(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + 20, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = cond
        t_emb = self.time_encoder(t)
        return self.net(torch.cat((x_t, t_emb), dim=-1))


def build_model(
    model_arch: ModelArch,
    pred_type: PredictionType,
    ambient_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> nn.Module:
    base_network = DenoisingMLP(ambient_dim, hidden_dim)
    return PredictionWrapper(base_network, pred_type, model_arch).to(device)


def compute_loss(
    loss_type: LossType,
    x1: torch.Tensor,
    eps: torch.Tensor,
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x1_hat, v_hat, eps_hat = predictions
    if loss_type == "x":
        return ((x1_hat - x1) ** 2).mean()
    if loss_type == "eps":
        return ((eps_hat - eps) ** 2).mean()
    if loss_type == "v":
        v_target = x1 - eps
        return ((v_hat - v_target) ** 2).mean()
    raise ValueError(f"Invalid loss type: {loss_type}")


def train_model(
    config: Config,
    data_bundle: DataBundle,
    device: torch.device,
    model_arch: ModelArch,
    pred_type: PredictionType,
) -> tuple[nn.Module, list[float]]:
    model = build_model(model_arch, pred_type, config.ambient_dim, config.hidden_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []

    @torch.compile(mode="max-autotune", disable=True)
    def train_step(x1: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(x1)
        t = torch.rand((x1.shape[0], 1), device=device).clip(config.t_eps, 1.0 - config.t_eps)
        x_t = t * x1 + (1.0 - t) * eps
        predictions = model(x_t, t)
        loss = compute_loss(config.loss_type, x1, eps, predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    print(f"Training {pred_type}-prediction {model_arch} model with {config.loss_type} loss...")
    progress = tqdm(range(config.train_steps))
    for _ in progress:
        for x1, _ in data_bundle.dataloader:
            x1 = x1.to(device, non_blocking=True)
            loss = train_step(x1)
            losses.append(loss.detach().item())
            progress.set_postfix(loss=loss.item())

    return model, losses


@torch.no_grad()
def sample_model(
    model: nn.Module,
    config: Config,
    device: torch.device,
    num_samples: int,
) -> torch.Tensor:
    x_t = torch.randn(num_samples, config.ambient_dim, device=device)
    ts = torch.linspace(config.t_eps, 1.0 - config.t_eps, config.sample_steps, device=device)
    dt = ts[1] - ts[0] if config.sample_steps > 1 else torch.tensor(1.0 - 2 * config.t_eps, device=device)
    for t_scalar in ts:
        t = torch.full((num_samples, 1), t_scalar.item(), device=device)
        _, v_hat, _ = model(x_t, t)
        x_t = x_t + v_hat * dt
    return x_t


def plot_intrinsic_scatter(
    ax: plt.Axes,
    data_intrinsic: np.ndarray,
    samples_intrinsic: np.ndarray | None = None,
    *,
    title: str | None = None,
) -> None:
    dim = data_intrinsic.shape[1]
    if dim == 2:
        ax.scatter(data_intrinsic[:, 0], data_intrinsic[:, 1], s=5, c="black", alpha=0.2, label="data")
        if samples_intrinsic is not None:
            ax.scatter(samples_intrinsic[:, 0], samples_intrinsic[:, 1], s=5, alpha=0.35, label="generated")
        ax.set_xlim(-2.75, 2.75)
        ax.set_ylim(-2.75, 2.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
    elif dim == 3:
        ax.scatter(
            data_intrinsic[:, 0],
            data_intrinsic[:, 1],
            data_intrinsic[:, 2],
            s=4,
            c="black",
            alpha=0.15,
            label="data",
            depthshade=False,
        )
        if samples_intrinsic is not None:
            ax.scatter(
                samples_intrinsic[:, 0],
                samples_intrinsic[:, 1],
                samples_intrinsic[:, 2],
                s=4,
                alpha=0.35,
                label="generated",
                depthshade=False,
            )
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_zlabel(r"$z_3$")
    else:
        raise ValueError(f"Plotting supports intrinsic dim 2 or 3, got {dim}")

    if title is not None:
        ax.set_title(title)
    if samples_intrinsic is not None:
        ax.legend(loc="best")


def plot_data(dataset: SyntheticAmbientDataset) -> None:
    data_intrinsic = dataset.unproject(dataset.data).detach().cpu().numpy()
    dim = dataset.intrinsic_dim
    fig = plt.figure(figsize=(7.5, 6.5) if dim == 3 else (6, 6))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    plot_intrinsic_scatter(ax, data_intrinsic, title="Ground Truth")
    plt.tight_layout()
    plt.show()


def plot_training_curves(results: dict[tuple[ModelArch, PredictionType], ExperimentResult]) -> None:
    plt.figure(figsize=(10, 6))
    for (model_arch, pred_type), result in results.items():
        plt.plot(result.losses, label=f"{model_arch} - {pred_type}")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True)
    plt.show()


def save_experiment_visualizations(
    config: Config,
    data_bundle: DataBundle,
    results: dict[tuple[ModelArch, PredictionType], ExperimentResult],
) -> None:
    output_dir = Path(config.experiment_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_intrinsic = data_bundle.dataset.unproject(data_bundle.data_ambient).detach().cpu().numpy()

    for (model_arch, pred_type), result in results.items():
        samples_intrinsic = data_bundle.dataset.unproject(result.samples_ambient).detach().cpu().numpy()
        dim = data_bundle.dataset.intrinsic_dim
        fig = plt.figure(figsize=(7.5, 6.5) if dim == 3 else (6.5, 6.5))
        ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
        plot_intrinsic_scatter(
            ax,
            data_intrinsic,
            samples_intrinsic,
            title=f"{model_arch} - {pred_type} ({config.loss_type} loss)",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_arch}_{pred_type}_{config.loss_type}.png", dpi=150)
        plt.close(fig)


def plot_samples(
    config: Config,
    data_bundle: DataBundle,
    results: dict[tuple[ModelArch, PredictionType], ExperimentResult],
    experiments: list[tuple[ModelArch, PredictionType]],
) -> None:
    x_preds = {
        key: data_bundle.dataset.unproject(result.samples_ambient).detach().cpu().numpy()
        for key, result in results.items()
    }
    data_intrinsic = data_bundle.dataset.unproject(data_bundle.data_ambient).detach().cpu().numpy()
    model_arches = list(dict.fromkeys([arch for arch, _ in experiments]))
    cols = ["original", "x", "eps", "v"]
    dim = data_bundle.dataset.intrinsic_dim
    fig = plt.figure(figsize=(24, 6 * len(model_arches) if dim == 3 else 5 * len(model_arches)))
    axes: list[list[plt.Axes]] = []
    for row_idx in range(len(model_arches)):
        row_axes: list[plt.Axes] = []
        for col_idx in range(len(cols)):
            subplot_index = row_idx * len(cols) + col_idx + 1
            row_axes.append(fig.add_subplot(len(model_arches), len(cols), subplot_index, projection="3d" if dim == 3 else None))
        axes.append(row_axes)

    for row_idx, model_arch in enumerate(model_arches):
        plot_intrinsic_scatter(axes[row_idx][0], data_intrinsic, title=f"{model_arch} - Ground Truth")
        for col_idx, pred_type in enumerate(["x", "eps", "v"], start=1):
            key = (model_arch, pred_type)
            if key not in x_preds:
                axes[row_idx][col_idx].set_axis_off()
                continue
            plot_intrinsic_scatter(
                axes[row_idx][col_idx],
                data_intrinsic,
                x_preds[key],
                title=f"{model_arch} - {pred_type}",
            )

    plt.tight_layout()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def main() -> None:
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Running on {device} with Ambient Dimension D={config.ambient_dim}")

    data_bundle = prepare_data(config, device)
    plot_data(data_bundle.dataset)

    results: dict[tuple[ModelArch, PredictionType], ExperimentResult] = {}
    for model_arch, pred_type in DEFAULT_EXPERIMENTS:
        model, losses = train_model(config, data_bundle, device, model_arch, pred_type)
        samples_ambient = sample_model(model, config, device, num_samples=config.n_points)
        results[(model_arch, pred_type)] = ExperimentResult(
            model=model,
            losses=losses,
            samples_ambient=samples_ambient,
        )
    
    plot_training_curves(results)
    save_experiment_visualizations(config, data_bundle, results)
    plot_samples(config, data_bundle, results, DEFAULT_EXPERIMENTS)


if __name__ == "__main__":
    main()
