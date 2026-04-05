import argparse
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid, save_image
from typing import Literal
import matplotlib.pyplot as plt
import wandb
from gen_modeling.utils.running_stats import RunningNormalizationStats
from gen_modeling.datasets.synthetic import (
    CheckerboardDataset,
    MoonsDataset,
    SwissRollDataset,
    SyntheticAmbientDataset,
)


class NetworkND(nn.Module):
    """MLP velocity field v(x, t). Time is required: x_t alone does not fix the rectified-flow target."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
        )
        layers = [nn.Linear(input_dim + 32, hidden_dim)]
        for _ in range(num_layers):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] != x.shape[0]:
            raise ValueError("t and x must share batch size")
        t = self.time_encoder(t.reshape(-1, 1))
        out = self.net(torch.cat([x, t], dim=-1))
        return out


class GlobalResidual(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw_pred = self.network(x_t, t)
        return raw_pred + x_t


class CorrectedResidual1(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        raw_pred = self.network(x_t, t)
        t = t.reshape(expand_shape)
        return raw_pred + x_t / (1.0 - t)


class CorrectedResidual2(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        raw_pred = self.network(x_t, t)
        t = t.reshape(expand_shape)
        return (t * raw_pred + x_t) / (1.0 - t)


class VPrediction(nn.Module):
    """The network predicts v_hat."""
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        raw_pred = self.network(x_t, t)
        x1_hat = x_t + (1.0 - t).reshape(expand_shape) * raw_pred
        v_hat = raw_pred
        eps_hat = x_t - t * v_hat
        return x1_hat, v_hat, eps_hat
        

class XPrediction(nn.Module):
    """The network predicts x1_hat."""
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        t = t.reshape(expand_shape)
        raw_pred = self.network(x_t, t)
        x1_hat = raw_pred
        v_hat = (x1_hat - x_t) / (1.0 - t)
        eps_hat = x_t - t * v_hat
        return x1_hat, v_hat, eps_hat


class LinearFlow:
    def __init__(
        self,
        network: nn.Module,
        noise_scale: float = 1.0,
        pred_type: Literal["v", "x", "eps"] = "v",
        loss_type: Literal["v", "x", "eps"] = "v",
    ):
        self.pred_type = pred_type
        self.loss_type = loss_type
        if self.pred_type == "v":
            self.pred_module = VPrediction(network)
        elif self.pred_type == "x":
            self.pred_module = XPrediction(CorrectedResidual2(network))
        else:
            raise ValueError(f"Invalid prediction type: {self.pred_type}")
        self.sample_shape = (network.input_dim,)
        self.noise_scale = noise_scale

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int) -> torch.Tensor:
        """Integrate dx/dt = v(x, t) over the same clipped time range used in training."""
        device = next(self.pred_module.parameters()).device
        x = torch.randn((num_samples,) + self.sample_shape, device=device) * self.noise_scale
        eps_t = 1e-2
        ts = torch.linspace(eps_t, 1.0 - eps_t, num_steps, device=device)
        dt = ts[1] - ts[0] if num_steps > 1 else torch.tensor(1.0 - 2 * eps_t, device=device)
        for t_scalar in ts:
            t = torch.full((num_samples,), t_scalar.item(), device=device)
            _, v_hat, _ = self.pred_module(x, t)
            x = x + v_hat * dt
        return x

    def compute_loss(self, x_1: torch.Tensor):
        expand_shape = (-1,) + (x_1.ndim - 1) * (1,)
        t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
        t = t.clip(1e-2, 1 - 1e-2)
        t = t.reshape(expand_shape)
        x_0 = torch.randn_like(x_1) * self.noise_scale
        x_t = t * x_1 + (1.0 - t) * x_0

        if self.loss_type == "v":
            _, v_hat, _ = self.pred_module(x_t, t)
            v_target = x_1 - x_0
            return F.mse_loss(v_hat, target=v_target)
        elif self.loss_type == "x":
            x1_hat, _, _ = self.pred_module(x_t, t)
            return F.mse_loss(x1_hat, target=x_1)
        elif self.loss_type == "eps":
            _, _, eps_hat = self.pred_module(x_t, t)
            return F.mse_loss(eps_hat, target=x_0)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")


DATASET_CLASSES = {
    "swiss_roll": SwissRollDataset,
    "moons": MoonsDataset,
    "checkerboard": CheckerboardDataset,
    # "mnist": MNIST,
}


def visualize_data_vs_samples(
    data: torch.Tensor,
    samples: torch.Tensor,
    dataset: SyntheticAmbientDataset,
    out_path: Path,
    *,
    max_points: int = 4000,
    title: str | None = None,
    seed: int = 0,
) -> bytes:
    """
    Plot data vs samples in intrinsic coordinates via ``dataset.unproject`` (z = x @ Q).
    """
    data = data.detach().cpu()
    samples = samples.detach().cpu()
    z_data = dataset.unproject(data).detach().numpy()
    z_samp = dataset.unproject(samples).detach().numpy()

    n_data = min(z_data.shape[0], max_points)
    n_samp = min(z_samp.shape[0], max_points)
    g = torch.Generator().manual_seed(seed)
    di = torch.randperm(z_data.shape[0], generator=g)[:n_data]
    si = torch.randperm(z_samp.shape[0], generator=g)[:n_samp]
    z_data = z_data[di.numpy()]
    z_samp = z_samp[si.numpy()]

    d = dataset.intrinsic_dim
    if d == 2:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(
            z_data[:, 0],
            z_data[:, 1],
            s=5,
            alpha=0.2,
            c="tab:blue",
            label="data",
            rasterized=True,
        )
        ax.scatter(
            z_samp[:, 0],
            z_samp[:, 1],
            s=5,
            alpha=0.35,
            c="tab:orange",
            label="generated",
            rasterized=True,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.legend(loc="best", framealpha=0.9)
        ax.set_title(title or "Data vs generated (intrinsic)")
        fig.tight_layout()
    elif d == 3:
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            z_data[:, 0],
            z_data[:, 1],
            z_data[:, 2],
            s=4,
            alpha=0.15,
            c="tab:blue",
            label="data",
            depthshade=False,
        )
        ax.scatter(
            z_samp[:, 0],
            z_samp[:, 1],
            z_samp[:, 2],
            s=4,
            alpha=0.35,
            c="tab:orange",
            label="generated",
            depthshade=False,
        )
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_zlabel(r"$z_3$")
        ax.legend(loc="best", framealpha=0.9)
        ax.set_title(title or "Data vs generated (intrinsic)")
        fig.tight_layout()
    else:
        raise ValueError(f"Plotting supports intrinsic dim 2 or 3, got {d}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    png = buf.getvalue()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    return png

from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow matching toy experiment with optional wandb.")
    parser.add_argument("--dataset", choices=list(DATASET_CLASSES.keys()), default="swiss_roll")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Train on raw data + scalar noise_scale=std(data); compare against per-feature norm.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gen-modeling-flow")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="flow_matching")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-type", choices=["v", "x", "eps"], default="v")
    parser.add_argument("--pred-type", choices=["v", "x"], default="v")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_normalization = not args.no_normalize

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DATASET_CLASSES[args.dataset](
        ambient_dim=64,
        n_samples=10_000,
        noise=0.05,
        random_state=args.seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed)
    )

    stats: RunningNormalizationStats | None = None
    if use_normalization:
        stats = RunningNormalizationStats()
        stats.update(dataset.data.cpu())
        noise_scale = 1.0
    else:
        with torch.no_grad():
            noise_scale = float(dataset.data.std())

    config = {
        "dataset": args.dataset,
        "use_normalization": use_normalization,
        "noise_scale": noise_scale,
        "ambient_dim": dataset.ambient_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
    }

    run = None
    if not args.no_wandb:
        default_name = f"{args.dataset}_{'norm' if use_normalization else 'raw'}_{args.pred_type}_{args.loss_type}"
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or default_name,
            config=config,
        )

    net = NetworkND(dataset.ambient_dim, num_layers=4).to(device)
    flow = LinearFlow(
        net,
        noise_scale=noise_scale,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    out_dir = Path(__file__).resolve().parent / "flow_matching_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        epoch_losses: list[float] = []
        for (x, _) in tqdm(dataloader, desc=f"epoch {epoch}"):
            x = x.to(device)
            x_model = stats.normalize(x) if stats is not None else x
            loss = flow.compute_loss(x_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        loss_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"Epoch {epoch}, loss: {loss_mean:.6f}")

        log_payload: dict = {"train/loss_epoch_mean": loss_mean}
        if epoch % 10 == 0:
            net.eval()
            generated_norm = flow.sample(num_samples=4000, num_steps=200)
            generated = (
                stats.unnormalize(generated_norm) if stats is not None else generated_norm
            )
            title = (
                f"Epoch {epoch} — intrinsic (x @ Q), "
                f"{'normalized train' if use_normalization else 'raw train'}"
            )
            path_epoch = out_dir / f"data_vs_gen_epoch_{epoch:03d}.png"
            png = visualize_data_vs_samples(
                dataset.data,
                generated,
                dataset,
                out_path=path_epoch,
                title=title,
                seed=epoch,
            )
            path_latest = out_dir / "data_vs_gen_latest.png"
            path_latest.write_bytes(png)
            if run is not None:
                # wandb.Image expects a path, numpy array, or PIL image — not BytesIO.
                log_payload["viz/data_vs_gen"] = wandb.Image(
                    str(path_epoch), caption=title
                )
            net.train()

        if run is not None:
            wandb.log(log_payload, step=epoch)

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
