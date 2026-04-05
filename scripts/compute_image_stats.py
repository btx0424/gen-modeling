from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gen_modeling.datasets.images import MNISTDataset, STL10Dataset


def build_dataset(name: str, data_root: Path, split: str):
    if name == "mnist":
        train = split == "train"
        return MNISTDataset(data_root, train=train, download=True, normalize=False)
    if name == "stl10":
        return STL10Dataset(data_root, split=split, download=True, normalize=False)
    raise ValueError(f"Unknown dataset: {name}")


@torch.no_grad()
def compute_stats(loader: DataLoader, num_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_sum = torch.zeros(num_channels, dtype=torch.float64)
    pixel_sq_sum = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0

    for images, _ in loader:
        images = images.to(dtype=torch.float64)
        batch_size, channels, height, width = images.shape
        if channels != num_channels:
            raise ValueError(f"Expected {num_channels} channels, got {channels}")
        images = images.reshape(batch_size, channels, height * width)
        pixel_sum += images.sum(dim=(0, 2))
        pixel_sq_sum += images.square().sum(dim=(0, 2))
        total_pixels += batch_size * height * width

    mean = pixel_sum / total_pixels
    var = (pixel_sq_sum / total_pixels) - mean.square()
    std = torch.sqrt(var.clamp_min(0.0))
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel mean/std for image datasets.")
    parser.add_argument("--dataset", choices=["mnist", "stl10"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.dataset, args.data_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    mean, std = compute_stats(loader, dataset.info.channels)
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
    }

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
