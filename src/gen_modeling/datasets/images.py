from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, STL10
from torchvision.transforms import Compose, Normalize, ToTensor


@dataclass(frozen=True)
class ImageDatasetInfo:
    channels: int
    height: int
    width: int
    num_classes: int


class MNISTDataset(Dataset):
    info = ImageDatasetInfo(channels=1, height=28, width=28, num_classes=10)

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = True,
        download: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        transforms = [ToTensor()]
        if normalize:
            transforms.append(Normalize((0.5,), (0.5,)))
        self.dataset = MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=Compose(transforms),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)


class STL10Dataset(Dataset):
    info = ImageDatasetInfo(channels=3, height=96, width=96, num_classes=10)

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        download: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        transforms = [ToTensor()]
        if normalize:
            transforms.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.dataset = STL10(
            root=str(root),
            split=split,
            download=download,
            transform=Compose(transforms),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)
