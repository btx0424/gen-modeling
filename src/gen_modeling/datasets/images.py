from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, STL10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


@dataclass(frozen=True)
class ImageDatasetInfo:
    """Metadata for image datasets. ``norm_*`` describe ``Normalize`` used in the loader (if any)."""

    channels: int
    height: int
    width: int
    num_classes: int
    norm_mean: tuple[float, ...] | None = None
    norm_std: tuple[float, ...] | None = None


def tensor_batch_to_display(x: torch.Tensor, info: ImageDatasetInfo) -> torch.Tensor:
    """
    Map a batch (B, C, H, W) or image (C, H, W) from model/dataset space to [0, 1] for ``imshow``.
    When ``info.norm_mean`` / ``norm_std`` are set, applies the inverse of ``Normalize``.
    """
    out = x.detach().cpu().float()
    if info.norm_mean is not None and info.norm_std is not None:
        m = torch.as_tensor(info.norm_mean, dtype=out.dtype)
        s = torch.as_tensor(info.norm_std, dtype=out.dtype)
        if out.dim() == 4:
            m = m.view(1, -1, 1, 1)
            s = s.view(1, -1, 1, 1)
        else:
            m = m.view(-1, 1, 1)
            s = s.view(-1, 1, 1)
        out = out * s + m
    return out.clamp(0, 1)


class MNISTDataset(Dataset):

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
        self.info = ImageDatasetInfo(
            channels=1,
            height=28,
            width=28,
            num_classes=10,
            norm_mean=(0.5,) if normalize else None,
            norm_std=(0.5,) if normalize else None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)


class STL10Dataset(Dataset):

    # pre-computed stats for ``Normalize``
    mean: Tuple[float, float, float] = (0.44671061635017395, 0.4398098587989807, 0.4066464602947235)
    std: Tuple[float, float, float] = (0.26034098863601685, 0.2565772831439972, 0.2712673842906952)

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        download: bool = True,
        normalize: bool = True,
        size: int | None = 64,
    ) -> None:
        super().__init__()
        if size is not None:
            h, w = size, size
        else:
            h, w = 96, 96
        self.info = ImageDatasetInfo(
            channels=3,
            height=h,
            width=w,
            num_classes=10,
            norm_mean=self.mean if normalize else None,
            norm_std=self.std if normalize else None,
        )
        transforms = []
        if size is not None:
            transforms.append(Resize((h, w)))
        transforms.append(ToTensor())
        if normalize:
            transforms.append(
                Normalize(self.mean, self.std)
            )
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
