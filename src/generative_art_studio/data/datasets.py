"""Dataset loading utilities.

`get_image_dataset` / `get_dataloader` are fully implemented and wrap any
folder of images (e.g. downloaded CelebA/WikiArt — see
scripts/download_data.py and docs/SETUP.md) with `torchvision.datasets.ImageFolder`.

`SyntheticImageDataset` generates deterministic random "images" on the fly.
It requires **no download** and is what the test suite / CI use so that
model-plumbing tests run in seconds anywhere. Use it for quick local
smoke-testing of your architectures before pointing them at real data.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SyntheticImageDataset(Dataset):
    """Deterministic random-noise image dataset — no download required.

    Useful for unit tests and for sanity-checking a model's forward/backward
    pass shapes before training on real data.
    """

    def __init__(self, num_samples: int = 32, image_size: int = 64, channels: int = 3, num_classes: int = 2, seed: int = 42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        generator = torch.Generator().manual_seed(seed)
        # Pre-generate so __getitem__ is deterministic and fast.
        self.images = torch.rand(
            num_samples, channels, image_size, image_size, generator=generator
        ) * 2 - 1  # in [-1, 1], matching Normalize(0.5, 0.5)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def get_image_dataset(root: str | Path, image_size: int = 64) -> datasets.ImageFolder:
    """Load a real image dataset from disk (ImageFolder layout: root/class_x/*.png).

    For CelebA without class subfolders, put every image under a single
    `root/all/` folder — ImageFolder just needs at least one subdirectory.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.ImageFolder(root=str(root), transform=transform)


def get_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Thin wrapper around torch's DataLoader with sensible defaults for CPU/CI use."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
