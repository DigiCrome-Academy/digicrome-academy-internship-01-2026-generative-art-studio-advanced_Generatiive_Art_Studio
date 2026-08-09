"""Visualization helpers shared by notebooks, tests, and the Streamlit app.

Fully implemented — use these instead of rewriting plotting code in every
notebook. They only depend on torch/torchvision/matplotlib/numpy so they
work headlessly in CI (the `Agg` backend has no display requirement).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI/servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def denormalize(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Undo the standard `Normalize(mean, std)` transform, clamped to [0, 1]."""
    return (tensor * std + mean).clamp(0, 1)


def make_image_grid(images: torch.Tensor, nrow: int = 8, normalize: bool = True) -> torch.Tensor:
    """Arrange a batch of images (B, C, H, W) into a single grid tensor."""
    if normalize:
        images = denormalize(images)
    return make_grid(images, nrow=nrow)


def plot_image_grid(images: torch.Tensor, nrow: int = 8, title: str | None = None, normalize: bool = True):
    """Render a batch of images as a matplotlib grid and return the Figure."""
    grid = make_image_grid(images, nrow=nrow, normalize=normalize)
    npimg = grid.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(nrow, nrow))
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, normalize: bool = True) -> Path:
    """Save a batch of images as a single PNG grid. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_image_grid(images, nrow=nrow, normalize=normalize)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
