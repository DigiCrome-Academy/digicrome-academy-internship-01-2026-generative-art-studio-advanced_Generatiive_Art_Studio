"""Vanilla (fully-connected) GAN — the simplest possible Generator/Discriminator pair.

Learning objective (docs/PROJECT_BRIEF.md, Phase 2):
"Implement vanilla GAN architecture."

This one is given fully implemented as a working reference for the
adversarial-training loop (see `src/generative_art_studio/training/train_gan.py`
and `losses.py`, where you implement the actual adversarial loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VanillaGenerator(nn.Module):
    """Maps a latent noise vector to a flat image via fully-connected layers."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        out_dim = img_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.net(z)
        return img.view(-1, self.img_channels, self.img_size, self.img_size)


class VanillaDiscriminator(nn.Module):
    """Classifies a flattened image as real (close to 1) or fake (close to 0)."""

    def __init__(self, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        in_dim = img_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        flat = img.view(img.size(0), -1)
        return self.net(flat)
