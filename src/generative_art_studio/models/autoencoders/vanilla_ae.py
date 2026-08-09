"""Vanilla (plain) convolutional autoencoder for image compression.

This one is given to you **fully implemented** as a working reference —
study it before implementing DenoisingAutoencoder and VAE, which reuse the
same encoder/decoder shape conventions.

Learning objective (docs/PROJECT_BRIEF.md, Phase 1):
"Build vanilla autoencoder for image compression."
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Downsamples a (B, C, 64, 64) image to a (B, latent_dim) vector."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    """Upsamples a (B, latent_dim) vector back to a (B, C, 64, 64) image."""

    def __init__(self, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)


class VanillaAutoencoder(nn.Module):
    """A plain autoencoder: image -> latent vector -> reconstructed image."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
