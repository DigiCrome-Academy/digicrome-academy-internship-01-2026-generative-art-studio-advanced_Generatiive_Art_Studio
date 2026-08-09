"""Phase 1 — Autoencoders (rubric category: `vae`, weight 20%)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from generative_art_studio.models.autoencoders.denoising_ae import (
    DenoisingAutoencoder,
    add_gaussian_noise,
)
from generative_art_studio.models.autoencoders.vanilla_ae import VanillaAutoencoder

pytestmark = pytest.mark.vae


def test_vanilla_autoencoder_reconstructs_correct_shape(tiny_image_batch, img_channels):
    model = VanillaAutoencoder(in_channels=img_channels, latent_dim=32)
    out = model(tiny_image_batch)
    assert out.shape == tiny_image_batch.shape


def test_vanilla_autoencoder_encoder_produces_latent_dim(tiny_image_batch):
    latent_dim = 16
    model = VanillaAutoencoder(in_channels=3, latent_dim=latent_dim)
    z = model.encoder(tiny_image_batch)
    assert z.shape == (tiny_image_batch.size(0), latent_dim)


def test_add_gaussian_noise_shape_and_range(tiny_image_batch):
    noisy = add_gaussian_noise(tiny_image_batch, noise_factor=0.3)
    assert noisy.shape == tiny_image_batch.shape
    assert torch.all(noisy >= -1.0) and torch.all(noisy <= 1.0)


def test_add_gaussian_noise_actually_perturbs_input(tiny_image_batch):
    noisy = add_gaussian_noise(tiny_image_batch, noise_factor=0.5)
    assert not torch.allclose(noisy, tiny_image_batch)


def test_add_gaussian_noise_zero_factor_is_identity(tiny_image_batch):
    noisy = add_gaussian_noise(tiny_image_batch, noise_factor=0.0)
    assert torch.allclose(noisy, tiny_image_batch, atol=1e-6)


def test_denoising_autoencoder_forward_shape(tiny_image_batch, img_channels):
    model = DenoisingAutoencoder(in_channels=img_channels, latent_dim=32, noise_factor=0.2)
    out = model(tiny_image_batch)
    assert out.shape == tiny_image_batch.shape
