"""Phase 2 — Vanilla GAN, DCGAN, Conditional GAN, and adversarial loss
(rubric category: `gan`, weight 30%)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from generative_art_studio.models.gans.conditional_gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
)
from generative_art_studio.models.gans.dcgan import (
    DCGANDiscriminator,
    DCGANGenerator,
    weights_init_dcgan,
)
from generative_art_studio.models.gans.vanilla_gan import (
    VanillaDiscriminator,
    VanillaGenerator,
)
from generative_art_studio.training.losses import discriminator_loss, generator_loss

pytestmark = pytest.mark.gan


# ---------------------------------------------------------------------------
# Vanilla GAN (given, reference implementation — should always pass)
# ---------------------------------------------------------------------------
def test_vanilla_generator_output_shape(batch_size, img_channels, image_size):
    gen = VanillaGenerator(latent_dim=20, img_channels=img_channels, img_size=image_size)
    z = torch.randn(batch_size, 20)
    out = gen(z)
    assert out.shape == (batch_size, img_channels, image_size, image_size)
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh output


def test_vanilla_discriminator_output_shape(tiny_image_batch):
    disc = VanillaDiscriminator(img_channels=3, img_size=64)
    out = disc(tiny_image_batch)
    assert out.shape == (tiny_image_batch.size(0), 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)  # Sigmoid output


# ---------------------------------------------------------------------------
# Adversarial loss (BCE-based)
# ---------------------------------------------------------------------------
def test_discriminator_loss_matches_bce_formula():
    real_pred = torch.rand(6, 1) * 0.5 + 0.5  # in (0.5, 1.0)
    fake_pred = torch.rand(6, 1) * 0.5  # in (0.0, 0.5)
    result = discriminator_loss(real_pred, fake_pred)
    expected = (
        F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        + F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    ) / 2
    assert torch.allclose(result, expected, rtol=1e-4)


def test_generator_loss_matches_bce_formula():
    fake_pred = torch.rand(6, 1)
    result = generator_loss(fake_pred)
    expected = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
    assert torch.allclose(result, expected, rtol=1e-4)


def test_discriminator_loss_is_low_when_confident_and_correct():
    real_pred = torch.full((4, 1), 0.99)
    fake_pred = torch.full((4, 1), 0.01)
    loss = discriminator_loss(real_pred, fake_pred)
    assert loss.item() < 0.05


def test_discriminator_loss_is_high_when_confidently_wrong():
    real_pred = torch.full((4, 1), 0.01)  # D thinks real images are fake
    fake_pred = torch.full((4, 1), 0.99)  # D thinks fake images are real
    loss = discriminator_loss(real_pred, fake_pred)
    assert loss.item() > 2.0


# ---------------------------------------------------------------------------
# DCGAN (best-practice conv architecture)
# ---------------------------------------------------------------------------
def test_dcgan_generator_output_shape(batch_size, img_channels):
    gen = DCGANGenerator(latent_dim=20, img_channels=img_channels, feature_maps=8)
    z = torch.randn(batch_size, 20)
    out = gen(z)
    assert out.shape == (batch_size, img_channels, 64, 64)


def test_dcgan_discriminator_output_shape(batch_size, img_channels):
    disc = DCGANDiscriminator(img_channels=img_channels, feature_maps=8)
    img = torch.randn(batch_size, img_channels, 64, 64)
    out = disc(img)
    assert out.shape == (batch_size, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_dcgan_weights_init_does_not_crash():
    gen = DCGANGenerator(latent_dim=10, img_channels=3, feature_maps=4)
    gen.apply(weights_init_dcgan)


def test_dcgan_generator_gradients_flow(batch_size):
    gen = DCGANGenerator(latent_dim=10, img_channels=3, feature_maps=4)
    z = torch.randn(batch_size, 10, requires_grad=True)
    out = gen(z)
    out.sum().backward()
    assert z.grad is not None
    assert any(p.grad is not None for p in gen.parameters())


# ---------------------------------------------------------------------------
# Conditional GAN
# ---------------------------------------------------------------------------
def test_conditional_generator_output_shape(batch_size, tiny_labels):
    gen = ConditionalGenerator(latent_dim=20, num_classes=2, img_channels=3, img_size=64)
    z = torch.randn(batch_size, 20)
    out = gen(z, tiny_labels)
    assert out.shape == (batch_size, 3, 64, 64)


def test_conditional_discriminator_output_shape(tiny_image_batch, tiny_labels):
    disc = ConditionalDiscriminator(num_classes=2, img_channels=3, img_size=64)
    out = disc(tiny_image_batch, tiny_labels)
    assert out.shape == (tiny_image_batch.size(0), 1)


def test_conditional_generator_labels_change_output(batch_size):
    gen = ConditionalGenerator(latent_dim=20, num_classes=2, img_channels=3, img_size=64)
    gen.eval()
    z = torch.randn(batch_size, 20)
    labels_a = torch.zeros(batch_size, dtype=torch.long)
    labels_b = torch.ones(batch_size, dtype=torch.long)
    out_a = gen(z, labels_a)
    out_b = gen(z, labels_b)
    assert not torch.allclose(out_a, out_b)
