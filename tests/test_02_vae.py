"""Phase 1 — Variational Autoencoder: reparameterization trick, VAE loss,
and latent-space interpolation (rubric category: `vae`, weight 20%)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from generative_art_studio.models.autoencoders.vae import VAE
from generative_art_studio.training.losses import vae_loss
from generative_art_studio.utils.latent_space import interpolate_latent

pytestmark = pytest.mark.vae


# ---------------------------------------------------------------------------
# Reparameterization trick
# ---------------------------------------------------------------------------
def test_reparameterize_output_shape():
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)
    z = VAE.reparameterize(mu, logvar)
    assert z.shape == mu.shape


def test_reparameterize_is_stochastic():
    """Two calls with the same (mu, logvar) must differ — proves eps is
    actually sampled fresh each time rather than being ignored."""
    mu = torch.zeros(8, 32)
    logvar = torch.zeros(8, 32)  # std = 1
    z1 = VAE.reparameterize(mu, logvar)
    z2 = VAE.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)


def test_reparameterize_near_zero_std_matches_mu():
    """With logvar very negative (std ~ 0), z should collapse to ~mu."""
    mu = torch.full((4, 8), 3.0)
    logvar = torch.full((4, 8), -20.0)
    z = VAE.reparameterize(mu, logvar)
    assert torch.allclose(z, mu, atol=1e-2)


def test_reparameterize_is_differentiable():
    mu = torch.zeros(4, 8, requires_grad=True)
    logvar = torch.zeros(4, 8, requires_grad=True)
    z = VAE.reparameterize(mu, logvar)
    z.sum().backward()
    assert mu.grad is not None and torch.any(mu.grad != 0)
    assert logvar.grad is not None


# ---------------------------------------------------------------------------
# Full VAE forward pass
# ---------------------------------------------------------------------------
def test_vae_forward_shapes(tiny_image_batch, img_channels):
    latent_dim = 16
    model = VAE(in_channels=img_channels, latent_dim=latent_dim)
    recon, mu, logvar = model(tiny_image_batch)
    assert recon.shape == tiny_image_batch.shape
    assert mu.shape == (tiny_image_batch.size(0), latent_dim)
    assert logvar.shape == (tiny_image_batch.size(0), latent_dim)


def test_vae_sample_generates_new_images(img_channels):
    model = VAE(in_channels=img_channels, latent_dim=16)
    images = model.sample(5, device="cpu")
    assert images.shape == (5, img_channels, 64, 64)


# ---------------------------------------------------------------------------
# VAE loss (reconstruction + KL divergence)
# ---------------------------------------------------------------------------
def test_vae_loss_matches_closed_form():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 8, 8) * 2 - 1
    recon_x = torch.rand(4, 3, 8, 8) * 2 - 1
    mu = torch.randn(4, 10)
    logvar = torch.randn(4, 10)

    total, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, kl_weight=1.0)

    expected_recon = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    expected_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    expected_total = expected_recon + expected_kl

    assert torch.allclose(recon_loss, expected_recon, rtol=1e-3)
    assert torch.allclose(kl_loss, expected_kl, rtol=1e-3)
    assert torch.allclose(total, expected_total, rtol=1e-3)


def test_vae_loss_kl_is_zero_for_standard_normal():
    """KL(N(0, I) || N(0, I)) == 0: mu=0, logvar=0 should give ~0 KL term."""
    x = torch.zeros(2, 3, 4, 4)
    recon_x = torch.zeros(2, 3, 4, 4)
    mu = torch.zeros(2, 6)
    logvar = torch.zeros(2, 6)
    _, _, kl_loss = vae_loss(recon_x, x, mu, logvar)
    assert torch.allclose(kl_loss, torch.tensor(0.0), atol=1e-5)


def test_vae_loss_respects_kl_weight():
    torch.manual_seed(1)
    x = torch.rand(2, 3, 4, 4)
    recon_x = torch.rand(2, 3, 4, 4)
    mu = torch.randn(2, 6)
    logvar = torch.randn(2, 6)
    total_w1, recon1, kl1 = vae_loss(recon_x, x, mu, logvar, kl_weight=1.0)
    total_w2, recon2, kl2 = vae_loss(recon_x, x, mu, logvar, kl_weight=2.0)
    assert torch.allclose(recon1, recon2)
    assert torch.allclose(total_w2, recon2 + 2.0 * kl2, rtol=1e-3)


# ---------------------------------------------------------------------------
# Latent space interpolation
# ---------------------------------------------------------------------------
def test_interpolate_latent_endpoints_and_shape():
    z1 = torch.randn(12)
    z2 = torch.randn(12)
    steps = 6
    path = interpolate_latent(z1, z2, steps=steps)
    assert path.shape == (steps, 12)
    assert torch.allclose(path[0], z1, atol=1e-5)
    assert torch.allclose(path[-1], z2, atol=1e-5)


def test_interpolate_latent_matches_linear_formula():
    z1 = torch.tensor([0.0, 0.0])
    z2 = torch.tensor([10.0, -10.0])
    path = interpolate_latent(z1, z2, steps=5)
    expected = torch.stack([(1 - t) * z1 + t * z2 for t in torch.linspace(0, 1, 5)])
    assert torch.allclose(path, expected, atol=1e-5)
