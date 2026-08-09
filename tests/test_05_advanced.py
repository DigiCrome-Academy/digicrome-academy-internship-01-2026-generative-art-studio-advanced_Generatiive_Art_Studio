"""Phase 3 — Pix2Pix, CycleGAN, and their losses (rubric category:
`advanced`, weight 20%)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_art_studio.models.advanced.cyclegan import (
    CycleGANGenerator,
    ResidualBlock,
)
from generative_art_studio.models.advanced.pix2pix import (
    PatchGANDiscriminator,
    UNetDown,
    UNetGenerator,
    UNetUp,
)
from generative_art_studio.training.losses import (
    cycle_consistency_loss,
    identity_loss,
    patchgan_discriminator_loss,
    pix2pix_generator_loss,
)

pytestmark = pytest.mark.advanced


# ---------------------------------------------------------------------------
# Pix2Pix U-Net skip connections
# ---------------------------------------------------------------------------
def test_unet_up_concatenates_skip_connection():
    up = UNetUp(in_channels=8, out_channels=4)
    x = torch.randn(2, 8, 4, 4)
    skip = torch.full((2, 6, 8, 8), 7.0)  # distinct channel count & value from the block output
    out = up(x, skip)
    assert out.shape == (2, 4 + 6, 8, 8)
    # The skip connection's values must survive untouched in the concatenation.
    assert torch.allclose(out[:, 4:], skip)


def test_unet_down_output_shape():
    down = UNetDown(3, 8)
    x = torch.randn(2, 3, 64, 64)
    out = down(x)
    assert out.shape == (2, 8, 32, 32)


def test_unet_generator_forward_shape(tiny_image_batch):
    gen = UNetGenerator(in_channels=3, out_channels=3, features=8)
    out = gen(tiny_image_batch)
    assert out.shape == tiny_image_batch.shape
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_patchgan_discriminator_output_is_a_patch_map(tiny_image_batch):
    disc = PatchGANDiscriminator(in_channels=6)
    out = disc(tiny_image_batch, tiny_image_batch)
    assert out.dim() == 4
    assert out.size(0) == tiny_image_batch.size(0)
    assert out.size(1) == 1


# ---------------------------------------------------------------------------
# CycleGAN residual blocks
# ---------------------------------------------------------------------------
def test_residual_block_adds_skip_connection():
    block = ResidualBlock(channels=3)
    block.block = nn.Identity()  # isolate the skip-connection logic from conv internals
    x = torch.randn(2, 3, 8, 8)
    out = block(x)
    assert torch.allclose(out, x + x)


def test_residual_block_output_shape_matches_input():
    block = ResidualBlock(channels=4)
    x = torch.randn(2, 4, 10, 10)
    out = block(x)
    assert out.shape == x.shape


def test_cyclegan_generator_forward_shape(tiny_image_batch):
    gen = CycleGANGenerator(in_channels=3, out_channels=3, features=8, num_residual_blocks=2)
    out = gen(tiny_image_batch)
    assert out.shape == tiny_image_batch.shape


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_pix2pix_generator_loss_matches_formula():
    disc_fake_pred = torch.randn(2, 1, 4, 4)
    fake_img = torch.rand(2, 3, 8, 8) * 2 - 1
    target_img = torch.rand(2, 3, 8, 8) * 2 - 1
    lambda_l1 = 50.0

    result = pix2pix_generator_loss(disc_fake_pred, fake_img, target_img, lambda_l1=lambda_l1)

    expected_adv = F.binary_cross_entropy_with_logits(disc_fake_pred, torch.ones_like(disc_fake_pred))
    expected_l1 = F.l1_loss(fake_img, target_img)
    expected = expected_adv + lambda_l1 * expected_l1
    assert torch.allclose(result, expected, rtol=1e-3)


def test_patchgan_discriminator_loss_low_when_correct():
    real_pred = torch.full((2, 1, 4, 4), 10.0)  # logits -> sigmoid ~= 1
    fake_pred = torch.full((2, 1, 4, 4), -10.0)  # logits -> sigmoid ~= 0
    loss = patchgan_discriminator_loss(real_pred, fake_pred)
    assert loss.item() < 0.01


def test_cycle_consistency_loss_matches_l1():
    real = torch.rand(3, 3, 8, 8)
    reconstructed = torch.rand(3, 3, 8, 8)
    result = cycle_consistency_loss(real, reconstructed)
    expected = F.l1_loss(reconstructed, real)
    assert torch.allclose(result, expected, rtol=1e-4)


def test_cycle_consistency_loss_is_zero_for_perfect_reconstruction():
    real = torch.rand(3, 3, 8, 8)
    assert cycle_consistency_loss(real, real.clone()).item() < 1e-6


def test_identity_loss_matches_l1():
    real = torch.rand(2, 3, 4, 4)
    same_domain_output = torch.rand(2, 3, 4, 4)
    result = identity_loss(real, same_domain_output)
    expected = F.l1_loss(same_domain_output, real)
    assert torch.allclose(result, expected, rtol=1e-4)
