"""Phase 2 — Wasserstein GAN: critic architecture, Wasserstein loss, and the
WGAN-GP gradient penalty (rubric category: `gan`, weight 30%)."""
from __future__ import annotations

import pytest
import torch

from generative_art_studio.models.gans.wgan import WGANCritic, clip_weights
from generative_art_studio.training.losses import (
    gradient_penalty,
    wgan_critic_loss,
    wgan_generator_loss,
)

pytestmark = pytest.mark.gan


def test_wgan_critic_output_shape_and_unbounded(batch_size, img_channels):
    critic = WGANCritic(img_channels=img_channels, feature_maps=8)
    img = torch.randn(batch_size, img_channels, 64, 64) * 5  # large-magnitude input
    out = critic(img)
    assert out.shape == (batch_size, 1)
    # A WGAN critic has no Sigmoid — its scores should NOT be squashed into [0, 1].
    assert (out.min() < 0) or (out.max() > 1)


def test_wgan_critic_loss_matches_formula():
    real_score = torch.tensor([[1.0], [2.0], [3.0]])
    fake_score = torch.tensor([[0.5], [0.5], [0.5]])
    result = wgan_critic_loss(real_score, fake_score)
    expected = fake_score.mean() - real_score.mean()
    assert torch.allclose(result, expected, atol=1e-5)


def test_wgan_generator_loss_matches_formula():
    fake_score = torch.tensor([[1.0], [-2.0], [3.0]])
    result = wgan_generator_loss(fake_score)
    assert torch.allclose(result, -fake_score.mean(), atol=1e-5)


def test_clip_weights_bounds_parameters():
    critic = WGANCritic(img_channels=3, feature_maps=4)
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(10.0)
    clip_weights(critic, clip_value=0.01)
    for p in critic.parameters():
        assert p.data.max().item() <= 0.01 + 1e-6
        assert p.data.min().item() >= -0.01 - 1e-6


def test_gradient_penalty_matches_known_value_for_linear_critic():
    """For critic(x) = sum(x) (a linear function), the gradient wrt every
    input element is exactly 1, independent of the interpolation point —
    so the penalty has a closed-form value we can check exactly:
    (||ones(D)||_2 - 1)^2 = (sqrt(D) - 1)^2.
    """

    def linear_critic(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).sum(dim=1, keepdim=True)

    real = torch.zeros(3, 4)
    fake = torch.ones(3, 4)
    penalty = gradient_penalty(linear_critic, real, fake, device="cpu")

    expected = (4 ** 0.5 - 1) ** 2  # D = 4
    assert torch.isclose(penalty, torch.tensor(expected), atol=1e-3)


def test_gradient_penalty_is_zero_for_unit_norm_gradient_critic():
    """critic(x) = x[:, 0] has gradient norm exactly 1 everywhere -> penalty ~ 0."""

    def unit_grad_critic(x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return flat[:, :1]

    real = torch.randn(5, 6)
    fake = torch.randn(5, 6)
    penalty = gradient_penalty(unit_grad_critic, real, fake, device="cpu")
    assert penalty.item() < 1e-4
