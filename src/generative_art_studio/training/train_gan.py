"""Reference vanilla/DCGAN adversarial training loop. Fully implemented —
it calls into `losses.generator_loss` / `losses.discriminator_loss`, so it
will raise `NotImplementedError` until you implement those (and, for
DCGAN, the generator/discriminator architectures themselves).

For Conditional GAN / WGAN / Pix2Pix / CycleGAN, adapt this loop in your
notebooks — the difference is only which loss functions and which
generator/discriminator calls you use per step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.latent_space import sample_latent
from .losses import discriminator_loss, generator_loss


@dataclass
class GANTrainHistory:
    g_loss: list[float] = field(default_factory=list)
    d_loss: list[float] = field(default_factory=list)


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    latent_dim: int,
    device: torch.device | str = "cpu",
    epochs: int = 1,
) -> GANTrainHistory:
    """Alternating Generator/Discriminator adversarial training."""
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    history = GANTrainHistory()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"GAN epoch {epoch + 1}/{epochs}")
        for real_images, _labels in pbar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # --- Train Discriminator ---
            d_optimizer.zero_grad()
            z = sample_latent(batch_size, latent_dim, device)
            fake_images = generator(z).detach()
            real_pred = discriminator(real_images)
            fake_pred = discriminator(fake_images)
            d_loss = discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator ---
            g_optimizer.zero_grad()
            z = sample_latent(batch_size, latent_dim, device)
            fake_images = generator(z)
            fake_pred = discriminator(fake_images)
            g_loss = generator_loss(fake_pred)
            g_loss.backward()
            g_optimizer.step()

            history.g_loss.append(g_loss.item())
            history.d_loss.append(d_loss.item())
            pbar.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item())

    return history
