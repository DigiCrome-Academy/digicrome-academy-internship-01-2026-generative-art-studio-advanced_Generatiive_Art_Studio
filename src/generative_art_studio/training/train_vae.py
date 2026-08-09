"""Reference VAE training loop. Fully implemented — it calls into
`losses.vae_loss`, so it will raise `NotImplementedError` until you've
implemented that function (and `VAE.reparameterize`). Use this from your
notebooks rather than rewriting the loop each time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.autoencoders.vae import VAE
from .losses import vae_loss


@dataclass
class VAETrainHistory:
    total: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)


def train_vae(
    model: VAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    epochs: int = 1,
    kl_weight: float = 1.0,
) -> VAETrainHistory:
    """Train `model` for `epochs` epochs over `dataloader`, returning a loss history."""
    model.to(device)
    model.train()
    history = VAETrainHistory()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"VAE epoch {epoch + 1}/{epochs}")
        for images, _labels in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            history.total.append(loss.item())
            history.recon.append(recon_loss.item())
            history.kl.append(kl_loss.item())
            pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

    return history
