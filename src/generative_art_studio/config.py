"""Global configuration shared across models, training, and the app.

Students may tune these values (e.g. LATENT_DIM, IMAGE_SIZE) while
experimenting, but should not need to change the names/shapes the test
suite depends on unless a TODO explicitly says so.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Compute device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image / data settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = 64          # square images, e.g. CelebA cropped/resized to 64x64
IMAGE_CHANNELS = 3       # RGB
NUM_CLASSES = 2          # e.g. for Conditional GAN class-conditioning demo

# ---------------------------------------------------------------------------
# Autoencoder / VAE settings
# ---------------------------------------------------------------------------
AE_LATENT_DIM = 128
VAE_LATENT_DIM = 128

# ---------------------------------------------------------------------------
# GAN settings
# ---------------------------------------------------------------------------
GAN_LATENT_DIM = 100
GENERATOR_FEATURE_MAPS = 64
DISCRIMINATOR_FEATURE_MAPS = 64

# WGAN
WGAN_CLIP_VALUE = 0.01
WGAN_GP_LAMBDA = 10.0
N_CRITIC = 5

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
