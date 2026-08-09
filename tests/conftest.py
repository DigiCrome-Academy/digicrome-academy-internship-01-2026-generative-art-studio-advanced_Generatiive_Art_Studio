"""Shared pytest fixtures for the auto-grading test suite.

All fixtures here are deliberately tiny (small batch sizes, small feature
widths) so the *entire* suite runs on a CPU-only GitHub Actions runner in
well under a minute, with no dataset download required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make `generative_art_studio` importable without requiring `pip install -e .`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generative_art_studio.utils.seed import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Reseed everything before each test for reproducible grading."""
    set_seed(42)
    yield


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def image_size() -> int:
    return 64


@pytest.fixture
def img_channels() -> int:
    return 3


@pytest.fixture
def tiny_image_batch(batch_size, img_channels, image_size) -> torch.Tensor:
    """A batch of images in [-1, 1], matching Normalize(0.5, 0.5) preprocessing."""
    return torch.rand(batch_size, img_channels, image_size, image_size) * 2 - 1


@pytest.fixture
def tiny_labels(batch_size) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size,))
