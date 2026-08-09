"""Environment sanity checks. Not tied to a rubric weight — if these fail,
nothing else in the suite can be trusted, so fix your environment first.
"""
from __future__ import annotations

import importlib


def test_package_importable():
    module = importlib.import_module("generative_art_studio")
    assert hasattr(module, "__version__")


def test_torch_available():
    import torch

    x = torch.randn(2, 2)
    assert x.shape == (2, 2)


def test_subpackages_importable():
    for name in [
        "generative_art_studio.config",
        "generative_art_studio.data",
        "generative_art_studio.utils",
        "generative_art_studio.models.autoencoders",
        "generative_art_studio.models.gans",
        "generative_art_studio.models.advanced",
        "generative_art_studio.training.losses",
        "generative_art_studio.evaluation.metrics",
        "generative_art_studio.app.model_registry",
    ]:
        importlib.import_module(name)
