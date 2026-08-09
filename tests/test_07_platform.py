"""Phase 4 — Generative Art Platform (rubric category: `platform`, weight 10%)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from generative_art_studio.app.model_registry import (
    Gallery,
    export_image,
    generate_samples,
    list_available_models,
    load_model,
)

pytestmark = pytest.mark.platform


def test_list_available_models_includes_vae_and_gan():
    models = list_available_models()
    assert "vae" in models
    assert "vanilla_gan" in models


def test_load_model_returns_correct_type():
    vae = load_model("vae", checkpoint_path=None, device="cpu")
    gan = load_model("vanilla_gan", checkpoint_path=None, device="cpu")
    assert isinstance(vae, torch.nn.Module)
    assert isinstance(gan, torch.nn.Module)


def test_load_model_unknown_key_raises():
    with pytest.raises(KeyError):
        load_model("not_a_real_model", device="cpu")


def test_generate_samples_vae_shape():
    model = load_model("vae", device="cpu")
    images = generate_samples("vae", model, num_samples=3, seed=0, device="cpu")
    assert images.shape[0] == 3
    assert images.shape[1] == 3  # RGB


def test_generate_samples_gan_shape():
    model = load_model("vanilla_gan", device="cpu")
    images = generate_samples("vanilla_gan", model, num_samples=3, seed=0, device="cpu")
    assert images.shape[0] == 3
    assert images.shape[1] == 3


def test_generate_samples_is_reproducible_with_seed():
    model = load_model("vanilla_gan", device="cpu")
    images_a = generate_samples("vanilla_gan", model, num_samples=2, seed=123, device="cpu")
    images_b = generate_samples("vanilla_gan", model, num_samples=2, seed=123, device="cpu")
    assert torch.allclose(images_a, images_b, atol=1e-5)


def test_export_image_creates_upscaled_file(tmp_path):
    image = torch.rand(3, 16, 16) * 2 - 1
    out_path = tmp_path / "art.png"
    scale = 4
    result_path = export_image(image, out_path, scale_factor=scale)

    assert Path(result_path).exists()
    from PIL import Image

    with Image.open(result_path) as im:
        assert im.size == (16 * scale, 16 * scale)


def test_gallery_add_and_clear():
    gallery = Gallery()
    assert len(gallery) == 0
    gallery.add(torch.zeros(3, 8, 8), "vae", seed=1)
    gallery.add(torch.zeros(3, 8, 8), "vanilla_gan", seed=2)
    assert len(gallery) == 2
    items = list(gallery)
    assert items[0]["model_key"] == "vae"
    gallery.clear()
    assert len(gallery) == 0
