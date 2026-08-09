"""Evaluation & Analysis — FID and Inception Score math (rubric category:
`evaluation`, weight 15%).

These tests use plain synthetic feature/probability arrays, not a real
Inception network, so they run instantly with no internet access — they
check that you implemented the *metric formulas* correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from generative_art_studio.evaluation.metrics import compute_fid, compute_inception_score

pytestmark = pytest.mark.evaluation


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def test_fid_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 16)).astype(np.float64)
    fid = compute_fid(features, features.copy())
    assert fid == pytest.approx(0.0, abs=1e-3)


def test_fid_is_positive_for_different_distributions():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, size=(200, 16))
    fake = rng.normal(loc=5.0, size=(200, 16))  # shifted mean
    fid = compute_fid(real, fake)
    assert fid > 10.0


def test_fid_matches_mean_shift_when_covariances_are_equal():
    """If real and fake share the same covariance, FID reduces to
    ||mu_real - mu_fake||^2 (the trace term cancels to 0)."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 8))
    shift = np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
    real = base
    fake = base + shift  # identical covariance, shifted mean, same samples otherwise
    fid = compute_fid(real, fake)
    assert fid == pytest.approx(float(np.dot(shift, shift)), rel=0.05)


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------
def test_inception_score_is_one_for_uniform_uninformative_predictions():
    """If every sample predicts the exact same uniform distribution over
    classes, p(y|x) == p(y) everywhere, so KL == 0 and IS == exp(0) == 1."""
    num_classes = 5
    preds = np.full((100, num_classes), 1.0 / num_classes)
    mean_is, std_is = compute_inception_score(preds, splits=5)
    assert mean_is == pytest.approx(1.0, abs=1e-3)


def test_inception_score_is_higher_for_confident_diverse_predictions():
    """Confident, evenly-spread-across-classes predictions should score
    higher than the uninformative uniform baseline."""
    num_classes = 4
    n = 400
    rng = np.random.default_rng(2)
    labels = rng.integers(0, num_classes, size=n)
    preds = np.full((n, num_classes), 0.01 / (num_classes - 1))
    preds[np.arange(n), labels] = 0.99
    mean_is, _ = compute_inception_score(preds, splits=5)
    assert mean_is > 1.5


def test_inception_score_returns_mean_and_std():
    preds = np.random.default_rng(3).dirichlet(alpha=[1, 1, 1], size=50)
    result = compute_inception_score(preds, splits=5)
    assert len(result) == 2
    mean_is, std_is = result
    assert mean_is > 0
    assert std_is >= 0
