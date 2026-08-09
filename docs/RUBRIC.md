# Grading Rubric

Total: 100%. The first five categories (95%) are **auto-graded** by
[`.github/workflows/grading.yml`](../.github/workflows/grading.yml) on
every push, via `pytest` markers defined in [`pyproject.toml`](../pyproject.toml).
The last category (5%) is manually reviewed by your instructor from your
submitted deliverables.

| # | Component | Weight | pytest marker | Test files |
|---|---|---|---|---|
| 1 | VAE Implementation | 20% | `vae` | `tests/test_01_autoencoders.py`, `tests/test_02_vae.py` |
| 2 | GAN Implementation | 30% | `gan` | `tests/test_03_gans_basic.py`, `tests/test_04_wgan.py` |
| 3 | Advanced Techniques | 20% | `advanced` | `tests/test_05_advanced.py` |
| 4 | Evaluation & Analysis | 15% | `evaluation` | `tests/test_06_evaluation.py` |
| 5 | Platform Development | 10% | `platform` | `tests/test_07_platform.py` |
| 6 | Portfolio & Presentation | 5% | *(manual)* | Instructor review of `outputs/portfolio/` + your video |

Run one category at a time while you work:

```bash
pytest -m vae -v
pytest -m gan -v
pytest -m advanced -v
pytest -m evaluation -v
pytest -m platform -v
```

## How your score is computed

For each category, `scripts/grade.py` (same logic the CI workflow uses)
computes:

```
category_score = (tests_passed / tests_total) * category_weight
```

and sums across categories. This is a **direct, mechanical mapping from
tests to points** — every TODO you correctly implement moves you toward
100%. There's no partial credit *within* a single test (each test either
passes or fails), but each category has multiple tests, so partially
correct work still earns partial credit at the category level.

## What each category is actually checking

**VAE Implementation (20%)** — Architecture quality (`VanillaAutoencoder`,
`DenoisingAutoencoder` shapes/behavior), the reparameterization trick
(`VAE.reparameterize`: correct formula, stochastic, differentiable), the
VAE loss (reconstruction + closed-form KL divergence, matching the
mathematically exact expected value), and latent-space interpolation.

**GAN Implementation (30%)** — Vanilla GAN (given, verifies your test
environment), DCGAN's best-practice convolutional architecture, adversarial
BCE loss matching the exact formula (not just "does it run"), Conditional
GAN's label-conditioning trick, and WGAN/WGAN-GP's Wasserstein loss +
gradient penalty (checked against closed-form values for a linear critic —
a real correctness test, not just a shape check).

**Advanced Techniques (20%)** — Pix2Pix's U-Net skip connections (checked
by verifying the skip tensor survives concatenation untouched), CycleGAN's
residual-block skip connection, and every Phase 3 loss function
(adversarial + L1 for Pix2Pix; cycle-consistency + identity for CycleGAN)
matching their exact mathematical formulas.

**Evaluation & Analysis (15%)** — FID's closed-form Fréchet distance
(checked against known values: ~0 for identical distributions, matching a
known mean-shift formula when covariances match) and Inception Score's
per-split KL-divergence formula (checked against the known IS=1 baseline
for uninformative uniform predictions).

**Platform Development (10%)** — The unified model registry
(`generate_samples` correctly branching between VAE sampling and GAN latent
sampling, and being seed-reproducible) and high-resolution export
(`export_image` correctly upscaling and saving a real PNG file).

**Portfolio & Presentation (5%, manual)** — Your instructor reviews:
artwork diversity and quality in `outputs/portfolio/` (50+ images), and
whether your video presentation clearly demonstrates the platform and
explains your model comparisons.
