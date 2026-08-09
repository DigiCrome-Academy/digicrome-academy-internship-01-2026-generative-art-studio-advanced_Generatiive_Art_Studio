# Technical Report Template

Target length: **15–18 pages**. Replace every bracketed prompt below with
your own content; delete prompts you've addressed. Include figures
(training curves, generated samples, latent-space walks, FID/IS tables) —
a report this length without figures will read as too thin.

## 1. Executive Summary (0.5–1 page)

[What did you build? What were the headline results — best FID, most
interesting qualitative finding, what the platform can do?]

## 2. Introduction (1 page)

- Problem statement and creative/business motivation (see docs/PROJECT_BRIEF.md's "Business Context")
- Datasets used (CelebA / WikiArt / other) and preprocessing
- Report roadmap

## 3. Autoencoder Foundation (2–3 pages)

- Vanilla AE architecture and reconstruction quality
- Denoising AE: noise model used, restoration quality vs. vanilla AE
- VAE architecture, the reparameterization trick (explain it — don't just
  cite it), and your loss curves (reconstruction vs. KL terms)
- Latent space: interpolation results (include image grids), what
  structure you observed (e.g. smooth semantic transitions)

## 4. GAN Development (3–4 pages)

- Vanilla GAN vs. DCGAN: architecture differences and why DCGAN's
  best practices (strided convs, BatchNorm, LeakyReLU) improve stability
- Conditional GAN: how conditioning changed generation control, examples
- **Mode collapse:** did you observe it? How did you detect it (e.g. low
  sample diversity, discriminator loss collapsing to near-zero)? What did
  you try to fix it?
- **Training instability:** loss curves for G and D, what oscillation/
  divergence looked like, and how WGAN(-GP) compared
- WGAN vs. BCE-GAN: Wasserstein loss behavior, gradient penalty vs. weight
  clipping, training curve comparison

## 5. Advanced GANs & Style Transfer (3–4 pages)

- Pix2Pix: paired translation task you chose, U-Net skip-connection
  rationale, qualitative results, L1 vs. adversarial loss weighting
  ablation if you ran one
- CycleGAN: unpaired style-transfer task, cycle-consistency +
  identity loss behavior, side-by-side style-transfer examples
- Any progressive-growing / higher-resolution experiments

## 6. Evaluation & Analysis (2–3 pages)

- FID and Inception Score results across **all** your models — one
  comparison table
- Discussion: does FID/IS ranking match your qualitative/visual judgment?
  Where do they disagree, and why might that be?
- **VAE vs. GAN comparison** (explicitly required): sample diversity,
  training stability, sample sharpness, ease of use for controlled
  generation (e.g. interpolation, conditioning) — pick a winner per use
  case, not an overall winner

## 7. Generative Art Platform (1–2 pages)

- Architecture of `src/generative_art_studio/app/`
- Feature walkthrough: generation controls, interpolation/style mixing,
  gallery, high-resolution export
- Screenshots

## 8. Challenges & Lessons Learned (1 page)

- Hardest bug / hardest concept, and how you resolved or worked around it
- What you'd do differently with more time/compute

## 9. Conclusion (0.5 page)

- Summary of contributions and results
- Future work (StyleGAN-style adaptive normalization? Higher resolution?
  Deploying the platform?)

## Appendix

- Hyperparameters per model (table)
- Additional generated samples
- Links: repo, portfolio folder, video presentation
