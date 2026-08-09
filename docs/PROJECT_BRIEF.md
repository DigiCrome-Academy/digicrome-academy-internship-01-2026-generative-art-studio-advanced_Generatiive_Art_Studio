# Advanced Generative Art Studio — Project Brief

## Project Overview

Students will build a comprehensive generative art platform combining
Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs)
for creative image synthesis, style transfer, and artistic content
generation. This project explores the cutting edge of generative models.

## Business Context

Generative AI is transforming creative industries — from graphic design to
game development and digital art. This project simulates building a
professional creative toolkit that enables artists and designers to
leverage AI for generating novel artwork, transferring styles, and creating
high-quality synthetic content.

## Learning Objectives

This project covers Month 5 advanced generative models:

- Autoencoder architectures (Vanilla, Denoising, Variational)
- VAE training and latent space manipulation
- GAN fundamentals (Generator, Discriminator, Adversarial loss)
- GAN variants (DCGAN, Conditional GAN, StyleGAN, Pix2Pix)
- Addressing GAN challenges (mode collapse, training instability)
- Evaluation metrics (FID, Inception Score)
- Differences between VAEs and GANs for various applications

## Datasets

- **Primary:** CelebA (Celebrity Faces) — 202,599 face images.
  Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Alternative:** LSUN Bedrooms or WikiArt (for style transfer)
- **Style Transfer:** WikiArt — https://www.kaggle.com/c/painter-by-numbers

See [SETUP.md](SETUP.md) and `scripts/download_data.py` for how to fetch
these. **You do not need real data to pass the auto-graded test suite** —
it uses synthetic tensors so it can run offline in CI. You do need real
data for the notebooks, portfolio, and platform deliverables.

## Technical Requirements

### Phase 1: Autoencoder Foundation (Week 1)

- Build vanilla autoencoder for image compression
- Implement denoising autoencoder for image restoration
- Develop Variational Autoencoder (VAE) with reparameterization trick
- Explore latent space interpolation and visualization
- Generate new samples by sampling from learned latent distribution

`src/generative_art_studio/models/autoencoders/`, `notebooks/01_VAE_Implementation.ipynb`

### Phase 2: GAN Development (Week 2)

- Implement vanilla GAN architecture
- Build Deep Convolutional GAN (DCGAN) with best practices
- Develop Conditional GAN for controlled generation
- Monitor and address mode collapse issues
- Implement Wasserstein GAN for training stability

`src/generative_art_studio/models/gans/`, `notebooks/02_Basic_GANs.ipynb`

### Phase 3: Advanced GANs & Style Transfer (Week 3)

- Implement Pix2Pix for image-to-image translation
- Build CycleGAN for unpaired style transfer
- Experiment with progressive growing techniques
- Evaluate generated images using FID and Inception Score

`src/generative_art_studio/models/advanced/`, `src/generative_art_studio/evaluation/`,
`notebooks/03_Advanced_GANs.ipynb`, `notebooks/04_Style_Transfer.ipynb`

### Phase 4: Generative Art Platform (Week 4)

- Create unified interface for all generative models
- Implement real-time generation with user controls
- Add style mixing and interpolation features
- Build gallery system to showcase generated artwork
- Implement export functionality for high-resolution outputs

`src/generative_art_studio/app/streamlit_app.py`

## Deliverables

1. **Jupyter Notebooks (4 notebooks):** VAE implementation, Basic GANs,
   Advanced GANs, Style transfer — `notebooks/`
2. **Generative Art Platform:** Interactive web application with all
   generation capabilities — `streamlit run src/generative_art_studio/app/streamlit_app.py`
3. **Technical Report (15–18 pages):** VAE vs GAN comparison, architecture
   details, training challenges, evaluation metrics — template in
   [TECHNICAL_REPORT_TEMPLATE.md](TECHNICAL_REPORT_TEMPLATE.md)
4. **Generated Art Portfolio:** Curated collection of 50+ generated images
   showcasing model capabilities — `outputs/portfolio/`
5. **Video Presentation (10–12 minutes):** Platform demonstration, model
   comparisons, artistic possibilities

## Evaluation Criteria

| Component | Weight | Evaluation Criteria |
|---|---|---|
| VAE Implementation | 20% | Architecture quality, latent space control, generation diversity |
| GAN Implementation | 30% | Multiple variants, training stability, image quality |
| Advanced Techniques | 20% | Style transfer, conditional generation, mode collapse handling |
| Evaluation & Analysis | 15% | FID/IS metrics, comparative analysis, critical assessment |
| Platform Development | 10% | User interface, feature richness, artistic utility |
| Portfolio & Presentation | 5% | Generated artwork quality, demonstration effectiveness |

See [RUBRIC.md](RUBRIC.md) for how the first five rows map to auto-graded
pytest markers, and what "Portfolio & Presentation" needs for manual review.

## Expected Learning Outcomes

- Deep understanding of autoencoder architectures and VAEs
- Mastery of GAN training and common pitfalls
- Proficiency in multiple GAN variants and their applications
- Understanding of generative model evaluation metrics
- Experience building creative AI applications
