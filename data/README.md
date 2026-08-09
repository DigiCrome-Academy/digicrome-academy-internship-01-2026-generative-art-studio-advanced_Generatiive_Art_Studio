# data/

Put downloaded datasets here (e.g. `data/celeba/`, `data/wikiart/`). This
folder is git-ignored — never commit raw image datasets to the repository.

See [docs/SETUP.md](../docs/SETUP.md) and `scripts/download_data.py` for
how to fetch CelebA and WikiArt.

The auto-graded test suite does **not** read from this folder — it uses
`generative_art_studio.data.SyntheticImageDataset` so CI runs offline.
