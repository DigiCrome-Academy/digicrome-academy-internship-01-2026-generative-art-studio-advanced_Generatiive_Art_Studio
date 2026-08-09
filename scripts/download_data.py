#!/usr/bin/env python
"""Dataset download helper for the Advanced Generative Art Studio project.

This script is **not** run in CI (the grading suite uses synthetic data —
see `generative_art_studio.data.SyntheticImageDataset`). Run it locally or
in Colab once you're ready to train on real data.

Usage:
    python scripts/download_data.py --dataset celeba --out data/celeba
    python scripts/download_data.py --dataset wikiart --out data/wikiart

Both datasets require a free Kaggle account and API token
(~/.kaggle/kaggle.json) — see docs/SETUP.md for step-by-step instructions.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "celeba": {
        "kaggle_slug": "jessicali9530/celeba-dataset",
        "kind": "dataset",
        "notes": "202,599 aligned & cropped celebrity face images. ~1.3GB zipped.",
    },
    "wikiart": {
        "kaggle_slug": "painter-by-numbers",
        "kind": "competition",
        "notes": "WikiArt paintings for style transfer (Kaggle 'Painter by Numbers'). Large (~30GB) — a subset is enough for this project.",
    },
}


def check_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "The 'kaggle' CLI is not installed/on PATH.\n"
            "Run: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json (see docs/SETUP.md)."
        )


def download(dataset_key: str, out_dir: Path) -> None:
    info = DATASETS[dataset_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset_key}' ({info['notes']}) into {out_dir} ...")
    if info["kind"] == "dataset":
        cmd = ["kaggle", "datasets", "download", "-d", info["kaggle_slug"], "-p", str(out_dir), "--unzip"]
    else:
        cmd = ["kaggle", "competitions", "download", "-c", info["kaggle_slug"], "-p", str(out_dir)]
    subprocess.run(cmd, check=True)
    print(f"Done. Point `get_image_dataset('{out_dir}')` (see src/generative_art_studio/data/datasets.py) at this folder.")
    print("Reminder: torchvision.datasets.ImageFolder needs at least one subdirectory of images "
          "under the root — e.g. move loose images into `<out_dir>/all/`.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    check_kaggle_cli()
    download(args.dataset, args.out)


if __name__ == "__main__":
    main()
