# Setup Guide

## 1. Local environment (CPU or GPU)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the default PyPI PyTorch build. If you have an
NVIDIA GPU and want CUDA acceleration for real training runs (not needed
for the auto-graded tests), install the matching CUDA build from
https://pytorch.org/get-started/locally/ **before** `pip install -r requirements.txt`,
e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Google Colab

Upload/clone the repo into Colab, then:

```python
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "src")
```

Colab's free-tier T4 GPU is enough for every model in this project at
64x64 resolution.

## 3. Getting the datasets

The auto-graded tests never need real data. For the notebooks, platform,
and portfolio deliverables, you do:

1. Create a free Kaggle account and API token: https://www.kaggle.com/settings → "Create New Token" → save `kaggle.json` to `~/.kaggle/kaggle.json` (`C:\Users\<you>\.kaggle\kaggle.json` on Windows).
2. `pip install kaggle`
3. `python scripts/download_data.py --dataset celeba --out data/celeba`
   or `python scripts/download_data.py --dataset wikiart --out data/wikiart`

`data/` is git-ignored — never commit raw datasets to the repo.

## 4. Running things

```bash
# Full auto-graded test suite (what CI runs)
pytest -v

# One rubric category at a time
pytest -m vae -v

# Your weighted grade, computed locally
python scripts/grade.py

# The Phase 4 platform
streamlit run src/generative_art_studio/app/streamlit_app.py

# A notebook
jupyter notebook notebooks/01_VAE_Implementation.ipynb
```

## 5. Common issues

- **`ModuleNotFoundError: generative_art_studio`** — run pytest/streamlit
  from the repo root (the `tests/conftest.py` and `streamlit_app.py`
  already add `src/` to `sys.path`, but running from the wrong directory
  can still break relative dataset paths).
- **CI is slow / times out** — it shouldn't be; the whole suite uses tiny
  synthetic batches and runs on CPU in well under a minute. If your local
  run is slow, check you haven't accidentally pointed a test at real data.
- **`streamlit` errors about missing script context** — that's expected
  when importing `streamlit_app.py` directly (e.g. in a test); always use
  `streamlit run ...` to launch it.
- **`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
  already initialized`** (Windows, common with an Anaconda/Miniconda
  Python) — this is a conflict between conda's MKL-linked NumPy/SciPy and
  PyTorch's bundled OpenMP runtime, not a bug in your code (it can crash
  the process with no Python traceback, e.g. mid-`compute_fid`). Fix it
  properly by using a plain `venv` instead of a conda env for this project
  (see step 1 above); if you must use conda, the quick workaround is
  `set KMP_DUPLICATE_LIB_OK=TRUE` (PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`)
  before running pytest — safe for coursework, not something to ship.
