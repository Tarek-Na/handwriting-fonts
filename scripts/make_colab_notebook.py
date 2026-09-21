"""Generate notebooks/colab_train.ipynb.

The notebook is generated rather than hand-edited because a .ipynb is JSON with
source stored as line lists, which is painful to diff and easy to corrupt by
hand. Edit this file and re-run it.
"""

from __future__ import annotations

import json
import pathlib

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "colab_train.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").rstrip(),
    }


CELLS = [
    md(
        """
# Few-Shot Handwriting Font Generation - Phase 1 (Latin)

Trains the glyph generator on a free Colab GPU.

**The thing that shapes this whole notebook:** a free Colab runtime can be
reclaimed at any moment, and when it is, the local disk goes with it. So the
project directory, the glyph cache and the checkpoints all live on **Google
Drive**, and training resumes from the last checkpoint automatically. If the
session dies, re-run this notebook top to bottom and it continues where it
stopped rather than starting over.

The Google Fonts corpus is the deliberate exception. It is cloned to *local*
disk every session, because it is ~1.2 GB, re-cloning takes about two minutes
on Colab's connection, and Drive is slow enough that reading training data
through it would throttle every step.
"""
    ),
    md("## 1. Check the GPU\n\nRuntime -> Change runtime type -> T4 GPU, before anything else."),
    code(
        """
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU."
"""
    ),
    md("## 2. Mount Drive\n\nCheckpoints, cache and sample images live here so they survive a disconnect."),
    code(
        """
from google.colab import drive
drive.mount('/content/drive')

import pathlib
PROJECT = pathlib.Path('/content/drive/MyDrive/DeepLearningProject')
PROJECT.mkdir(parents=True, exist_ok=True)
CACHE = PROJECT / 'data' / 'cache'
RUN = PROJECT / 'runs' / 'phase1'
print('project:', PROJECT)
"""
    ),
    md(
        """
## 3. Get the code onto Colab

Two options, pick one.

**A - from GitHub** (best once this is in a repo): set `REPO_URL` and run.

**B - upload a zip**: zip the project on your machine so the archive contains
`src/hfont/`, and upload it when prompted.
"""
    ),
    code(
        """
REPO_URL = ""   # e.g. "https://github.com/you/handwriting-fonts.git"

import shutil, subprocess, sys
CODE = PROJECT / 'code'

if REPO_URL:
    if (CODE / '.git').exists():
        subprocess.run(['git', '-C', str(CODE), 'pull'], check=False)
    else:
        subprocess.run(['git', 'clone', REPO_URL, str(CODE)], check=True)
else:
    from google.colab import files
    print("Upload a zip containing src/hfont/ ...")
    uploaded = files.upload()
    CODE.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(next(iter(uploaded)), str(CODE))

# Locate src/ however the archive happened to be nested.
roots = [p.parent.parent for p in CODE.rglob('hfont/__init__.py')]
assert roots, "could not find src/hfont under " + str(CODE)
SRC = roots[0]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print('source root:', SRC)
"""
    ),
    md("## 4. Dependencies\n\nColab ships torch, numpy and Pillow already; these are the rest."),
    code(
        """
!pip install -q fonttools scikit-image uharfbuzz

import fontTools, skimage
print("fontTools", fontTools.version, "| skimage", skimage.__version__)
"""
    ),
    md(
        """
## 5. Fetch the font corpus

To local disk, not Drive - see the note at the top. `--depth 1` keeps it to a
single revision.
"""
    ),
    code(
        """
import pathlib, subprocess
FONTS = pathlib.Path('/content/google-fonts')
if not (FONTS / 'ofl').is_dir():
    subprocess.run(['git', 'clone', '--depth', '1', '--single-branch',
                    'https://github.com/google/fonts.git', str(FONTS)], check=True)
print(sum(1 for _ in FONTS.rglob('*.ttf')), "ttf files")
"""
    ),
    md(
        """
## 6. Index and render the corpus

Indexing opens every font, keeps the ones that cover the whole charset, and
assigns a train/val/test split **by family** so no weight of a family can leak
across a split boundary.

Rendering is the slow part (~10-20 minutes). It writes to Drive, so it happens
once: later sessions find the cache and skip straight past.
"""
    ),
    code(
        """
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S', force=True)
logging.getLogger('fontTools').setLevel(logging.ERROR)

from hfont.charset import LATIN_CORE
from hfont.data.corpus import build_manifest, Manifest

MANIFEST = PROJECT / 'data' / 'manifest.json'
if MANIFEST.exists():
    manifest = Manifest.load(MANIFEST)
else:
    manifest = build_manifest([FONTS], LATIN_CORE, min_coverage=1.0, max_per_family=4)
    manifest.save(MANIFEST)

print(manifest.summary())
"""
    ),
    code(
        """
import os
from hfont.data.prepare import build_cache, pack_cache
from hfont.data.render import RenderConfig

build_cache(manifest, CACHE, LATIN_CORE, RenderConfig(size=128),
            workers=min(8, os.cpu_count() or 2))

# Consolidate into one memory-mapped array. Without this, a few thousand fonts
# sampled at random thrash the per-worker LRU and every sample decompresses two
# fonts, which leaves the GPU waiting on two CPU cores.
if not (CACHE / 'packed_images.npy').exists():
    pack_cache(CACHE)

total = sum(f.stat().st_size for f in CACHE.rglob('*') if f.is_file())
print('cache size: %.0f MB' % (total / 1e6))
"""
    ),
    md(
        """
## 7. Train

Resumes from `last.pt` automatically. Checkpoints are written every 2000 steps
or 10 minutes, whichever comes first, and atomically - an eviction mid-write
leaves the previous checkpoint intact rather than a truncated file.

Adversarial loss is off until `adversarial_start`. Get the letterforms right
under reconstruction first, then bring the discriminator in to harden the stroke
edges, which is what the vectorizer needs.
"""
    ),
    code(
        """
from hfont.train import TrainConfig, Trainer
from hfont.models.generator import GeneratorConfig
from hfont.models.losses import LossConfig
from hfont.data.dataset import DatasetConfig

cfg = TrainConfig(
    cache_dir=str(CACHE),
    out_dir=str(RUN),
    steps=120_000,
    batch_size=32,
    lr=2e-4,
    betas=(0.9, 0.999),
    num_workers=2,
    amp=True,
    adversarial_start=40_000,
    generator=GeneratorConfig(base_channels=48, style_dim=256),
    loss=LossConfig(l1=10.0, dice=2.0, advance=1.0, adversarial=0.0),
    data=DatasetConfig(n_refs=(1, 8), handwriting_weight=3.0, epoch_size=20_000),
)

Trainer(cfg).train()
"""
    ),
    md("## 8. Look at what it is producing\n\nEach strip is content input / generated / ground truth."),
    code(
        """
import glob
from IPython.display import Image, display

shots = sorted(glob.glob(str(RUN / 'samples' / '*.png')))
for path in shots[-3:]:
    print(path.split('/')[-1])
    display(Image(path))
"""
    ),
    code(
        """
import pandas as pd
df = pd.read_csv(RUN / 'metrics.csv')
if 'val_iou' in df:
    df[df['val_iou'].notna()].plot(x='step', y='val_iou', figsize=(9, 3),
                                   grid=True, title='held-out IoU')
df[df['l1'].notna()].plot(x='step', y=['l1', 'dice'], figsize=(9, 3),
                          grid=True, title='training loss')
"""
    ),
    md(
        """
## 9. Export an installable font

Generates the whole charset from one held-out style, traces it to Bezier
outlines and assembles an `.otf`. This is the end-to-end check that separates a
working system from a grid of plausible images: the output is a file you can
install and type with.
"""
    ),
    code(
        """
from hfont.generate import generate_from_cache_font

out = generate_from_cache_font(
    checkpoint=RUN / 'best.pt',
    cache_dir=CACHE,
    font_id=None,              # None picks the first validation font
    out_path=PROJECT / 'output' / 'generated.otf',
    family='Generated Phase1',
)
print(out)

from google.colab import files
files.download(str(out))
"""
    ),
]


def to_source(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main() -> None:
    notebook = {
        "cells": [dict(cell, source=to_source(cell["source"])) for cell in CELLS],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
