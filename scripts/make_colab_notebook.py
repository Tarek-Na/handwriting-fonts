"""Generate notebooks/colab_train.ipynb: the baseline and G3 on a paid Colab GPU.

The notebook is generated rather than hand-edited because a .ipynb is JSON with
source stored as line lists, which is painful to diff and easy to corrupt by
hand. Edit this file and re-run it.

What the notebook runs (instruction.md §4 and §5.3, and the November 2026 GPU
report in reports/): rebuild the baseline (lost with the laptop), fine-tune it
with SD19's real writers (G3), then score the shipped model, the baseline and
G3 the same way. Every step can be re-run: anything already on Drive is
skipped, and training resumes from Drive after a disconnect.
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
# Handwriting fonts: baseline and G3 on Colab

Rebuilds the baseline model, fine-tunes it on real handwriting (NIST SD19,
experiment G3), and scores both against the shipped model.

**Before the first run**
1. Runtime → Change runtime type → **L4 GPU** (an A100 also works, faster and dearer).
2. Put `hfont_private_files.zip` in Google Drive at `MyDrive/hfont/`. It holds the
   ten handwriting photos, the shipped model and the Noto Sans font. It is private:
   Drive only, never GitHub.

**If the session disconnects**, run every cell again from the top. Finished
steps are skipped (the data caches are on Drive) and training resumes from its
last checkpoint on Drive.

Rough cost on an L4: data ~40 min the first time, baseline ~45 min, G3 ~35 min,
scoring ~20 min, about 4-6 compute units in all.
"""
    ),
    md("## 1. GPU and Drive"),
    code(
        """
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > L4 GPU'

from google.colab import drive
drive.mount('/content/drive')

import pathlib
DRIVE = pathlib.Path('/content/drive/MyDrive/hfont')
for sub in ('data', 'runs', 'results'):
    (DRIVE / sub).mkdir(parents=True, exist_ok=True)
assert (DRIVE / 'hfont_private_files.zip').exists(), 'Put hfont_private_files.zip in MyDrive/hfont/'
print('Drive folder:', DRIVE)
"""
    ),
    md("## 2. Code, dependencies and private files\n\nThe code is cloned fresh to local disk each session."),
    code(
        """
%cd /content
!rm -rf /content/handwriting-fonts
!git clone -q https://github.com/Tarek-Na/handwriting-fonts.git
%cd /content/handwriting-fonts
!pip install -q -e .
!unzip -oq "{DRIVE}/hfont_private_files.zip" -d /content/handwriting-fonts
!ls runs/phase1c data/fonts
!git log --oneline -1
"""
    ),
    md(
        """
## 3. Font training data (first session only)

Clones Google Fonts and renders it into the glyph cache, then keeps the cache
on Drive as one archive. Later sessions only unpack it (a minute or two).
Training reads it from local disk, because reading through Drive is slow.
"""
    ),
    code(
        """
import os, subprocess
CACHE_TAR = DRIVE / 'data' / 'cache.tar'
if not CACHE_TAR.exists():
    !python -m hfont.cli fetch --dest /content/google-fonts
    !python -m hfont.cli index /content/google-fonts --out /content/manifest.json
    !python -m hfont.cli prepare --manifest /content/manifest.json --out /content/cache --workers {os.cpu_count()}
    !tar -cf "{CACHE_TAR}.part" -C /content cache && mv "{CACHE_TAR}.part" "{CACHE_TAR}"
    !cp /content/manifest.json "{DRIVE}/data/manifest.json"
elif not os.path.exists('/content/cache/meta.json'):
    !tar -xf "{CACHE_TAR}" -C /content
!du -sh /content/cache
"""
    ),
    md(
        """
## 4. SD19 real-handwriting data (first session only)

Downloads SD19's per-writer images straight from NIST, pulls its labels out of
the 1995 edition, and adds ~3,400 writers to **a copy** of the font cache.
SD19 must stay out of the baseline's cache: `data.sd19_weight` defaults to 1.0,
so it would silently enter the baseline (lesson from the laptop runs).
"""
    ),
    code(
        """
SD19_TAR = DRIVE / 'data' / 'cache_sd19.tar'
if not SD19_TAR.exists():
    !mkdir -p data/sd19
    !curl -sS -C - --retry 5 -o data/sd19/by_write.zip https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
    !python scripts/sd19_labels.py --fetch | tail -12
    !rm -rf /content/cache_sd19 && cp -r /content/cache /content/cache_sd19
    !python scripts/sd19_cache.py --cache /content/cache_sd19
    !tar -cf "{SD19_TAR}.part" -C /content cache_sd19 && mv "{SD19_TAR}.part" "{SD19_TAR}"
elif not os.path.exists('/content/cache_sd19/meta.json'):
    !tar -xf "{SD19_TAR}" -C /content
!du -sh /content/cache_sd19
"""
    ),
    md(
        """
## 5. Baseline (replaces the laptop's lost hw40k)

The recommended recipe with handwriting fonts sampled 6×. On the laptop the
40K-step run was best at step 14K and then over-fitted the fonts, so this runs
16K. Checkpoints go to Drive; re-running this cell resumes.
"""
    ),
    code(
        """
BASE = DRIVE / 'runs' / 'base'
!python -m hfont.cli train --recipe recommended --cache /content/cache --out "{BASE}" --set steps=16000 --set data.handwriting_weight=6.0 --set num_workers=4
"""
    ),
    md(
        """
## 6. G3: fine-tune on fonts plus SD19's real writers

One variable changed from the baseline: real handwriting in the training mix.
Warm-started from the baseline's best checkpoint.
"""
    ),
    code(
        """
G3 = DRIVE / 'runs' / 'g3'
!python -m hfont.cli train --recipe recommended --cache /content/cache_sd19 --out "{G3}" --set init_from="{BASE}/best.pt" --set steps=10000 --set data.handwriting_weight=6.0 --set data.sd19_weight=1.0 --set num_workers=4
"""
    ),
    md(
        """
## 7. Score every model the same way

For each model: leave-one-out on the ten photographed writers with the style
numbers (pass: similarity across writers ≤ 0.67, attribution ≥ 25%, clDice ≥
0.36), then the nearest-font floor on held-out SD19 writers. The floor here
retrieves from the Google Fonts cache, so its value differs from the 0.576 /
0.590 of earlier machines: compare each model with the shipped model **on this
same run**, not with old numbers. `best.pt` and `last.pt` are both scored,
because `best.pt` is chosen by font IoU, which peaks early.

The letter sheets go to Drive (`results/`): private, they show real handwriting.
"""
    ),
    code(
        """
MODELS = {
    'shipped': 'runs/phase1c/hfont_step030000.pt',
    'base_best': f'{BASE}/best.pt', 'base_last': f'{BASE}/last.pt',
    'g3_best': f'{G3}/best.pt', 'g3_last': f'{G3}/last.pt',
}
OUT = DRIVE / 'results'
for tag, ckpt in MODELS.items():
    if not os.path.exists(ckpt):
        print(tag, 'missing, skipped'); continue
    print('=' * 20, tag)
    !python scripts/eval_writers.py --checkpoint "{ckpt}" --out "{OUT}/writers" --tag "_{tag}" 2>&1 | grep -E "mean|style"
    !python scripts/compose_eval.py --cache /content/cache_sd19 --fonts /content/cache --checkpoint "{ckpt}" --limit 200 2>&1 | tail -2 | tee "{OUT}/floor_{tag}.txt"
"""
    ),
    code(
        """
import json, pandas as pd
rows = []
for tag in MODELS:
    path = OUT / 'writers' / f'summary_{tag}.json'
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    style = s.pop('_style', {})
    floor = (OUT / f'floor_{tag}.txt').read_text().strip().splitlines()[-1] if (OUT / f'floor_{tag}.txt').exists() else ''
    rows.append({'model': tag,
                 'LOO clDice': round(sum(v['cl_dice'] for v in s.values()) / len(s), 3),
                 'alike across writers': round(style.get('generated_similarity', float('nan')), 3),
                 'attributed': f"{style.get('attribution', float('nan')):.1%}",
                 'floor (every letter not on the sheet)': floor})
pd.set_option('display.max_colwidth', 120)
pd.DataFrame(rows)
"""
    ),
    md(
        """
Copy this table into `reports/` as text and commit it (numbers only, no images).
If G3 passes, the next arms are G4 and G5 (instruction.md §5.3) on top of it.
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
            "colab": {"provenance": [], "gpuType": "L4", "toc_visible": True},
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
