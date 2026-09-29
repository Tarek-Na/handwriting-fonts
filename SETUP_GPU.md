# Setting up a GPU machine

Everything needed to go from a fresh computer with an NVIDIA GPU to training
the recommended model and comparing it with the current one. The code comes
from GitHub; the private files (handwriting photos, the current model) come
separately and must never be committed, because this repository is public.

## What you need

- An NVIDIA GPU (8 GB of memory or more is comfortable) with a current driver.
- Python 3.11 or newer, and git.
- About 15 GB of free disk: 1–4 GB for the font corpus, 3.7 GB for the
  rendered training data, ~1 GB for the NIST handwriting archives, and room
  for checkpoints (about 0.5 GB each; only the latest and best are kept).
- `hfont_private_files.zip`, copied over by hand (USB stick or a private
  drive link — not a public share).

## 1. Code and private files

```bash
git clone https://github.com/Tarek-Na/handwriting-fonts.git
cd handwriting-fonts
# unzip hfont_private_files.zip here, at the project root: it restores the
# photos, runs/phase1c/hfont_step030000.pt and data/fonts/ at their paths
```

Check that git does not see the photos: `git status` must list nothing new.

## 2. Python environment

Install PyTorch built for the machine's CUDA version first — pick the command
at <https://pytorch.org/get-started/locally/> — then the project:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q          # all pass; tests needing a glyph cache skip
```

## 3. Training data (≈30 minutes, faster with more CPU cores)

Rebuild it rather than copy it: the rasterizer produces byte-identical glyphs
on any machine, so this is the same data the earlier runs used.

```bash
python -m hfont.cli fetch --dest data/google-fonts
python -m hfont.cli index data/google-fonts --out data/manifest.json
python -m hfont.cli prepare --manifest data/manifest.json --out data/cache
```

## 4. Train the recommended recipe

Reference attention plus the style-contrastive loss, 40,000 steps, with
handwriting fonts sampled 6× instead of 3×:

```bash
python -m hfont.cli train --recipe recommended --cache data/cache --out runs/hw40k \
    --set steps=40000 --set data.handwriting_weight=6.0
```

On a Colab T4 this ran at 77 images/s, about 4.5 hours; a recent desktop GPU
should be faster. Checkpoints are written every 2,000 steps, and re-running
the same command resumes from the last one after any interruption. Held-out
IoU is logged every 2,000 steps (0.478 at step 2,000 on the T4 run).

## 5. Compare it with the current model

```bash
# every photographed writer: leave-one-out scores, plus a sheet per writer of
# what they wrote, what the model drew blind, and the letters never written
python scripts/eval_writers.py --tag _phase1c                                  # current model
python scripts/eval_writers.py --checkpoint runs/hw40k/best.pt --tag _hw40k   # new model

# held-out corpus fonts, against the content-copy baseline
python -m hfont.cli report --checkpoint runs/hw40k/best.pt --cache data/cache --split test
```

The sheets land in `output/writers/` (private, ignored by git). The question
they answer is the one the README leaves open: does the model now draw a
writer's own shapes — a two-stroke `k`, a personal `r` — or still a typeset
letter in their weight?

## 6. Real handwriting data (NIST Special Database 19)

For the fine-tuning step after the recommended run. Two archives, ~1.1 GB:

```bash
mkdir -p data/sd19
curl -o data/sd19/by_write.zip https://s3.amazonaws.com/nist-srd/SD19/by_write.zip   # 568 MB, per writer
curl -o data/sd19/by_field.zip https://s3.amazonaws.com/nist-srd/SD19/by_field.zip   # 541 MB, per letter
```

`by_write` holds each of 3,669 writers' digits, capitals, lowercase and a
Constitution paragraph, but its images carry no letter labels; `by_field`
holds the same images sorted by letter, so labels are recovered by matching
identical images between the two. The converter into the training cache is
not written yet. `data/sd19/` is ignored by git.
