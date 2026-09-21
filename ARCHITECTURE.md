# Architecture, data and training

How the system is put together: what it takes in, what the model is, what it
was trained on, and how a photograph becomes an installable font. Written for
someone who has not read the code. Results and open problems live in
[README.md](README.md); this file explains the machinery.

---

## 1. The problem, in input/output terms

**In:** ~24–30 letters written by one person, photographed.
**Out:** an installable OpenType font (`.otf`) with all 75 characters in that
person's hand, including the ~45–51 they never wrote.

This is *few-shot style transfer with a structural constraint*. The style is
defined by a handful of examples, and the output must be a real font file —
outlines and spacing, not pictures of letters.

Two properties shape every decision below:

1. **The target is deterministic.** For a given (character, hand) there is one
   correct letterform. That justifies a strong reconstruction loss and makes
   adversarial training a refinement rather than the main objective.
2. **The output is vectorized afterwards.** The model's raster is traced into
   Bézier curves, so edge sharpness matters more than it would if the raster
   were the deliverable: a soft edge becomes a wobbly outline in the font.

---

## 2. Data

### 2.1 Where the training data comes from

**Google Fonts**, cloned from `github.com/google/fonts` (~4 GB). There is no
handwriting-sample dataset of the kind this task wants — thousands of hands,
every letter, consistently normalized — so fonts stand in for hands. Each font
is treated as one "writer": a consistent style with ground truth for every
glyph.

After filtering to fonts that cover the full 75-character set:

| | count |
|---|---|
| Fonts | 3,029 |
| Families | 1,999 |
| Train / validation / test | 2,568 / 224 / 237 fonts |
| Rendered glyphs | 227,175 |
| Packed cache | 3.72 GB |

**Splits are by family, not by file.** Roboto ships 18 weights; one weight in
training and another in validation would leak style across the split and report
a meaningless number.

Categories come from each family's `METADATA.pb`: sans-serif 1,262, serif 660,
display 591, **handwriting 387**, monospace 95, unknown 34. The handwriting
category is the slice closest to the real task, so training samples it **3×**
more often than its share of the corpus.

### 2.2 The character set

`LATIN_CORE`, 75 glyphs: `A–Z a–z 0–9` plus `! " & ' ( ) , - . / : ; ?`.

Every glyph is carried as a `(character, contextual form)` pair from the start
— `Form.isol` for Latin. Arabic needs `init/medi/fina` variants of each letter,
so Phase 2 adds table entries rather than changing function signatures.

The **seed sets** are the letters a person is asked to write:

* `seed24` = `AEGHMORS abegkmnorsty 036 ,` — the gate's standard, chosen to
  cover capitals, x-height letters, ascenders, descenders, round and straight
  forms, digits and one punctuation mark.
* `seed30` = seed24 + `BDNfij` — the default for the product command, because
  `i` and `j` are what the model draws worst, and letters the writer provides
  are passed through rather than generated.

### 2.3 Rendering fonts into training images

Every font is rasterized to 128×128 greyscale coverage maps, ink = 1.

**The rasterizer is written from scratch** (`data/raster.py`): a scanline
non-zero-winding fill with exact horizontal coverage and 4× vertical
supersampling. Not FreeType — the corpus is rendered on one machine and trained
on another, and a FreeType version difference would silently change hinting and
stem darkening on every sample. This implementation is byte-identical across
Windows/Python 3.14 and Colab/Python 3.13 (verified by SHA-256).

**Framing is the subtle part** (`data/frame.py`), and it is shared with photo
intake:

* the baseline goes on a **fixed canvas row** (0.75 of the height), because a
  generated glyph has no source font to ask where its baseline was;
* the baseline is *estimated from ink* — the median bottom edge of
  `a c e m n o r s u v w x z` — not taken from the font's designed baseline,
  because a photographed hand has no designed baseline;
* **one scale per font**, computed from the ink extent of the `seed24` letters
  only, never per glyph: relative proportions (x-height against cap height,
  ascender length) *are* the style;
* glyphs are centred horizontally on their own ink.

Using the same definition on both sides is not cosmetic. When the renderer used
the font's designed metrics and intake estimated its own, photographed
handwriting reached the model **17% larger** than the same font had been seen
in training.

Renders are cached as a memory-mapped `uint8` array plus an index
(`packed_images.npy`, `packed_index.npz`), with per-glyph advance widths and
pixel hashes.

### 2.4 How a training example is drawn

One sample = (content image, K references, character id) → (target image,
target advance).

* Pick a font, weighted 3× if it is handwriting.
* Pick a target glyph from it.
* Pick **K ∈ [1, 8] references** from *other* glyphs of the same font. Variable
  K is what lets a model accept ~24–30 references at inference.
* Pick the **content image**: the same character rendered in a *different,
  randomly chosen* font. Randomizing it is why the model ignores the content
  font's stroke weight — measured directly: swapping Light for Bold content
  changes output ink by 0.05%.

**The target is never among the references**, and the exclusion is by *rendered
pixels*, not character name: fonts routinely point several characters at one
outline, and without this the task degenerates into copying.

---

## 3. The model

Trained **from scratch**. No pretrained backbone, no diffusion model, no
external checkpoint. 128×128 single-channel glyphs are far from the natural
image statistics ImageNet-style pretraining provides, and the corpus is large
enough (227k glyphs) to train a small network directly.

### 3.1 Generator — 31.36M parameters

A content/style-disentangled encoder–decoder with AdaIN conditioning.

```
 references (B,K,1,128,128) ──► Style encoder ──► style vector (256)
                                                        │
 content image (B,1,128,128) ─► Content encoder ─► bottleneck (384,8,8)
                                    │ skips              │
 character id ───────► embedding (64) ──► concat ──► 1×1 conv
                                                        │
                                            AdaIN ResBlocks ×4
                                                        │
                                       Upsample ×4 (+ skips, AdaIN)
                                                        │
                                          tanh ──► glyph raster 128×128
 style (256) + char (64) ──► MLP ──────────────────────► advance width
```

| Part | Params | What it does |
|---|---|---|
| Style encoder | 3.79M | 5 stride-2 conv blocks → global pool per reference → **masked mean ⊕ masked max** over the K references → 2-layer MLP → 256-d style vector |
| Content encoder | 12.83M | stem + 4 stride-2 blocks (128→8) + 4 residual blocks; keeps 4 skip tensors |
| Decoder | 14.52M | 4 AdaIN residual blocks, then 4 upsampling blocks that take skips and are modulated by the style at every scale; 7×7 conv to one channel, `tanh` |
| Char embedding | 75 × 64 | identity, merged into the bottleneck by a 1×1 conv |
| Advance head | small MLP | style ⊕ char embedding → advance width |

Normalization is **InstanceNorm** throughout (per-sample, so batch composition
never leaks between glyphs); style modulation is AdaIN, i.e. the style vector
predicts per-channel scale and shift.

Design decisions worth stating plainly:

* **Why a content image at all?** With a few thousand fonts there is not enough
  data to learn letterforms from a class embedding alone. A skeleton to deform
  is a much easier starting point than a blank canvas. The character embedding
  rides alongside it because a content glyph can be ambiguous (`l`/`I`, `O`/`0`).
* **Why pool the references instead of attending over them?** Pooling is
  exactly permutation-invariant and indifferent to how many references arrive,
  which is what lets a model trained on 1–8 references accept 30. The cost is
  identified in README known issue 1: with one pooled vector the decoder cannot
  ask "which of this writer's letters should *this* stroke resemble", and that
  uncertainty is why it hedges with thicker strokes.
* **Mean ⊕ max pooling**: the mean describes the hand on average, the max
  catches a feature present in only one or two references (a single sharp
  terminal, one unusually long descender).
* **Why predict advance widths?** A raster alone cannot be assembled into a
  font. Without per-glyph advances, exported text is set at uniform pitch and
  looks obviously wrong however good the letterforms are.

One engineering note that mattered: the style encoder runs on the *valid*
references only, but the count is rounded up to a multiple of 32. Encoding the
exact number changes the batch shape nearly every step, and with
`cudnn.benchmark` on, cuDNN re-autotunes per shape — which took training from
~200 images/s to **9** on a T4. Bucketing restored it.

### 3.2 Discriminator — 2.57M parameters

A PatchGAN with spectral normalization: 4 stride-2 conv scales (128→8) → a
patch realism map rather than one scalar, because local stroke texture is what
it is meant to fix and a patch verdict gives a gradient at every stem edge.
Two auxiliary heads hang off the pooled features: a character classifier (used,
weight 0.1) and a font classifier (available, unused). It appeared only in the
last 40% of run 3 (§4.3) and is not part of the current best recipe.

---

## 4. Training

### 4.1 Objective

```
L = 10·L1_banded  +  2·Dice_soft  +  1·L1_advance   [+ 0.1·hinge + 0.1·char_CE]
```

* **Banded L1** — L1 weighted 4× inside a band around every stroke, where the
  band is the target ink *dilated* by 2px. The dilation makes it symmetric.
  Weighting the target ink alone is one-sided: missing ink costs 4× what extra
  ink costs, so wherever the model is unsure where an edge is, painting too much
  is the cheaper bet. Trained that way, run 1 produced **1.51× the true ink** —
  every font came out bolder than the hand it was imitating.
* **Soft Dice** — set overlap, scale-free in the amount of ink, so a thin `l`
  contributes as much shape signal as a heavy `M`. Measured to be the source of
  the residual 11–14% over-inking *and* worth keeping: removing it cuts ink to
  1.047 but drops tol-F1 from 0.801 to 0.780, and worst-case fonts much further
  (README known issue 1).
* **Advance L1** — on the predicted width.
* **Component weighting** (run 3 on) — a per-pixel weight map that boosts
  connected components smaller than half the glyph's largest part, up to 8×,
  which puts an `i` dot at 2.5–4×. L1 converges to the per-pixel median; with a
  dot's position uncertain across hands, that median is blank, which is why the
  model used to omit dots entirely.
* **Adversarial + character CE** (optional, late) — hinge loss from the patch
  discriminator plus an auxiliary classification term, so a dotless `i` is
  penalized for not being recognizable as an `i`.

### 4.2 Setup

| | |
|---|---|
| Optimizer | Adam, β = (0.9, 0.999), no weight decay |
| LR | 2e-4 (runs 1–2), 1e-4 (run 3), warmup then cosine to 5% |
| Batch | 32 glyphs |
| Precision | AMP (fp16 autocast) with gradient scaling |
| Grad clip | 5.0 |
| EMA | 0.999 — inference always uses the EMA weights, because step-to-step jitter in glyph edges becomes visible wobble once traced |
| Hardware | one Colab T4, ~110 images/s |
| Evaluation | every 2,000 steps on held-out fonts; checkpoints are atomic |

### 4.3 What was actually trained

Three runs, each warm-started from the previous, **40K steps total (~4 GPU
hours)**:

| Run | Steps | Change | Result (validation) |
|---|---|---|---|
| 1 | 10K | first training | IoU 0.497, ink **1.51** — every glyph bold |
| — | — | fixed the one-sided ink weighting; unified framing with intake | |
| 2 | 16K | re-rendered corpus, symmetric band | IoU 0.565, ink 1.15 |
| 3 | 14K | component weighting; adversarial phase from step 8.4K | IoU 0.571, ink 1.13 |
| ablation | 3 × 4K | Dice 2.0 / 0.5 / 0.0 | Dice confirmed as the ink source, and confirmed worth keeping |

The current weights are `runs/phase1c/hfont_step030000.pt` (31.36M params,
fp16 EMA, 61 MB).

Two results that shaped what to do next: the **adversarial phase changed almost
nothing** at weight 0.1 over 5,600 steps, and a control arm showed 4,000 more
steps of the current recipe buys +0.001 tol-F1. This recipe is finished
improving; the next gain has to come from the architecture (the pooled style
vector), not from tuning.

---

## 5. Inference: photograph → font

```
photo.jpg
   │  intake.py  — find the sheet, measure ink, cut one crop per letter,
   │               all in one coordinate frame
   ▼
30 normalized 128×128 references
   │  generate.py — encode style once, then decode all 75 characters
   ▼
75 rasters + advance widths
   │  vector/trace.py — raster → cubic Bézier outlines
   ▼
fontbuild/build.py — outlines → CFF OpenType, + synthesized space
   ▼
my_hand.otf          (+ leave-one-out quality score)
```

### 5.1 Intake — two paths

**Printed template** (`build_template` / `extract_cells`): a page with red
dropout guides and four black registration squares. Markers are located, a
projective transform undoes the perspective, and cells are cut from the **red
channel** so the printed guides vanish and only the writing remains.

**Freehand, no printer** (`freehand_cells`): letters written in rows on blank
paper. Everything the template provided has to be recovered from the writing:

1. find the sheet (Otsu, largest filled region, eroded) so a desk in shot is
   not read as ink;
2. measure ink **once over the whole page** — a local-contrast window inside a
   letter-sized crop mistakes a thick pen's own interior for paper;
3. cluster letter *bodies* into rows; attach dots to the stem they sit over,
   because a dot is nearer the centre of the row above than its own;
4. fit each row's writing line through the feet of its non-descending letters;
5. cut one crop per letter, all the same height, each positioned against its
   row's line — so a bottom edge means the same thing in every crop, which is
   the one property the printed grid was providing;
6. mask every crop to its own letter's ink, since rows of handwriting sit close
   enough for a neighbour's descender to intrude.

Both paths end in `normalize_samples`, which applies the §2.3 framing.

### 5.2 Generation

The style vector is encoded **once per font** from all provided letters and
reused for all 75 characters — re-encoding per glyph would be wasteful and a
source of drift between glyphs of the same font. Content images are rendered
from Noto Sans, quantized to `uint8` exactly as the cache is, so the model sees
the value distribution it was trained on. **Letters the writer actually wrote
are passed through unchanged** rather than regenerated.

### 5.3 Raster → outlines → font

`vector/trace.py`: marching squares at the 0.5 coverage level (with the
half-pixel pixel-centre convention), corner detection, Ramer–Douglas–Peucker
simplification, then Schneider cubic fitting with windowed tangent estimation
and a straight-run shortcut. Control points are clamped to the contour's convex
hull — an unclamped fit bows straight stems outward, which is invisible per
glyph and shows up as a few points of aggregate fidelity.

`fontbuild/build.py`: outlines → CFF `.otf` via fontTools `FontBuilder` and
`T2CharStringPen`, contour orientation corrected, advance widths applied, and a
**word space synthesized** (no glyph is drawn for it, but a font without one is
unusable).

Round-tripping a real font through trace-and-rebuild scores **0.98 mean IoU**,
which bounds what the model's output can achieve through this stage.

`fontbuild/arabic.py` already emits GSUB `init`/`medi`/`fina` lookups and a
lam-alef ligature, validated in HarfBuzz — Phase 2's export path, de-risked
before any Arabic modelling.

---

## 6. How quality is measured

| Metric | Why |
|---|---|
| **IoU** | the gate's headline. Harsh on thin strokes: a 1px offset on a 4px stroke drops it from 1.0 to ~0.6 |
| **Tolerant F1** (1.5px, via distance transform) | the standard for thin structures; the fairer measure for handwriting |
| **SSIM** | perceptual structure |
| **Ink coverage ratio** | generated ink ÷ true ink; catches systematic boldness that IoU can hide |
| **Content-copy baseline** | the same glyph drawn in Noto Sans, ignoring the references — what "no style transfer" scores. Beating it is the real bar |
| **Leave-one-out** | generate each seed letter from the *other* seed letters. The only score available for a real writer, where nothing but their own letters has ground truth. Correlates with true held-out quality at **r = 0.98** across 461 fonts |

Reported per font, with the **10th percentile** alongside the mean: a model
that is excellent on 90% of hands and unusable on the rest has not passed.

---

## 7. Repository map

```
src/hfont/
  charset.py          glyph inventory; (character, contextual form) throughout
  data/
    raster.py         scanline rasterizer, non-zero winding
    frame.py          the single definition of glyph placement
    render.py         font file → normalized glyph images
    corpus.py         font discovery, metadata, family-level splits
    prepare.py        corpus → packed on-disk cache
    dataset.py        training sampler, component weight maps
  models/
    blocks.py         AdaIN, residual and up blocks, masked pooling
    generator.py      style encoder, content encoder, decoder, advance head
    discriminator.py  PatchGAN + character head
    losses.py         banded L1, soft Dice, hinge, metrics
  train.py            training loop, EMA, warm start, evaluation
  generate.py         inference: references → rasters → font
  vector/trace.py     raster → cubic Béziers
  fontbuild/          outlines → OTF; Arabic GSUB
  evaluate/           metrics, gate report, HarfBuzz shaping checks
  intake.py           template and freehand photo → normalized glyphs
  cli.py              fetch · index · prepare · train · report · template · font
scripts/              end-to-end checks, photo simulators, diagnostics
tests/                27 regression tests, each naming the bug it guards
```

Typical commands are in [README.md](README.md#pipeline).
