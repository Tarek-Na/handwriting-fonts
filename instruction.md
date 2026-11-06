# Instructions: picking this project up on a new machine

Written 29 September 2026 for whoever continues this work with no access to
the earlier conversations: the project owner on a new computer, or a new Claude
session. It records what cannot be recovered from the code alone: the rules
the project runs by, what has been measured and ruled out, where things stand,
and what to do next and why.

Read in this order: this file, then [README.md](README.md) (results and open
problems), [ARCHITECTURE.md](ARCHITECTURE.md) (the model, data and pipeline in
depth), and [SETUP_GPU.md](SETUP_GPU.md) (the exact setup commands).

---

## 1. The project in one paragraph

A person photographs about 30 letters they wrote by hand, and the system
produces an installable OpenType font of all 75 Latin characters in their
handwriting. Their own 30 letters go into the font as written; a neural network
draws the other 45 in their style. Latin is Phase 1 and a validation gate;
Arabic is a stated possibility for Phase 2 ([PROJECT_BRIEF.md](PROJECT_BRIEF.md)).
The whole chain works end to end: photo → letters → model → traced outlines →
`.otf` that installs and sets text. **The weak link is the model: every letter
it generates looks typeset, in the writer's stroke weight but not their
shapes.** Everything below is about fixing that.

## 2. Setting up the new machine

Follow [SETUP_GPU.md](SETUP_GPU.md) step by step. In short:

1. `git clone https://github.com/Tarek-Na/handwriting-fonts.git`
2. Unzip **`hfont_private_files.zip`** at the project root. It is not on GitHub
   and must never be: it holds the handwriting photos of real people. The
   owner copies it over by hand (USB or a private link). Contents, 60 MB,
   sha256 starting `a4f2911809695af4`:
   - `MyHandwriting.jpeg` (the owner's), `AlansHand.jpeg`, `BahaaHand.jpeg`,
     `EliasHand.jpeg`, `ElissasHand.jpeg`, `KarimsHand.jpeg`,
     `LayanesHand.jpeg`, `MarkHand.jpeg`
   - `output/AntoineHand/AntoineHandwriting.jpg`, `output/JadHand/JadHandWriting.jpeg`
   - `runs/phase1c/hfont_step030000.pt`: the current ("shipped") model, 63 MB
   - `data/fonts/NotoSans[wdth,wght].ttf`: the content font the model reads
3. PyTorch for the machine's CUDA version, then `pip install -e ".[dev]"`, then
   `python -m pytest tests -q`. 75 tests passed on the old machine; any that
   need a glyph cache skip cleanly until step 4 has built one.
4. Rebuild the training data (`hfont fetch`, `index`, `prepare`; ~30 minutes).
   Rendering is byte-identical across machines, so this is the same data.
5. Optional: `data/sd19/by_write.zip` (568 MB, complete) was downloaded on the
   old machine and can be copied instead of downloaded again.

## 3. Rules this project runs by

These were set by the owner or learned the hard way. Keep them.

- **Handwriting stays private.** The repository is public. Photos of real
  handwriting, and every image cut or rendered from them (crops, overlays,
  letter sheets, model outputs), are ignored by `.gitignore` and must never be
  committed. Text and numeric reports about them are fine. Before committing,
  `git status` must show no image of anyone's writing. Three photos (the
  owner's, Antoine's, Jad's) remain in *older* commits on GitHub; the owner has
  not yet decided between making the repository private or rewriting history
  (a force-push). Do neither without the owner's explicit say-so.
- **Measure before and after, one change at a time.** Each fix is its own
  commit, and the commit message states what it fixes, why, and the numbers
  before and after. A change to intake is kept only if it does not lower mean
  leave-one-out on the writers by more than 0.01, unless it fixes a defect you
  can see in the font; then say so in the commit.
- **Every fix gets a test that fails without it.** Check this by stashing the
  source change and running the new test. Each test's docstring names the
  failure it guards against.
- **Intake changes must not disturb other sheets.** After any change to
  `src/hfont/intake.py`, confirm every sheet still reads (10, 10, 10) and that
  sheets the change is not aimed at give byte-identical crops.
- **Held-out writers are for evaluation only.** Never train or tune on the ten
  photographed writers. They are the only honest test of the product.
- **Decide a gate or metric before the run it judges,** not after seeing
  results.
- **Look at the output, not just the score.** The biggest defects in this
  project (clipped descenders, grid ticks, typeset letters) were found by
  looking at fonts and letter sheets, after the scores had missed them.
- Style of the code and docs: comments and docstrings explain *why*, with the
  measurement that motivated the choice. Match it.

## 4. Where things stand

### Numbers to beat

The current model is `runs/phase1c/hfont_step030000.pt` (no reference
attention; ~31M parameters). On the ten photographed writers, leave-one-out
clDice (hide each written letter, draw it from the other 29, compare shapes;
higher is better) from `python scripts/eval_writers.py --tag _phase1c`:

| writer | clDice | | writer | clDice |
|---|---|---|---|---|
| My (owner, dev) | 0.283 | | Alans | 0.323 |
| Antoine (dev) | 0.381 | | Bahaa | 0.300 |
| Jad (dev) | 0.279 | | Elias | 0.398 |
| | | | Elissas | 0.556 |
| | | | Karims | 0.275 |
| | | | Layanes | 0.453 |
| | | | Mark | 0.468 |

Mean 0.372 over all ten; 0.314 over the three development writers the intake
fixes were built on; 0.396 over the seven held-out ones. On 237 held-out
corpus fonts: IoU 0.606, tol-F1 0.828 (content-copy baseline 0.655). The gate
wants IoU ≥ 0.75 (see README, "Gate status"; gate 1 should be re-specified,
see §5.5).

### What was found, so it is not rediscovered

- **The model ignores the content font.** Feeding Times or Comic Sans instead
  of Noto Sans changes the output by under 0.5%; zeroing all four U-Net skip
  connections changes it by under 1%. Feeding the wrong letter (an `o` for
  everything) makes it draw that letter. So the content path supplies only
  *which* letter; the shape drawn is the model's own average letter for that
  character, learned from ~3,000 mostly typeset fonts, with the writer's weight
  and slant applied. Training already draws the content from a random font
  each time (`content_source="random"`), which is why. **Consequence: skip
  dropout or content deformation would change nothing.**
- **Style enters the decoder only through AdaIN** (a per-channel scale and
  shift from the pooled 256-number style code). AdaIN can change how strokes
  look, not where they are. The only path that can move strokes is reference
  cross-attention, which exists (`models/blocks.py`, `ReferenceAttention`) but
  sits at the 8×8 bottleneck and reads the writer's letters as 4×4 grids.
- **The losses reward the average letter.** L1 + Dice under uncertainty favour
  a hedged, average shape. A Dice ablation showed the excess ink is that hedge
  and removing Dice makes letters worse (README, known issue 1). This is why a
  mode-seeking objective (adversarial, later diffusion) is on the table.
- **The recommended recipe** (`hfont.train.recommended_config`: reference
  attention + style-contrastive loss) was best in a 10,000-step comparison:
  val IoU 0.571 against 0.492 for the control, writer leave-one-out 0.247
  against 0.241, and the only arm that pulled two writers' style codes apart
  (cosine 0.994 → 0.926). It has never been trained to full length.
- A 40,000-step run of it (handwriting fonts sampled 6×) was started on Colab
  at 77 images/s on a T4 and reached step ~3,500 (val IoU 0.478 at step 2,000)
  before the connection dropped and the runtime was reclaimed. Nothing was
  kept from it.

### Tried and rejected (do not redo without new evidence)

- Scale jitter augmentation: hurt (val IoU 0.492 → 0.450).
- Irregularity (wobble) augmentation: failed its pre-flight check.
- Thinning generated letters to the writer's weight: breaks strokes (clDice
  0.307 → 0.275). The writer's letters are thickened instead.
- Removing or lowering the Dice loss: makes letters worse (see above).
- Per-writer fine-tuning (test-time adaptation, November 2026, on the shipped
  model, Mark's sheet, 6 of his letters held out and the model tuned on the
  other 24): the tuned letters are memorised (clDice up to 0.87–0.92) while
  the held-out ones get worse every way it was tried. Full model, 150 steps:
  0.485 → 0.382 over two folds. Full model, 30 steps at half the learning
  rate: 0.578 → 0.421. Style projections only (2M of 31M parameters), 100
  steps: 0.578 → 0.503. A zero-step run reproduces the untuned score exactly,
  so this is not a bug. 24 letters are too few to adapt this model without
  overwriting what it knows about other letters.
- Weight-space programming (an LLM reading and editing weights directly,
  MIT CSAIL, November 2026): not attempted. It has worked on networks of
  thousands to ~1.5M weights and on single localised faults, needs a frontier
  model with a code sandbox, and by its authors' account could not add an
  ability a model lacked, which is this project's problem.
- For squared paper: separating grid from pen by line spacing, by straight-line
  detection (paper curves; slow), by clipped darkness, or by per-pen relative
  thresholds. The working method is thickness plus pen-likeness (darker or
  coloured than the grid), in `intake._pen_ink`.

### Intake (photo → letters): state of the art in this repo

All ten sheets read (10, 10, 10). The pipeline handles plain, lined and squared
paper, coloured and black pens, faint grids, tilted photos, letters cut off by
the frame, detached strokes, and long descenders (fitted, not clipped). At
font assembly, generated letters are resized zone by zone to the writer's
x-height, ascender and descender, and the writer's letters are thickened to
the model's stroke weight. Details in README, "Intake" and "Held-out test".

## 5. What to do next, in order

### 5.1 Train the recommended recipe to full length

```bash
python -m hfont.cli train --recipe recommended --cache data/cache --out runs/hw40k \
    --set steps=40000 --set data.handwriting_weight=6.0
```

Every later decision depends on whether attention + contrastive holds up at
full length. It resumes by itself if interrupted (re-run the same command).

### 5.2 Compare it with the current model, on writers and on fonts

```bash
python scripts/eval_writers.py --checkpoint runs/hw40k/best.pt --tag _hw40k
python -m hfont.cli report --checkpoint runs/hw40k/best.pt --cache data/cache --split test
```

Compare against §4's table, **and look at the sheets** in `output/writers/`
(what they wrote / the model's blind attempt / letters never written). The
question: does the blind attempt now follow the writer's own shapes (Bahaa's
two-stroke `k`, Mark's `i` with a top bar, Layane's `r`), or is it still a
typeset letter?

Leave-one-out clDice alone cannot answer that: it barely moves between very
different models. Two measures should be added to `eval_writers.py` before
the arms in 5.3 (both cheap, CPU is enough):
- **style sensitivity**: generate each writer's letters from *another*
  writer's references and measure how much the output changes (clDice between
  the two). Near 1.0 means the writer is ignored.
- **cross-writer similarity**: mean clDice between different writers'
  generated letters for the same character. The current model sits near 0.99.

### 5.3 Model experiments, one variable each

Each at 10,000 steps, warm-started from the 5.1 checkpoint
(`--set init_from=runs/hw40k/best.pt`), judged on 5.2's measures and sheets:

1. **Reference attention at higher resolution.** Add attention after the
   decoder's first one or two up-blocks (16×16, 32×32), reading the style
   encoder's matching intermediate feature maps (its backbone downsamples
   128 → 64 → 32 → 16 → 8 → 4; today only the 4×4 output is attended to).
   `ReferenceAttention` starts as an exact identity (zero-initialised output),
   so a warm start is safe. This is the most direct attack on "the style path
   cannot move strokes".
2. **Adversarial loss, properly weighted.** The patch discriminator with
   character and font-identity heads already exists (`loss.adversarial`,
   `loss.char_aux`, `loss.font_aux`, `adversarial_start`). Run 3 used 0.1 for
   the last 40% of training and it changed almost nothing; this arm tests
   whether a mode-seeking objective moves shapes away from the average.
3. **Fine-tune on real handwriting** (5.4).

### 5.4 Real handwriting data: NIST Special Database 19

Fonts teach letterforms; real hands teach wobble and idiosyncrasy, which no
font has. SD19 (verified against NIST's own user guide):

- 3,669 writers, 814,255 handprinted characters, each a 128×128 binary PNG,
  classes 0-9, A-Z, a-z. Writers: US Census Bureau staff (1990s) and one
  Bethesda high-school class.
- Each writer filled one form with digit fields, one uppercase alphabet field,
  one lowercase alphabet field and the Constitution preamble; the alphabet
  order is randomised per form template (100 templates).
- Downloads (2nd edition, public, no login):
  `https://s3.amazonaws.com/nist-srd/SD19/by_write.zip` (568 MB, per writer)
  and `.../by_field.zip` (541 MB, per field and class). Also `by_class.zip`
  (1 GB), `hsf_page.zip` (348 MB, full form scans), and the guides
  `sd19_users_guide_edition_2.pdf`, `1stEditionUserGuide.pdf`.
- `by_write/hsf_N/fWWWW_TT/` holds one folder per field: `dWWWW_TT` digits,
  `uWWWW_TT` uppercase, `lWWWW_TT` lowercase, `cWWWW_TT` Constitution, with
  files `uWWWW_TT_00000.png` and up. **These images carry no letter labels**
  (the label files and the per-template answer keys are not in the public
  edition). `by_field` holds the same images sorted into class folders (hex
  ASCII names, e.g. `41` = A), writer order preserved but writer IDs dropped.
  **Recover labels by matching identical images** (hash the decoded pixels)
  between the two; drop any image whose hash maps to more than one class.

Converter plan (not written yet): per writer, recover labels; check coverage
(how many writers have a complete a-z and A-Z); **inspect how letters sit in
the 128×128 image first**: they were written in a row inside a box, so the
shared baseline and the x-height/capital ratio may be lost in the crops, and
the pipeline's framing depends on both (`data/frame.py`). Then write each
writer as a "font" into the cache format (`data/prepare.py`: `meta.json` plus
`fonts/<id>.npz` holding `images` uint8 (N, S, S), `keys`, `advances`;
advances are unknown for SD19, so give them no weight in the loss). Fine-tune
on a mix of SD19 and Google Fonts; evaluate on the ten photographed writers,
never train on them. Caveats: boxed, fairly uniform 1990s printing, binary
scans. Keep `data/sd19/` out of git (already ignored).

### 5.5 In parallel, no GPU needed

- **Build missing letters from the writer's own strokes:** `u` from `n` turned
  over, `d` from `a` plus the stem of `b`, `F` and `L` from `E`, `P` from `R`,
  `Q` from `O`, `9` from `6`. Puts real handwriting into ~15 of the 45
  generated characters whatever the model does. Would live at font assembly
  (`generate.py`, beside `match_seed_proportions`).
- **Re-specify gate 1** on tol-F1 and clDice relative to the content-copy
  baseline, before the next result is in. IoU ≥ 0.75 is unreachable for thin
  strokes (a 1 px offset on a 4 px stroke takes IoU from 1.0 to ~0.6).
- **Build the gate 3 human test:** a two-alternative page ("which of these is
  the real letter?") shown to people who know the writers; ≥70% "same hand"
  passes. Ten writers are now in hand, the gate's count.

### Not now

- Diffusion models: a big step; try the adversarial arm first, and revisit if
  higher-resolution attention cannot move a two-stroke `k` off the average `k`.
- Phase 2 (Arabic): cursive joins already fail in Latin (README, known issue
  4). The Arabic export path itself is done and tested (gate 4).

## 6. Known open issues

- **Karim's `y` is misplaced.** His second row slopes; the straight baseline
  fitted through the row's feet (`intake._row_baseline`) runs a letter-height
  high at the right end, so his `y` there is placed as if hanging wholly below
  the line and gets squashed by descender fitting. Likely fix: a more robust
  baseline fit (outlier-resistant, or allowing gentle curvature).
- On squared paper photographed as JPEG, grid lines inside small loops (Mark's
  `o`) survive: colour bleeds into them, so they pass as pen.
- A heavily blurred grid is not recognised as ruled paper at all.
- Writer 3 (Jad) wrote `j` and `y` too faintly; the model's versions are used.
- The CLI's good/usable/rough verdict is calibrated on tol-F1, which rewards
  small glyphs; read clDice beside it.

## 7. Map of the repository

```
src/hfont/
  intake.py        photo -> 30 letter images (the largest module; all the
                   paper, pen, grid and layout handling lives here)
  models/          generator.py (encoders, decoder, advance head), blocks.py
                   (AdaIN, attention, residual/up blocks), discriminator.py,
                   losses.py
  train.py         training loop; recommended_config() is the best recipe
  generate.py      inference and font assembly (weight, size, broken letters)
  data/            corpus.py, render.py, raster.py, frame.py, prepare.py,
                   dataset.py, augment.py
  vector/trace.py  raster -> Bezier outlines
  fontbuild/       build.py (OTF), arabic.py (Phase 2 shaping)
  evaluate/        metrics.py (IoU, tol-F1, clDice), report.py, shaping.py
  cli.py           hfont fetch | index | prepare | train | template | font | report
scripts/
  eval_writers.py  a checkpoint on every photographed writer (scores + sheets)
  how_good.py      leave-one-out per letter (attention-aware)
  ... one-off analyses behind README findings (style_collapse.py,
      style_identification.py, ink_diagnostics.py, simulate_*_photo.py, ...)
output/review/     REVIEW.md, STYLE_COLLAPSE.md, DECISIONS.md: past analyses
tests/test_pipeline.py   75 tests; each docstring names the bug it guards
```

Private, on disk only: the photos (project root and `output/*Hand/`),
`output/holdout/`, `output/writers/`, `output/review/*.png`, `data/sd19/`.

## 8. Everyday commands

```bash
# make a font from a photographed sheet (letters in rows, seed30 order)
python -m hfont.cli font AlansHand.jpeg --checkpoint runs/phase1c/hfont_step030000.pt \
    --freehand --rows 10,10,10 --debug output/holdout/Alan_found.png \
    --family "Alan Hand" --out output/holdout/AlanHand.otf

# the letters a writer should write, in order (seed30)
#   AEGHMORSab  egkmnorsty  036,BDNfij

python -m pytest tests -q
python scripts/eval_writers.py --checkpoint <model.pt> --tag _<name>
```

## 9. If training goes back to Colab

It worked, with quirks: the Colab MCP connection drops and must be reconnected
by the owner; a background training process does not count as activity, so an
idle runtime is reclaimed (poll it every ≤15 minutes); checkpoints are 522 MB,
so export a slim one (generator weights with the averaged weights applied, in
half precision, ~64 MB) to download; Google Drive would not mount through the
connection. The owner has a second Google account for more GPU quota
(`?authuser=2` in the Colab URL, before the `#` fragment).

## 10. Working with the owner

The owner wants short progress updates, plain explanations, and to be asked
before anything irreversible or outward-facing (pushing, deleting, changing
repository visibility). They collect handwriting sheets from friends; each new
sheet is a genuine held-out test, and the practice has been to run it through
intake unchanged first and fix what breaks.
