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
| | | | Karims | 0.291 |
| | | | Layanes | 0.453 |
| | | | Mark | 0.468 |

Mean 0.373 over all ten; 0.314 over the three development writers the intake
fixes were built on; 0.398 over the seven held-out ones (Karim's sheet
re-read after the bent-row fix: 0.275 → 0.291). On 237 held-out
corpus fonts: IoU 0.606, tol-F1 0.828 (content-copy baseline 0.655). The gate
wants IoU ≥ 0.75 (see README, "Gate status"; gate 1 should be re-specified,
see §5.5).

Style, from the same run (`hfont.evaluate.style`, printed on the last line of
`eval_writers.py`; November 2026). These measure whether the model draws
*this* writer's letter, which clDice cannot:

| measure | current model | target |
|---|---|---|
| blind attempts alike across writers (mean clDice between writers, same letter) | 0.768 | the real letters: 0.287 |
| blind attempt nearest its own writer's real letter, of 10 | 13.7% | chance is 10% |

Pass thresholds for the GPU arms (§5.3), fixed before any run: an arm passes
if similarity across writers falls to **0.67 or below** and attribution rises
to **25% or above**, while mean leave-one-out clDice stays at **0.36 or
above**. Attribution is per letter over 10 writers × 30 letters, so a gain of
under about 5 points is noise. (Elissa alone is attributed at 53%; every
other writer sits at 3-17%.)

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

- **The model barely beats the nearest typeface.** Same 80 SD19 writers,
  every letter not on their sheet (2,511 letters): the model's letter scores
  0.598 clDice against the writer's real one, and the most similar of the
  361 local fonts, resized to the writer's proportions, scores 0.576. The
  model is closer for only 54% of letters. This is the floor
  (`scripts/compose_eval.py`) that any new arm must clear clearly.

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
- Building missing letters from the writer's own strokes (November 2026;
  `hfont/compose.py`, `scripts/compose_eval.py`): `u` from `n` turned over,
  `w` from `m`, `d`/`p`/`q` from `b` mirrored, flipped or turned, `9` from
  `6`, `W` from `M`. It was scored on 80 held-out SD19 writers, given only
  their seed letters, against what they really wrote (clDice, 501 letters):
  composed 0.505, model 0.580. Only `9` ties (0.477 vs 0.468). People do not
  write these letters as turned-over copies (many write `u` as a stemless cup
  and `w` as two cups), and mirroring or flipping reverses the slant. Not
  used in fonts. (`F`/`L` from `E`, `P` from `R` and `Q` from `O` need stroke
  surgery and were not tried.)
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
different models. `eval_writers.py` therefore also prints the style numbers in
§4 (added November 2026; the two measures first planned here turned out to be
one, since the output depends only on letter and references):
- **style sensitivity**: generate each writer's letters from *another*
  writer's references and measure how much the output changes (clDice between
  the two). Near 1.0 means the writer is ignored.
- **cross-writer similarity**: mean clDice between different writers'
  generated letters for the same character. The current model sits near 0.99.

### 5.3 Model experiments, one variable each

Ranked in `reports/Few shot handwriting font techniques.md` (research,
November 2026). **All the code is written and tested on CPU** (each arm starts
as an exact identity on a warm start; `tests/test_gpu_arms.py` includes one
real training step with G1 and G2 on). Each arm is one switch. Run each for
10,000 steps, warm-started from the previous winner, and judge it on §4's
pass thresholds and the sheets:

```bash
BASE="python -m hfont.cli train --recipe recommended --cache data/cache --set steps=10000"
# G1  reference attention inside the decoder, at 16x16 and 32x32
$BASE --out runs/g1 --set init_from=runs/hw40k/best.pt --set generator.attention_levels=[0,1]
# G2  self-reconstruction: the target as its own reference (FS-Font)
$BASE --out runs/g2 --set init_from=runs/g1/best.pt --set generator.attention_levels=[0,1]       --set loss.self_recon=1.0
# G3  SD19 real writers in the mix (needs §5.4's sd19_cache.py first)
$BASE --out runs/g3 --set init_from=runs/g2/best.pt ... --set data.sd19_weight=1.0
# G4  learned per-target reference weighting (n reads h, m, u)
$BASE --out runs/g4 ... --set generator.reference_weighting=true
# G5  projection loss + adversarial well above Run 3's 0.1
$BASE --out runs/g5 ... --set loss.projection=0.1 --set loss.adversarial=0.5       --set loss.char_aux=1.0 --set adversarial_start=0
```

(`...` = the previous arm's `init_from` and switches, if it passed.) Why each,
in brief: the bottleneck attention reads references as 4×4 grids, too coarse
to place a stroke (G1). Nothing in training rewards copying from a reference,
since the target is never among them (G2). Fonts teach no idiosyncrasy (G3).
Every reference is read alike for every target (G4). L1 and Dice reward the
average letter (G5). A Latin diffusion model (G6) only if G1-G5 leave the
style numbers flat. Warm starts accept a checkpoint lacking only the new
blocks; any other mismatch is an error.

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
  files `uWWWW_TT_00000.png` and up. These PNGs carry no labels in the 2nd
  edition, **but the 1995 1st edition (`1stEdition1995.zip`, 406 MB) has a
  `.cls` label file per field**, and the k-th label belongs to the PNG with
  index k. Done and checked (November 2026): `scripts/sd19_labels.py --fetch`
  pulls only the ~10,700 label files (~1 MB) out of that zip by HTTP range
  requests into `data/sd19/by_write_labels.json`. Cross-checked against
  `by_field` (same images sorted into class folders) by pixel hash:
  **490,746 of 490,746 agree**, plus all 47 letters of one writer by eye.

What the data is like (`python scripts/sd19_labels.py`):
- 3,594 writers with labelled fields; 2,897 have at least 40 of the 52
  letters, 1,103 at least 48, only 67 all 52. Median 22 of 26 lowercase and
  23 of 26 capitals (segmentation lost the rest); digits nearly always complete.
- **Every letter is centred in its 128×128 box** (centre row 63 for `o`, `b`,
  `p` and `Q` alike), so the baseline and descenders are lost. **Sizes are
  kept** (median ink height `o` 32 px, `b` 51, `p` 46, `A` 53).

The converter (`hfont/data/sd19.py`, `scripts/sd19_cache.py`) puts the line
back: letters sit on it, `g p q y` hang from the writer's own x-height (median
height of their `a c e m n o r s u v w x z`), `j` and `Q` use typefaces'
proportions. Then it frames them with the photographed sheets' code
(`intake.normalize_samples`). Each writer becomes one cache "font" of category
`SD19`, split train/val/test 94/3/3 by writer; advances are NaN and the loss
skips them. Converting 40 writers takes 10 s on CPU (all of them ~12 min). On
the GPU machine:

```bash
# download by_write.zip (568 MB) into data/sd19/ first (URL above)
python scripts/sd19_labels.py --fetch            # labels, ~3 min
python scripts/sd19_cache.py --cache data/cache  # adds ~2,900 writers, re-packs
# then fine-tune with --set data.sd19_weight=1.0 (0 leaves them out)
```

Fine-tune on a mix of SD19 and Google Fonts; evaluate on the ten photographed
writers, never train on them. Caveats: boxed, fairly uniform 1990s printing,
binary scans; SD19's own segmentation sometimes merges two letters (a `u`
holding a `v`) or drops an `i`'s dot. Keep `data/sd19/` out of git (ignored).

### 5.5 In parallel, no GPU needed

- **Build missing letters from the writer's own strokes:** `u` from `n` turned
  over, `d` from `a` plus the stem of `b`, `F` and `L` from `E`, `P` from `R`,
  `Q` from `O`, `9` from `6`. Puts real handwriting into ~15 of the 45
  generated characters whatever the model does. Would live at font assembly
  (`generate.py`, beside `match_seed_proportions`).
- **Re-specify gate 1** on tol-F1 and clDice relative to the content-copy
  baseline, before the next result is in. IoU ≥ 0.75 is unreachable for thin
  strokes (a 1 px offset on a 4 px stroke takes IoU from 1.0 to ~0.6).
- **Gate 3 human test: built (November 2026), needs raters.**
  `python scripts/gate3_page.py --checkpoint <model>` writes
  `output/gate3/index.html` (private: real handwriting; git-ignored; share it
  directly, never publish it) and `plan.json`. A rater sees eight of a
  writer's letters and judges whether one more is by the same person. The
  probes are the model's blind attempt, the writer's own held-out letter
  (ceiling) and a stranger's (floor), always the same character: 120
  trials, 40 per condition, about ten minutes. Raters download a
  `gate3_<name>.json`, then run `python scripts/gate3_score.py answers/*.json`.
  The result to read is the anchored score (generated - floor) /
  (ceiling - floor). If the ceiling is not above the floor, the raters cannot
  tell these hands apart and the test says nothing. 5-8 raters give several
  hundred judgements per condition. Pilot it with one rater first (the
  owner); the page's letters have not yet been judged by anyone.

### Not now

- Diffusion models: a big step; try the adversarial arm first, and revisit if
  higher-resolution attention cannot move a two-stroke `k` off the average `k`.
- Phase 2 (Arabic): cursive joins already fail in Latin (README, known issue
  4). The Arabic export path itself is done and tested (gate 4).

## 6. Known open issues

- ~~Karim's `y` is misplaced~~ (fixed November 2026, `intake._letter_lines`:
  a `g p q y` that the row's straight line would cut through the top half of
  takes its line from its own top plus the row's x-height). The row line is
  still straight, so a bent row can misplace other letters by a few pixels.
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
