# Few-Shot Handwriting Font Generation

Takes ~24 handwritten letters and produces a complete, installable OpenType font
in that handwriting. Phase 1 targets Latin as a validation gate; Phase 2 targets
Arabic, where the research contribution is.

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the problem statement and staging,
and [ARCHITECTURE.md](ARCHITECTURE.md) for the model, the data and how a
photograph becomes a font. This file is results and open problems.

## Status

The full product path runs end to end: photographed template → intake → model →
installable OTF. A model is trained on the full Google Fonts corpus (3,029 fonts,
1,999 families). **Phase 1 has not passed its gate** — see below — and the gap is
model quality, not missing pipeline.

| Stage | State | Evidence |
|---|---|---|
| Glyph rasterizer | done | byte-identical output on Windows/py3.14 and Colab/py3.13 (SHA-256 match) |
| Corpus | done | 3,029 fonts / 1,999 families, family-level splits 2,568 / 224 / 237 |
| Training | done | 40K steps on a T4 over three warm-started runs (see "Training history") |
| Raster → Bézier tracer | done | 0.98 mean IoU round-tripping Times and Liberation Serif |
| Font assembly (OTF) | done | installs, shapes in HarfBuzz, synthesized word space |
| Template + photo intake | done | recovers photographed cells at tol-F1 1.000 (simulated photos) |
| Arabic GSUB export | done | gate 4 passes; see `tests/test_arabic_shaping.py` |
| Real handwriting | tested, three writers | see "Real handwriting", below; needs more writers |

## Results

Held-out **test** split, 237 fonts never seen in training. Each font's 51
non-seed glyphs are generated from its 24 seed glyphs. *Baseline* draws each
glyph in the neutral content font (Noto Sans) and ignores the references: it
is what "no style transfer" scores.

| Category | Fonts | IoU | tol-F1 | SSIM | Ink ratio | Baseline tol-F1 | LOO tol-F1 |
|---|---|---|---|---|---|---|---|
| **All** | 237 | 0.606 | **0.828** | 0.893 | 1.11 | 0.655 | 0.845 |
| Handwriting | 26 | 0.366 | 0.683 | 0.911 | 1.02 | 0.291 | 0.704 |
| Display | 40 | 0.559 | 0.782 | 0.860 | 1.14 | 0.561 | 0.783 |
| Sans-serif | 102 | 0.696 | 0.884 | 0.911 | 1.11 | 0.776 | 0.901 |
| Serif | 61 | 0.593 | 0.831 | 0.887 | 1.12 | 0.681 | 0.859 |
| Monospace | 8 | 0.566 | 0.791 | 0.829 | 1.08 | 0.567 | 0.802 |

10th percentile across fonts: IoU 0.346, tol-F1 0.668, SSIM 0.841. The model
beats the baseline on 92% of test fonts (88% of validation). Validation (224
fonts) agrees: IoU 0.581, tol-F1 0.800. These are run 3; against run 2 it is
ahead on every aggregate, by little overall (tol-F1 0.823 → 0.828) and most on
handwriting (0.663 → 0.683), and it is better on 149 of 237 test fonts
individually. Full per-font results are in
`runs/colab/runs/phase1c/report_{val,test}/` (run 2: `phase1b`).

**Unseen handwriting through the whole product path.** Three Windows fonts that
are not in Google Fonts stand in for writers. Their 24 seed letters are written
into the template, photographed (simulated: perspective, uneven light, blur,
noise, JPEG), taken through intake, and turned into a font. Every generated glyph
is then scored against the real one (`scripts/handwriting_e2e.py`):

| Writer | Model tol-F1 | (run 2) | Baseline | Ink ratio | Worst glyphs | Font |
|---|---|---|---|---|---|---|
| Segoe Print | 0.753 | 0.744 | 0.520 | 1.10 | `) L J ;` | shapes OK |
| Segoe Script | 0.631 | 0.634 | 0.489 | 1.03 | `l i u v h` | shapes OK |
| Ink Free | 0.540 | 0.562 | 0.347 | 0.99 | `7 ) W 8` | shapes OK |

Run 3 is mixed here: better on one writer, level on one, worse on one. Three
writers is too few to outweigh the 461 held-out fonts, but it is the closest
thing to the product, so it is reported as measured. Fonts and side-by-side
renders are in `output/handwriting_run3/` (run 2: `output/handwriting/`).

**Real handwriting, three writers, no printer.** Genuine samples: 30 letters
written on paper and photographed on a phone, with no template — so there is no
grid and no registration marks, and `intake.freehand_cells` recovers them from
the writing itself (rows by clustering letter bodies, the writing line by
fitting each row's feet, dots by attaching them to the stem they belong to).

| Writer | Paper | Pen | x-height on canvas | Leave-one-out | Font |
|---|---|---|---|---|---|
| 1 (`MyHandwriting.jpeg`) | blank | red gel | 25px | **0.41** | `output/myhand/` |
| 2 (`output/AntoineHand/`) | squared | black fineliner | 15px | **0.57** | `output/AntoineHand/` |
| 3 (`output/JadHand/`) | blank | pale orange | 20px | **0.34** | `output/JadHand/` |

All three fonts install and shape. Writers 1 and 2 read as handwriting; writer
3's is barely legible. None is a close match to its writer, against 0.70 for
the corpus handwriting category and 0.52–0.80 for the simulated writers. The
generated letters come back heavier and more regular than the hand that wrote
them — writer 1's idiosyncratic `r` comes back conventional.

Five things came out of the attempts, and they matter more than the scores:

* **People write with the pen they have, in the light they have.** Writer 3's
  pale orange pen is nearly invisible in the red channel and weak in green —
  0.8% of the page read as ink there, none of it strongly, and letters arrived
  as disconnected fragments. Ink is now the darkest channel per pixel, so the
  colour of the pen stops mattering, and segmentation uses hysteresis: weak ink
  continuing from strong ink is the rest of that stroke, weak ink on its own is
  paper. That took writer 3 from a broken read to a complete one (0.28 → 0.34).
  A shadow across a page is also no longer mistaken for the desk the page is
  lying on — the test is now how much darker it is, not merely that it is wide
  and dark, since blanking a shadowed corner throws away the letters in it.
* **People write on the paper they have.** Writer 2 used squared paper, where
  the printed grid is 10% of the page against 3% for the writing, and the
  letters are lost in it. Rulings are now separated from writing by *thickness*
  rather than direction — a photographed page is never square to the camera and
  its lines are never straight — by eroding to cores and regrowing them inside
  the original ink, so a stroke with any thick part returns whole while a
  ruling, having no core, cannot. Only `i` and `j` may then take a dot; without
  that rule the leftovers of printed lines get attached to letters as dots, and
  the model duly imitates them.
* **The width head under-sets these hands.** Real faces leave a median of
  0.05–0.12 em between a glyph's advance and its ink (Segoe Print, Ink Free,
  Times, Calibri). The model predicted 0.015 em for writer 2, and on 13% of
  glyphs an advance narrower than the ink itself, which sets text with letters
  colliding. The builder now tracks a whole font open when its median falls
  short, rather than flooring each glyph, so the model's relative widths — what
  it is actually good at — survive.

* **Real hands are outside the corpus's proportions, and the framing rule
  punishes them for it.** Ascenders and descenders, in x-heights:

  | | writer 1 | writer 2 | writer 3 | Segoe Print | Times |
  |---|---|---|---|---|---|
  | ascent | 2.48 | 2.80 | 1.67 | 1.61 | 1.49 |
  | descent | 0.96 | 1.53 | 1.14 | 0.52 | 0.44 |

  Framing fits the tallest-to-deepest extent onto the canvas, which leaves 19%
  of its height below the writing line. A hand whose descenders run past that
  can only be fitted by shrinking everything, so all three writers arrive with
  x-heights of 15–25px where a font's lands at 39–55 — writer 3 at 40% scale,
  and his font is the one that came out illegible. The model has never seen a
  letter that small. Scaling a sample back up afterwards does not recover the
  score (0.33–0.41 across the range for writer 1), because the letters were
  already rendered at the wrong size; the fix has to be in the framing rule
  itself, and it has to apply to the corpus too. See next steps.
* **A latent framing bug, found by that measurement.** Intake framed on every
  sample it was handed while the renderer frames on `SEED_24` alone, so moving
  to `seed30` — the default for the `font` command — added `f` and `j` to the
  frame and shrank every letter by ~40%. Fixed, and pinned by
  `test_frame_does_not_depend_on_the_seed_set`.

Intake itself is not the limiting factor: run the same freehand path over a
*simulated* sheet and it costs nothing against perfect framing (Segoe Print
0.75–0.82 versus 0.80 corpus-framed; `scripts/simulate_freehand_photo.py`).

**Leave-one-out is a valid proxy for real handwriting.** On real handwriting,
only the 24 letters the person wrote have ground truth, so the only available
score is leave-one-out: generate each seed letter from the other 23. Across all
461 held-out fonts it correlates with true held-out quality at **r = 0.98**
(Spearman 0.97), with a mean absolute gap of 0.026. A writer's own sample
therefore predicts how good their font will be before anyone looks at it.

## Gate status

| Gate | Result |
|---|---|
| 1. Held-out fidelity (IoU ≥ 0.75, SSIM ≥ 0.85) | **FAIL.** IoU 0.606; SSIM 0.893 passes |
| 2. Exported fonts are fonts | **Pass** on every font exported so far (10/10) |
| 3. Real photographed handwriting | **Run on three writers, not passed.** Leave-one-out 0.41, 0.57 and 0.34, against 0.70 for corpus handwriting. The gate asks for ≥10 writers and a human judging ≥70% of glyphs "same hand"; on these samples the honest answer looks like no |
| 4. Arabic export de-risked | **Pass.** 8/8 tests; 6/8 fail with GSUB removed |

The gate was set before training and is reported against as written. One
methodological point bears on gate 1, stated here rather than used to move it.
IoU turned out to be a poor measure for this task. On a 4px stroke a 1px offset
drops IoU from 1.0 to about 0.6, so the thin-stroked handwriting category — the
one closest to the product — scores worst by construction. Tolerance-based F1
(the standard for thin structures) is a fairer measure. If the gate is
re-specified, it should be on tol-F1 relative to the baseline, and decided
before the next run rather than after it.

## Known issues

In order of how visible they are in a generated font:

1. **Generated glyphs carry 11–14% too much ink — and the excess is
   load-bearing.** It is the most visible defect beside a writer's real letters
   (`runs/colab/runs/phase1c/probe_*.png`, columns are real / run 2 / run 3),
   and the adversarial phase did not touch it. Five explanations have been
   tested. Four are ruled out on 40 Windows fonts
   (`scripts/ink_diagnostics.py` reproduces them):
   - *Copying the content font's weight.* With Noto Sans Light and then Bold as
     content, output ink changes by 0.05%. The model ignores content weight,
     which follows from training on a random content font per sample.
   - *Regression toward average weight.* The excess is the same for the
     lightest, middle and heaviest fonts (1.12 / 1.16 / 1.11, slope 0.03).
     Stroke weight is tracked properly.
   - *Soft edges.* The ratio is 1.135 counting only pixels above 50% and 1.139
     counting all ink, and the model draws the same number of grey edge pixels
     as the real fonts. The extra ink is solid.
   - *Tracing at the wrong level.* Cutting at 0.6–0.8 instead of 0.5 removes
     ink (down to 1.075) but lowers IoU and tol-F1. The extra ink is shape
     error, not an even layer around each stroke.

   The fifth, the soft-Dice term, *is* the cause — and removing it makes the
   fonts worse. Three arms of 4,000 steps, warm-started from run 3 with the
   same seed and data, differing only in `loss.dice`
   (`runs/colab/dice_ablation/`):

   | Dice | tol-F1 | handwriting tol-F1 | handwriting ink | 10th pct tol-F1 |
   |---|---|---|---|---|
   | 2.0 (run 3) | **0.801** | **0.719** | 1.11 | **0.555** |
   | 0.5 | 0.791 | 0.697 | 0.98 | 0.481 |
   | 0.0 | 0.780 | 0.662 | 0.88 | 0.428 |

   Ink falls exactly as predicted (validation coverage 1.132 → 1.083 → 1.047)
   and every letter measure falls with it, worst at the bottom of the
   distribution. On handwriting the model goes from 11% too much ink to 12%
   too little: it stops hedging and starts dropping thin strokes.

   So the excess is not a mis-set weight. It is how the model covers its
   uncertainty about *where* a stroke goes — a slightly fat stroke still
   overlaps the truth, a missing one scores zero. The lever is reducing that
   uncertainty (better style conditioning), not re-weighting the loss. Run 3's
   setting stands.
2. **Dots on `i` and `j`, partly fixed.** The model dots 85% of `i`/`j` on
   validation fonts, but only 23 of 54 in handwriting fonts. That is where the
   dot's position and shape vary most, and no seed letter shows them. L1
   converges to the per-pixel median, and under that much uncertainty the median
   is a blank. Run 3's per-component loss weight raised these to 88% and 25 of
   54, and `i` now usually gets a clear dot, but `j` mostly still does not.
   `seed30` asks the writer for `i j` directly and passes them through.
3. **A real hand is drawn more regularly than it was written.** On the one real
   sample, generated letters come back heavier and more even than the writer's,
   and a personal `r` is replaced by a conventional one. The same shortcoming
   as issue 1 seen from the other side: averaging is the safe bet when the
   model cannot tell exactly what this hand does.
4. **Cursive joins are lost.** Segoe Script's generated lowercase has no entry or
   exit strokes, which is why `l i h u v` are its worst glyphs. This is the
   stroke-connection problem the brief places at the heart of Phase 2, already
   showing in Latin.
5. **Weak letters with unusual structure.** `f` is the worst case: the target's
   form differs a lot from the content font's, and the model partly follows the
   content skeleton.
6. **Em size convention.** Generated fonts set smaller than handwriting fonts
   at the same point size (the canvas is mapped to 0.8 em ascender). Cosmetic.

## Training history

Three runs, and the first one's failure is the most instructive part:

* **Run 1** (10K steps) plateaued at val IoU 0.497 with an ink ratio stuck at
  **1.51**. Looking at samples showed why: every output was bold. The loss
  weighted errors on target-ink pixels 4× and nothing else, so missing ink cost
  four times what extra ink did, and the model learned that painting too much
  was the safe bet. Fixed with a symmetric band around every stroke edge.
* In parallel, testing intake against simulated photos showed that photographed
  handwriting arrived **17% larger** than the model had seen the same font in
  training. The corpus used the font's designed baseline and the extent of all
  75 glyphs, and intake — which has neither — estimated both from the 24 seed
  letters. Both paths now go through one definition (`data/frame.py`), and
  framing agreement went from tol-F1 0.59–0.78 to 1.000.
* **Run 2** (16K steps, warm-started from run 1's weights on the re-rendered
  corpus) reached val IoU 0.565 with an ink ratio of 1.15.
* **Run 3** (14K steps from run 2's weights) added two things. First, a loss
  weight on small connected components: any part smaller than half the glyph's
  largest part is boosted, up to 8×, which puts an `i` dot at 2.5–4×. Second,
  from step 8,400, a hinge adversarial loss (weight 0.1) plus a
  character-classification term (0.1), both from a spectral-norm patch
  discriminator. Val IoU went from 0.565 to 0.571 and the ink ratio from 1.149
  to 1.133. A snapshot taken before the adversarial phase shows that the
  component weight did the work. It raised the validation dot rate from 85% to
  88% by step 8,400, with no gain afterwards. The ink ratio dipped to 1.122 at
  step 10K and then returned. As configured, with a fresh discriminator and
  5,600 steps, the adversarial phase was close to a no-op.
* **Dice ablation** (three arms of 4K steps from run 3's weights, same seed and
  data, `loss.dice` 2.0 / 0.5 / 0.0). Confirmed that Dice causes the excess ink
  and that the excess is worth paying for — see known issue 1. The control arm
  also shows what 4K more steps at lr 1e-4 buys on its own: val IoU 0.581 →
  0.585, tol-F1 0.800 → 0.801. Diminishing returns on this recipe.

## Style conditioning: four arms, one variable each

The pooled style vector was the suspected weakness, and it was measured rather
than assumed. Three writers' style codes sat at cosine 0.98+ and their generated
letters were near-identical (`output/review/STYLE_COLLAPSE.md`). Four arms were
then trained from identical init on the same corpus for 10,000 steps, each
changing one thing:

| arm | val IoU | writer LOO (clDice) | style cos w1~w3 |
|---|---|---|---|
| control | 0.4917 | 0.241 | 0.994 |
| + scale jitter | 0.4503 | 0.199 | 0.997 |
| + reference attention | **0.5784** | 0.221 | 0.969 |
| **+ attention & contrastive** | 0.5709 | **0.247** | **0.926** |

**Reference attention** (content features as queries, reference features as keys
and values, after FS-Font) is the largest quality gain the project has produced.
On its own it cost real-writer leave-one-out; adding a **style-contrastive loss**
on the style codes recovers that and goes past the control. The combination is
also the only configuration that has ever pulled two writers' style codes apart.

Two things this did *not* fix, stated plainly. Writer 3 is still identified
*below* chance (0.33 against writer 1), and he is the writer with the thinnest
strokes at 1.07px — so the residual failure tracks intake quality and scale,
which is next step 2, not the style path. And every number here is from a
10,000-step run; the ordering is unconfirmed at full length.

Recipe: `hfont.train.recommended_config(cache, out)`. It is deliberately *not*
the dataclass default — flipping `style_attention` would make every checkpoint
saved before it existed fail to load.

## Intake: seven fixes, no retraining

Setting real text in each writer's font, and looking at every glyph, found
defects between the photograph and the canvas. Each is measured on the shipped
checkpoint with nothing else changed, in the order they were made.

| fix | what it corrects | mean writer LOO (clDice) |
|---|---|---|
| — | shipped | 0.239 |
| percentile framing | one sprawling letter shrank the whole alphabet, so fonts set at about half size | 0.257 |
| pen-darkness normalisation | a light pen never reached full ink; export discarded 44% of writer 3's strokes | 0.275 |
| descender attachment | a `j`'s hook curls away from its stem and was dropped as a stray mark | 0.275 |
| baseline snapping (freehand only) | one sample's wobble became a permanent offset on every copy of that glyph | 0.307 |
| solid-foot snapping | a faint wisp, not the stroke, was being rested on the line | 0.304 |
| ruling-stub pruning | squared paper left a tick at the foot of every letter that sat on a line, and a `+` on `g` and `y` | 0.303 |
| page-wide ink, used as measured | each crop's ink was measured a second time inside its cell, shattering faint strokes | **0.313** |

**+0.074, about 31% relative.** Some steps dip slightly on the score while
fixing something visible — ruling-stub pruning removed a grid line running
through writer 2's `i`, and the score dropped because the model's `i`, drawn
from 29 now-cleaner references, changed. That drop is not significant (paired
p = 0.32) and is almost all one letter.

Properties that made these safe to ship without a retrain, each pinned by a
test:

* **Framing is a no-op on typefaces.** For a designed font the 90th percentile
  of ascent *is* the maximum, so twelve Windows fonts render byte-identically.
* **Pen normalisation and its outlier stage are no-ops on a dark pen.** Writers
  1 and 2 came through byte-identical.
* **Snapping is off for renders.** A designed font's placement is intentional —
  round letters overshoot the line — and snapping it broke the intake/renderer
  identity until it was scoped to freehand.
* **Stub pruning only runs on ruled paper**, and reclaims only what is thin
  across a whole neighbourhood, so it cannot eat a stroke's edge or an `i` dot.

Two more fixes happen when the font is assembled, after the model has read
the writer's letters — so neither can move a leave-one-out score:

* **One stroke weight per font.** Generated letters came out up to 50% heavier
  than the writer's own, visible in any word. Thinning them was rejected three
  ways (their extra weight hedges against uncertain stroke position, and
  thinning breaks them: clDice 0.307 → 0.275). Thickening the writer's letters
  to meet them instead keeps every skeleton intact (clDice ~0.99) and closes the
  gap to within 1% on all three writers. The cost: a writer with a light pen
  gets a font bolder than his handwriting. `match_weight=False` restores it.
* **Broken letters are drawn by the model.** A written letter with under 30%
  of the ink of the model's version of it has lost strokes, not just weight —
  healthy letters sit at 0.4–0.7. Only writer 3's `j` (0.14) and `y` (0.20)
  qualified, and "`.umps .am laz: fl:`" now sets as "`jumps jam lazy fly`".

A measurement caveat that affects every older number in this file: **tol-F1
rewards small glyphs.** Its tolerance is a fixed 1.5px, so identical glyphs
scored at half size gain +0.092 while clDice does not move. Photographed hands
used to arrive at half scale, so their tol-F1 scores — and the CLI's
good/usable/rough verdicts, which were calibrated at corpus scale — were
optimistic. The CLI now prints clDice alongside.

## Next steps

1. **More writers — now the most important item on this list.** Every intake
   fix below was found on, and measured against, the same three sheets. Each
   has a structural reason to generalise (two are exact no-ops on the cases
   they should not touch), but three writers cannot show that it does. Seven
   more through `--freehand` (no printer needed) would, and would also settle
   whether the remaining gap is these hands or the method.
2. ~~**Reframe on x-height, and re-render the corpus with it.**~~ **Done, and
   it needed neither a re-render nor a retrain** — see "Intake" below. Framing
   on the 90th percentile of extent rather than the maximum turns out to be a
   no-op on typefaces, so the corpus is unchanged while real hands double in
   size.
3. ~~**Attack positional uncertainty, not the loss weights.**~~ **Done — see
   "Style conditioning" below.** Reference-to-target attention was built and
   trained, and it works: +0.087 val IoU, the largest single gain this project
   has recorded. Combined with a style-contrastive loss it also lifts real-writer
   leave-one-out above the control for the first time. `hfont.train.recommended_config`
   carries the recipe. What remains is confirming it at 40K steps.
4. Decide whether gate 1 is re-specified on tol-F1 against the baseline.
5. Only then Phase 2. The export side is ready (gate 4); the open problem is
   joining, which issue 3 above shows the current model does not handle.

## The validation gate

The brief asks for this to be concrete *before* Phase 1 runs rather than
rationalized afterwards. Phase 2 starts only if all four hold:

1. **Held-out fidelity.** Mean IoU ≥ 0.75 and SSIM ≥ 0.85 against held-out
   fonts, generating all non-seed glyphs from 24 seed glyphs. Measured per font,
   with the 10th percentile reported — a model that is excellent on 90% of fonts
   and unusable on the rest has not passed.
2. **The font is a font.** 100% of exported fonts pass `hfont.evaluate.shaping`:
   installs, every charset codepoint maps to a real glyph, no zero advances,
   text shapes correctly in HarfBuzz.
3. **Real handwriting.** On ≥ 10 genuine photographed samples, a human judges
   ≥ 70% of generated glyphs "plausibly by the same hand" in a two-alternative
   test against held-out real glyphs. **This is the condition most likely to
   fail quietly** — held-out typeset fonts are a far easier distribution than
   ink on paper, and metric 1 can look excellent while this fails outright.
4. **Arabic export is de-risked.** Before Phase 2 modelling begins, a
   hand-built Arabic font with correct GSUB `init`/`medi`/`fina` substitution
   and a lam-alef ligature must shape correctly in HarfBuzz. This validates the
   export path independently of any model, because the brief is right that its
   complexity is easy to underestimate and it is not exercised at all by Latin.

Budget note: the brief's chief risk is Phase 1 overrunning. Gate 4 is
deliberately not gated on 1–3 — it is cheap, independent, and should be done in
parallel rather than after.

## Install

```bash
pip install torch numpy pillow fonttools scikit-image uharfbuzz
```

## Pipeline

```bash
# 1. Corpus. Skip the fetch if you already have fonts somewhere.
python -m hfont.cli fetch --dest data/google-fonts
python -m hfont.cli index data/google-fonts --out data/manifest.json

# 2. Render to a glyph cache (this is what training reads).
python -m hfont.cli prepare --manifest data/manifest.json --out data/cache

# 3. Train. On Colab, use notebooks/colab_train.ipynb instead.
python -m hfont.cli train --cache data/cache --out runs/phase1

# 4. Score a checkpoint on held-out fonts, beside the content-copy baseline.
python -m hfont.cli report --checkpoint runs/phase1/best.pt --cache data/cache --split test
```

### Making a font from your handwriting

```bash
python -m hfont.cli template --out output/template.png   # seed30 by default; print in colour
# ...fill it in with a dark pen, photograph it flat with all four squares in view...
python -m hfont.cli font photo.jpg --checkpoint runs/phase1c/hfont_step030000.pt \
    --family "My Hand" --out output/my_hand.otf
```

No printer? Write the seed letters straight onto blank paper, in rows, in seed
order, with a gap between letters, and photograph the sheet:

```bash
python -m hfont.cli font photo.jpg --checkpoint runs/phase1c/hfont_step030000.pt \
    --freehand --rows 10,10,10 --debug output/letters_found.png \
    --family "My Hand" --out output/my_hand.otf
```

`--debug` writes the photo back with every letter boxed and named, which is the
quickest way to see a miscount — the command refuses to guess if the rows do
not hold the number of characters it was told to expect.

The command prints a leave-one-out quality score for your sample, which
predicts the quality of the generated letters (r = 0.98 on held-out fonts).
`seed30` asks for six more letters than `seed24` — including `i`, `j` and `f`,
which the current model generates worst — and is the default until known issue 2 is
fixed. Gate numbers are reported on `seed24`, the harder setting. The content font is Noto Sans, downloaded to
`data/fonts/` (from `ofl/notosans` in google/fonts).

### Checking the pipeline without a model

```bash
python -m hfont.cli roundtrip C:/Windows/Fonts/times.ttf
```

Renders a real font, traces it, rebuilds it as an OTF and scores the result.
This is the fastest way to tell whether a quality problem is in the tracer and
font builder or in the model — if this does not score well, nothing the
generator produces can either. Currently ~0.98 mean IoU.

## Training on Colab

`notebooks/colab_train.ipynb` is built for a free GPU runtime being reclaimed at
any moment: the project directory and checkpoints live on Drive, and training
resumes from the last checkpoint automatically. The font corpus is deliberately
*not* on Drive — it is re-cloned locally each session, because Drive I/O would
throttle every step.

Regenerate the notebook with `python scripts/make_colab_notebook.py` (it is
generated rather than hand-edited so it stays diffable).

When training runs as a background process instead, Colab does not count it as
activity. An idle session is reclaimed after roughly 90 minutes, and the
corpus, cache and checkpoints go with it. Keep executing cells while a job
runs, and download results as soon as it finishes.

## Layout

```
src/hfont/
  charset.py          glyph inventory; (character, contextual form) from day one
  data/
    raster.py         scanline rasterizer, non-zero winding
    frame.py          the one definition of glyph placement, shared with intake
    render.py         font file -> normalized glyph images
    corpus.py         font discovery, metadata, family-level splits
    prepare.py        corpus -> on-disk glyph cache
    dataset.py        training sampler
  models/             generator, discriminator, losses
  vector/trace.py     raster -> cubic Béziers
  fontbuild/build.py  outlines -> installable OTF
  fontbuild/arabic.py Arabic GSUB: positional forms + lam-alef (Phase 2)
  evaluate/           metrics, gate report, HarfBuzz shaping validation
  intake.py           template layout; photo -> registered cells -> glyphs,
                      and the freehand path for writing with no template
scripts/
  handwriting_e2e.py        full product path on an unseen handwriting font
  simulate_template_photo.py  intake stress test with simulated phone photos
  simulate_freehand_photo.py  the same, for letters written on blank paper
  ink_diagnostics.py        the four excess-ink checks behind known issue 1
  train.py            training loop
  generate.py         inference: samples -> font
```

## Design decisions worth knowing

These are the choices that are load-bearing, and the reasons are in the module
docstrings.

**The rasterizer is written here rather than bound to FreeType.** The corpus is
hundreds of thousands of rendered glyphs and the training machine is not the
machine that prepares them; a FreeType version difference between the two would
silently change hinting and stem darkening on every sample. This implementation
produces identical bytes anywhere.

**The baseline is pinned to a fixed canvas row.** Placing the ink box and
letting the baseline fall where it may renders just as well and is unusable for
export — a generated glyph has no source font to ask where its baseline was, and
a font with a baseline off by a few percent has every letter floating off the
line.

**Normalization is per font, never per glyph, and defined only by what a
photograph also has.** Relative proportions — x-height against cap height,
ascender length — *are* the style, so one frame is shared by all glyphs of a
font. That frame comes from the ink of the 24 seed letters, with the baseline
estimated from the bottoms of `a c e m n o r s u v w x z`, never from the font's
designed baseline or its full character set. Handwriting has neither of those,
and when the renderer used them, photographed handwriting reached the model 17%
larger than anything it was trained on. `data/frame.py` is the single definition
both paths use.

**Splits are by family.** Roboto ships 18 weights; one in train and one in
validation would report a meaningless number.

**The target glyph is never among the style references,** and the exclusion is
on rendered pixels, not character names — fonts routinely point several
characters at one outline. Without this the task degenerates into copying.

**The model predicts advance widths.** A raster alone cannot be assembled into a
font; without per-glyph advances the export sets every letter on the same pitch
and looks obviously wrong however good the letterforms are.

**Adversarial loss starts late and is off by default.** Get reconstruction
right, then bring in the discriminator to harden stroke edges — which matters
specifically because the raster is subsequently traced, and a soft edge becomes
a wobbly outline in the installed font. In run 3, at weight 0.1 for the last 40%
of training, it changed almost nothing measurable (see training history). That
still needs explaining before it is tuned up.

## Tests

```bash
python -m pytest tests -q
```

Each test names the failure it guards against. Most were real bugs found during
development that were invisible glyph-by-glyph and showed up only as a few
points of aggregate fidelity — a half-pixel coordinate convention mismatch, a
curve fitter bowing straight edges out of the glyph, a missing space.

Tests needing a glyph cache skip cleanly when there isn't one.

## Phase 2 notes

The code carries `(character, contextual form)` throughout, so Arabic adds
entries to a table rather than changing signatures. What Phase 2 actually needs:

- the Arabic glyph inventory (~100–120 forms plus lam-alef ligatures);
- GSUB `init`/`medi`/`fina` and ligature lookups in `fontbuild` — see gate 4,
  worth doing early and independently;
- a joining constraint in the model, which the brief expects will need explicit
  modelling rather than emerging on its own;
- a seam-discontinuity metric in `evaluate`, measuring joins in rendered words
  rather than per-glyph quality. `evaluate.shaping.render_string` already
  rasterizes shaped text through real advances, which is the hook for it.
