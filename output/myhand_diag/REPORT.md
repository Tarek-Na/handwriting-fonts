# Where writer 1's score goes, and what would raise it

Diagnosis of `MyHandwriting.jpeg` (freehand, blank paper, red gel pen) against
`runs/phase1c/hfont_step030000.pt`, seed30. Nothing was retrained and the corpus
was not re-rendered. All numbers below are measured, including the ones that
contradict README.

**Headline:** the 0.411 is not lost in the photograph or in intake. It is where
this model puts a thin, slanted hand even when handed a clean font version of
one — Bradley Hand, rendered directly from its outlines with no camera
involved, scores 0.422. Scale costs a measurable but modest ~0.06, and is not
recoverable at inference. The remaining ~0.20 is the model regularizing toward
an upright, heavier, simpler hand.

Reproduced commands are at the end.

---

## Step 1 — Baseline reproduces

`0.411` leave-one-out tol-F1, matching the 0.41 in README. Written to
`output/myhand_diag/`; `output/myhand/` untouched.

The `--debug` overlay ([letters_found.png](letters_found.png)) shows all 30
letters found, correctly named, in the right order, with dots attached to `i`
and `j` only, no merges and nothing else cropped — **with one defect**: the
`E` box excludes its top bar. Intake logged it as `dropped a 365 px mark at
(203, 178, 210, 245) that belongs to no letter`.

This is a regression from the writer-2 grid fix: `_attach_marks` now lets only
`i`/`j` keep a detached mark above the body, so a genuine detached stroke — the
bar of an `E`, the crossbar of a `T` — is discarded. Writer 1's exported `E` is
therefore missing its top bar in the font, since seed letters are passed
through unchanged.

**Measured cost of fixing it naively** (allow any letter to keep a mark above):
LOO **0.411 → 0.403**. Per glyph, `E` 0.574 → 0.481, `D` 0.356 → 0.283, `N`
0.597 → 0.570. Loosening the rule attaches other scraps to other letters and
loses more than the `E` gains. So this is a **product defect, not a score
defect**, and the fix has to be narrower than the version I measured.

---

## Step 2 — Per-glyph breakdown

Sorted worst first. *base* is the Noto Sans content copy scored against the same
real letter; *Δ* is the model's margin over it. *ink* is generated ÷ real ink.
*stroke* is mean half-width in px, real / generated. *segments* is how many
Bézier segments the tracer needs, real / generated — a proxy for how much shape
detail survives.

Image: [per_glyph_worst_first.png](per_glyph_worst_first.png) (real above,
generated below, worst first). Raw: [per_glyph.json](per_glyph.json).

| glyph | tol-F1 | IoU | base | Δ | ink | stroke r/g | segments r/g |
|---|---|---|---|---|---|---|---|
| `S` | 0.168 | 0.048 | 0.117 | +0.051 | 1.42 | 1.32 / 1.38 | 49 / 58 |
| `A` | 0.173 | 0.041 | 0.365 | -0.192 | 0.78 | 1.36 / 1.71 | 83 / 53 |
| `r` | 0.216 | 0.071 | 0.240 | -0.024 | 0.79 | 1.31 / 1.91 | 59 / 16 |
| `m` | 0.220 | 0.074 | 0.067 | +0.153 | 1.15 | 1.39 / 1.75 | 69 / 50 |
| `6` | 0.222 | 0.049 | 0.088 | +0.133 | 1.70 | 1.33 / 1.48 | 52 / 58 |
| `,` | 0.244 | 0.078 | 0.436 | -0.191 | 0.49 | 1.21 / 1.43 | 27 / 17 |
| `O` | 0.277 | 0.088 | 0.205 | +0.072 | 0.96 | 1.29 / 1.40 | 94 / 56 |
| `n` | 0.296 | 0.088 | 0.178 | +0.118 | 1.13 | 1.34 / 1.63 | 56 / 31 |
| `G` | 0.297 | 0.080 | 0.297 | -0.000 | 1.02 | 1.19 / 1.42 | 72 / 55 |
| `B` | 0.339 | 0.118 | 0.142 | +0.197 | 1.14 | 1.35 / 1.57 | 92 / 68 |
| `3` | 0.341 | 0.096 | 0.091 | +0.250 | 1.82 | 1.29 / 1.42 | 45 / 65 |
| `D` | 0.356 | 0.091 | 0.111 | +0.245 | 1.53 | 1.36 / 1.57 | 55 / 46 |
| `g` | 0.365 | 0.125 | 0.302 | +0.063 | 0.96 | 1.39 / 1.53 | 79 / 67 |
| `i` | 0.372 | 0.124 | 0.341 | +0.031 | 0.94 | 1.30 / 1.59 | 35 / 16 |
| `y` | 0.382 | 0.127 | 0.385 | -0.004 | 1.21 | 1.30 / 1.60 | 64 / 60 |
| `M` | 0.391 | 0.119 | 0.140 | +0.252 | 1.71 | 1.32 / 1.64 | 61 / 74 |
| `k` | 0.410 | 0.154 | 0.212 | +0.198 | 1.47 | 1.29 / 2.34 | 51 / 43 |
| `R` | 0.428 | 0.155 | 0.197 | +0.231 | 1.19 | 1.39 / 1.62 | 71 / 64 |
| `j` | 0.459 | 0.161 | 0.228 | +0.231 | 0.84 | 1.29 / 1.60 | 62 / 28 |
| `0` | 0.501 | 0.213 | 0.232 | +0.269 | 1.48 | 1.35 / 1.40 | 55 / 54 |
| `o` | 0.513 | 0.163 | 0.180 | +0.333 | 0.76 | 1.34 / 1.46 | 82 / 32 |
| `f` | 0.520 | 0.198 | 0.294 | +0.225 | 0.81 | 1.29 / 2.02 | 76 / 38 |
| `H` | 0.541 | 0.204 | 0.091 | +0.450 | 1.28 | 1.31 / 1.65 | 79 / 49 |
| `a` | 0.554 | 0.200 | 0.191 | +0.363 | 1.81 | 1.23 / 1.57 | 41 / 43 |
| `E` | 0.574 | 0.269 | 0.322 | +0.252 | 1.31 | 1.30 / 1.63 | 66 / 49 |
| `e` | 0.575 | 0.239 | 0.304 | +0.272 | 0.81 | 1.33 / 1.53 | 52 / 35 |
| `b` | 0.576 | 0.223 | 0.115 | +0.461 | 2.03 | 1.34 / 1.84 | 41 / 42 |
| `N` | 0.597 | 0.208 | 0.065 | +0.532 | 1.49 | 1.34 / 1.55 | 51 / 54 |
| `t` | 0.644 | 0.278 | 0.528 | +0.116 | 0.74 | 1.43 / 1.74 | 54 / 20 |
| `s` | 0.767 | 0.341 | 0.181 | +0.586 | 1.73 | 1.16 / 1.67 | 35 / 43 |

Mean 0.411, baseline mean 0.221, **model beats the content copy on 25 of 30**.

### What the worst letters share

| group | n | mean tol-F1 |
|---|---|---|
| digits + comma | 4 | 0.327 |
| capitals | 11 | 0.376 |
| descenders `g j y` | 3 | 0.402 |
| round forms `o O e a c s S G 3 6` | 9 | 0.413 |
| lowercase | 15 | 0.458 |
| ascenders `b d f h k l t` | 4 | 0.537 |

It is **not** descenders or round forms. The pattern is:

1. **Where my form differs from the conventional one.** `r` (0.216) is the
   clearest — my `r` is a single curved stroke, and the model returns a
   conventional printed `r`; its outline collapses from 59 segments to 16. `A`
   (0.173) and `,` (0.244) are the two glyphs where the model actually loses to
   a plain Noto Sans copy by a wide margin (−0.19 each), meaning the style
   transfer actively moved *away* from my letter.
2. **Digits and capitals**, which I write largest and which come back most
   over-inked (`3` 1.82, `6` 1.70, `M` 1.71, `b` 2.03 — digits+comma mean ink
   1.37 against 1.15 for lowercase).
3. **`S` (0.168)** is the single worst: mine is a small, thin, two-curve `S`,
   the model returns a big even one.

---

## Step 3 — Attribution

Gap to explain: **0.411 → ~0.70** (corpus handwriting category), i.e. 0.29.

### 3.1 Scale / framing — costs ~0.06, and is **not** recoverable at inference

`scripts/scale_cost.py`. The Google Fonts corpus cache is not on this machine
(it lived on the Colab runtime), so the fonts used are the ten handwriting-style
faces that ship with Windows and are not in Google Fonts — the same stand-ins
`scripts/handwriting_e2e.py` uses. Each font is rescaled about the baseline on
the normalized canvas so its x-height lands at the target, then scored.

Each font against **its own** native score, at 25px (my x-height):

| font | native x-height | native | at 25px | change |
|---|---|---|---|---|
| JUICE___ | 63px | 0.740 | 0.479 | −0.261 |
| Gabriola | 37px | 0.801 | 0.712 | −0.089 |
| segoepr | 46px | 0.796 | 0.733 | −0.063 |
| segoesc | 47px | 0.618 | 0.588 | −0.030 |
| MISTRAL | 28px | 0.751 | 0.783 | +0.032 |
| FREESCPT | 28px | 0.670 | 0.709 | +0.039 |
| BRADHITC | 29px | 0.422 | 0.476 | +0.054 |
| Inkfree | 39px | 0.552 | 0.613 | +0.061 |

Fonts that undergo a **real** shrink (native ≥37px, as mine would be):
**mean −0.076**. Fonts already near 25px: +0.042, i.e. noise. So scale is worth
roughly **0.05–0.08**.

But it cannot be collected at inference. Reframing my own sample larger,
resampled from the full-resolution photo (not upscaling the finished crop):

| framing | x-height | LOO | glyphs touching a canvas edge |
|---|---|---|---|
| as delivered (margin +0.06) | 25px | **0.411** | 0 |
| margin 0.00 | 33px | 0.383 | 2 |
| margin −0.10 | 43px | 0.372 | 9 |
| margin −0.20 | 48px | 0.389 | 10 |
| margin −0.30 | 54px | 0.400 | 12 |
| margin −0.45 | 61px | 0.390 | 17 |

Upscaling the finished 128px crops instead is also worse (0.400 / 0.364 / 0.369
/ 0.388 at ×1.2 / ×1.4 / ×1.6 / ×1.8).

The reason is geometric. My hand spans 2.48 x-heights above the line and 0.96
below — 3.44 in total. At the corpus's typical 46px x-height that needs 158px of
canvas; there are 128. **Landing at corpus scale and keeping every letter whole
are mutually exclusive for this hand.** This qualifies README next step 2: on
the current model, clipping costs more than the scale it buys. A model retrained
on x-height framing would have learned to read clipped ascenders, which this
test cannot simulate.

### 3.2 Intake quality — costs ~0.00

Same font, clean render versus a simulated sheet photographed and taken through
the current freehand intake (pen width matched to mine):

| font | clean render | photographed + intake | cost |
|---|---|---|---|
| segoepr | 0.796 | 0.790 | **−0.006** |
| Inkfree | 0.552 | 0.705 | **+0.153** |
| segoesc | 0.618 | — | simulator failed segmentation (9 letters in row 1) |

Photographing a thin hand can *help*: the pen thickens strokes toward what the
model handles best. The segoesc failure is an artifact of the synthetic sheet's
letter spacing, not of intake on a real photo.

On my own sample: 6 of 30 glyphs arrive in 2 pieces where the character needs 1
(`A H b k n t`) — but those glyphs average **0.44**, above the 0.411 overall, so
multi-piece arrival is not a cost; it is simply how I write them. Row baselines
fit cleanly. Photo is 1280×762, pen half-width 2.0px, letters 67px tall.

The one real intake defect is the dropped `E` bar (Step 1), and fixing it
naively *lowers* LOO.

### 3.3 Style hedging — the residual, ~0.20

Measured on my own letters, generated against real:

| | real | generated |
|---|---|---|
| ink | 1.00 | **1.17** |
| slant of the ink's principal axis | 59.6° | **73.2°** (13.6° more upright) |
| outline segments per glyph | 60.3 | **46.1** (−23%) |
| stroke half-width | 1.31px | **1.62px** (+24%) |
| stroke-width spread (CV) | 0.322 | 0.411 |

The model returns a heavier, more upright, simpler hand. Note the last row
**contradicts README known issue 3**: the generated letters are not drawn with a
*more even* pen — their stroke width varies more. What is lost is outline
detail, not variation.

That this is a property of the model rather than of my photograph is confirmed
independently. Across the ten held-out handwriting fonts, rendered cleanly with
no camera at all, LOO correlates with how upright the font is at **r = +0.50**:

| font | LOO | slant |
|---|---|---|
| BRADHITC | 0.422 | 54.1° |
| Inkfree | 0.552 | 68.0° |
| segoesc | 0.618 | 52.8° |
| … | … | … |
| segoepr | 0.796 | 60.6° |
| Gabriola | 0.801 | 71.4° |
| JUICE___ (most upright) | 0.740 | 85.9° |

My hand sits at 59.6°, in the band where the model scores worst, and my 0.411 is
within noise of Bradley Hand's 0.422 — a clean font file.

### 3.4 Seed coverage — worth +0.005

Same 24 glyphs scored both ways, so the numbers are comparable:

| references given | LOO on those 24 glyphs |
|---|---|
| 23 (seed24) | 0.399 |
| 29 (seed30) | 0.403 |

Six extra letters buy **+0.005**. seed30 earns its place by passing `i j f`
through into the exported font, not by improving the model's output.

### Breakdown of the 0.29 gap

| cause | share | basis |
|---|---|---|
| Model regularizing a hard hand | **~0.20 (70%)** | slant r=+0.50; BRADHITC 0.422 clean; +17% ink, −23% detail, +13.6° upright |
| Scale | **~0.06 (20%)** | −0.076 mean when real fonts are shrunk to 25px |
| Seed coverage | ~0.005 (2%) | 23 → 29 references |
| Intake + photograph | **~0.00** | −0.006 segoepr, +0.153 Inkfree |
| unexplained | ~0.02 | — |

---

## Step 4 — Recommendations, ranked by gain per unit of effort

### A. What you can change when writing and photographing

Honest summary: **there is very little here**, which is itself the finding.

| # | change | est. gain | cost | evidence |
|---|---|---|---|---|
| A1 | **Write with a thicker pen** (marker, not gel) at the same letter size | +0.02 to +0.05, unproven for your hand | 10 min to test | Thickening Ink Free's thin strokes gained +0.153; your stroke/x-height is 0.052, mid-range, so expect much less |
| A2 | Write more letters (seed30 → more) | +0.005 | 10 min | §3.4, measured |
| A3 | Better light / higher resolution / flatter photo | ~0.00 | — | §3.2: intake costs nothing already |
| A4 | Larger letters on the page | ~0.00 | — | Scale on the canvas is set by your proportions, not by how big you write |
| A5 | *(excluded by your constraint)* Shorter ascenders and descenders | +0.05 to +0.08 | changes your hand | §3.1 |

A1 is the only one worth trying, and it is a ten-minute test: same 30 letters,
same page, a fibre-tip pen.

### B. Code changes needing no retrain

| # | change | est. gain | cost | evidence |
|---|---|---|---|---|
| B1 | **Narrow fix for detached strokes**: attach a mark above a letter when it lies inside that letter's horizontal span and no other letter claims it, instead of allowing it only for `i`/`j` | 0.00 on the score, but **fixes a visibly broken `E`** in the exported font | ~1 h incl. re-measuring writers 2 and 3 | Step 1: the naive version costs −0.008, so it must be narrow |
| B2 | Inference-time reframing to land bigger | **−0.01 to −0.04** — do not do it | — | §3.1, six framings measured |
| B3 | Width-head tracking correction | already in; your font needed +0.002 em | — | Step 1 log |

Group B cannot move the score. B1 is worth doing for the product, not the
metric.

### C. Changes needing a re-render and/or retrain

Ranked by expected gain per GPU hour. **Not run — experiment plans only.**

**C1. Shear augmentation (new, from §3.3)** — *best value*
The model scores worst on slanted hands (r = +0.50) and returns letters 13.6°
more upright than they were written. The corpus's handwriting category is only
387 fonts and most are near-upright.
*Plan:* add a random shear of ±15° to the target/reference pair during training
(both sheared identically, so the task is unchanged); warm-start from
`hfont_step030000.pt` for 14K steps at lr 1e-4. Measure on the same held-out
val/test splits, plus the three real writers, plus the ten Windows handwriting
fonts with slant recorded — the specific prediction is that the LOO-vs-slant
correlation flattens.
*Cost:* ~1.5 GPU hours, no re-render. *Est. gain:* +0.03 to +0.08 on slanted
hands; near zero on upright ones.

**C2. Reference-to-target attention** (README next step 3)
Targets the largest slice (~0.20). The decoder currently sees one pooled style
vector, so it cannot ask which of my letters a given stroke should resemble, and
hedges — which is exactly what the Dice ablation showed is load-bearing.
*Plan:* replace pooled AdaIN conditioning with cross-attention from decoder
features to per-reference encoder features, keeping the pooled vector as a
residual path so the model degrades gracefully with one reference. Train from
scratch (the decoder's conditioning changes shape, so no warm start), 30K steps.
Compare against run 3 on identical splits; report per-category and the
10th-percentile font, not just the mean.
*Cost:* ~4–6 GPU hours plus implementation. *Est. gain:* unknown, but it is the
only lever aimed at the dominant cause.

**C3. x-height framing** (README next step 2)
*Plan as written in README, with one correction from §3.1:* for this hand,
framing on x-height at corpus scale needs 158px of a 128px canvas, so the
re-render must either raise the canvas to ~176px (≈1.9× the training cost) or
accept clipped ascenders. Both arms should be rendered and trained, because the
inference-only test shows clipping costs more than scale buys on a model that
never saw it.
*Cost:* ~30 min re-render + 1.5–3 GPU hours per arm. *Est. gain:* +0.05 to
+0.08 if clipping is handled; possibly negative if it is not.

**C4. More handwriting data.** 387 handwriting fonts is the smallest category
doing the most work. Adding a handwriting-specific corpus, or augmenting with
elastic warps of existing handwriting fonts, addresses the same root cause as
C1/C2. *Cost:* a day of data work. *Est. gain:* unmeasured.

---

## Reproducing

```bash
# Step 1
python -m hfont.cli font MyHandwriting.jpeg --checkpoint runs/phase1c/hfont_step030000.pt \
    --freehand --seed seed30 --out output/myhand_diag/TarekHand.otf \
    --debug output/myhand_diag/letters_found.png

# Step 2
python scripts/diagnose_writer.py --photo MyHandwriting.jpeg --out output/myhand_diag

# Step 3.1
python scripts/scale_cost.py --sizes 45,35,30,25,20

# Step 3.2 / 3.3 / 3.4
python scripts/attribute_gap.py --photo MyHandwriting.jpeg --out output/myhand_diag
```

`python -m pytest tests -q` → 27 passed.

---

# Round 2 — testing the slant hypothesis causally

No training. Outputs in `round2/`. **The slant hypothesis from round 1 does not
survive. Stroke thinness does.** Details below; the three questions asked at the
end are answered in "Verdict".

## 1. The `E` bar, fixed

A detached stroke now joins a letter when it overlaps that letter's column and
no other letter can claim it. (Not "lies within the span": the top bar of an
`E` is *wider* than the stem it belongs to, which is why my first attempt still
dropped it.) The `i`/`j` restriction on dots is untouched, so writer 2's grid
protection is unchanged.

| writer | LOO before | LOO after |
|---|---|---|
| 1 (blank paper, red gel) | 0.411 | **0.403** |
| 2 (squared paper) | 0.573 | 0.573 |
| 3 (pale orange) | 0.335 | 0.335 |

Writer 1's `E` reference goes from 339px of ink and 59px tall to **439px and
64px**, with 113px of that in the top fifth of the glyph — the bar. The exported
font now sets `BEEFED` with complete `E`s ([E_in_font.png](round2/E_in_font.png)),
and writer 2's font still has no stray dots ([writer2_check.png](round2/writer2_check.png)).

The −0.008 is the cost predicted in round 1: a complete `E` is a harder target
than a truncated one. This buys product correctness, not score.

Regression test: `test_detached_top_bar_stays_with_its_letter`, which fails
without the fix and passes with it. 28 tests pass.

## 2. The slant metric was measuring the wrong thing

`scripts/slant.py` measures the dominant direction of near-vertical stroke
edges — a magnitude-weighted orientation histogram, peak rather than mean,
because averaging across a band reads a 10° shear as 7°. Self-check: shearing a
font by a known angle and measuring it back is accurate to 1–2° on regular
fonts (Gabriola +10.0 → +11.2, JUICE +20.0 → +19.8), worse on Segoe Print's
irregular stems (+10.0 → +7.0).

**r = +0.50 becomes r = −0.02.**

| font | LOO | stem slant | old axis "slant" |
|---|---|---|---|
| Gabriola | 0.801 | −2.5° | 71.4° |
| segoepr | 0.796 | −13.7° | 60.6° |
| JUICE___ | 0.740 | −0.3° | 85.9° |
| MISTRAL | 0.751 | −10.8° | 69.3° |
| PRISTINA | 0.688 | −12.1° | 62.8° |
| FREESCPT | 0.670 | −25.8° | 59.0° |
| RAGE | 0.623 | −22.8° | 50.6° |
| segoesc | 0.618 | −14.6° | 52.8° |
| Inkfree | 0.552 | −6.0° | 68.0° |
| BRADHITC | **0.422** | **−3.5°** | 54.1° |

correlation with LOO: **slant r = −0.02**, |slant| r = +0.02, n = 10.

The worst font is nearly upright and the two most slanted score mid-range. The
old +0.50 was the axis metric tracking glyph proportions, exactly as you
suspected.

**My own hand measures −1.5°, not 59.6°.** It is upright. Round 1's "generated
letters are 13.6° more upright than mine" was an artifact too: the real
difference is **1.3°** (−1.5° real, −0.2° generated).

## 3. Shear intervention: slant *does* cause drops — to hands that have it

Shearing references and targets alike, framing untouched:

| font | 0° | 10° | 15° | 25° |
|---|---|---|---|---|
| Gabriola | 0.801 | 0.604 | 0.477 | 0.371 |
| segoepr | 0.796 | 0.594 | 0.488 | 0.376 |
| JUICE___ | 0.740 | 0.394 | 0.316 | 0.268 |

**Control — is this resampling blur?** No. Shearing +15° and back again (two
interpolations, no net slant) leaves the score alone:

| font | plain | +15°/−15° | 15° |
|---|---|---|---|
| Gabriola | 0.801 | 0.804 | 0.477 |
| segoepr | 0.796 | 0.796 | 0.488 |
| JUICE___ | 0.740 | 0.756 | 0.316 |

So the model is genuinely slant-fragile: 15° of slant costs it ~0.32, more than
the entire gap being investigated. **It is a real weakness and it is not mine.**
At −1.5° there is nothing here to recover.

## 4. Test-time deslant: does not work

| sample | slant | as-is | deslanted | change | glyphs clipped |
|---|---|---|---|---|---|
| writer 1 | −1.5° | 0.403 | 0.399 | −0.004 | 0 |
| BRADHITC | −3.5° | 0.422 | 0.432 | +0.010 | 0 |
| segoesc | −14.6° | 0.618 | 0.442 | **−0.176** | 0 |

Nothing clips. The one sample slanted enough to benefit is the one it damages
most: straightening, generating and leaning back puts two resamplings around a
model whose output is already soft, and the reconstruction loses more than the
uprightness gains.

## 5. Thinness explains what slant did not

Dilating references and targets:

| sample | as-is | +1px | +2px |
|---|---|---|---|
| writer 1 | 0.403 | 0.508 | **0.593** |
| BRADHITC | 0.422 | 0.599 | **0.697** |

**Control — are thick strokes easier, or just easier to score?** Both, and the
model's share is the larger one. The content-copy baseline rises too, but the
model's margin over it grows:

| sample | dilation | model | content copy | margin |
|---|---|---|---|---|
| writer 1 | 0 | 0.403 | 0.223 | +0.181 |
| writer 1 | +1px | 0.508 | 0.269 | +0.239 |
| writer 1 | +2px | 0.593 | 0.305 | +0.288 |
| BRADHITC | 0 | 0.422 | 0.209 | +0.213 |
| BRADHITC | +1px | 0.599 | 0.265 | +0.333 |
| BRADHITC | +2px | 0.697 | 0.310 | +0.387 |

Of writer 1's +0.190 from 2px of dilation, the baseline accounts for +0.082 —
so **roughly 55–60% is a genuine improvement** in what the model draws, and
40–45% is the tolerance metric forgiving fat strokes.

This also fits the round-1 feature scan, rerun here: across the ten fonts,
absolute stroke width is the strongest single predictor of LOO (**r = +0.60**),
ahead of ink per glyph (+0.45) and outline detail (−0.37), with slant last at
−0.02 (`scripts/what_predicts_score.py`). My hand has 1.31px strokes — beside
Bradley Hand's 1.10px, the two thinnest samples and the two worst scores — and
the *most* outline detail of any sample measured (60.7 segments per glyph
against 45–53 for the fonts).

### An inference-only version that works

Show the model dilated references, then erode its output back to the writer's
own weight. **Scored against the unmodified real letters, so no metric
artefact:**

| sample | as-is | fat refs, thinned back | change |
|---|---|---|---|
| writer 1 | 0.403 | 0.417 | **+0.014** |
| BRADHITC | 0.422 | 0.489 | **+0.067** |

## Verdict

**Slant or thinness?** Thinness, for this writer. Slant is a real and large
model weakness — 15° costs ~0.32, proven causally against a resampling control
— but my hand is upright (−1.5°), so none of that applies to it. Thinness is
what my sample actually has, it correlates best across fonts (r = +0.60), and
intervening on it recovers 0.19 of which ~0.11 is genuine.

**What C1 (shear augmentation) is now expected to buy.** For writer 1,
essentially **nothing** — there is no slant to be robust to. Its value is now
better targeted and smaller in scope: it should help the slanted minority
(Segoe Script −14.6°, RAGE −22.8°, FREESCPT −25.8°) and nothing else. Given the
shear curve, the headroom there is large (0.62 → potentially 0.80 for a hand
like Segoe Script's), but it is worth doing only if real writers turn out to
slant; two of our three do not. **Demote it below a stroke-weight intervention**,
which is the same cost and targets the property that actually predicts the
score. The natural experiment is the mirror image of C1: augment training with
random dilation/erosion of ±1px applied to reference and target alike, warm-start
14K steps, and measure whether the LOO-vs-stroke-width correlation flattens.

**Is test-time deslant worth shipping?** No. It is neutral-to-harmful on all
three samples and actively bad (−0.176) on the only one slanted enough to be a
candidate. **The thickness trick in §5 is worth shipping instead** — same
inference-only shape, +0.014 for writer 1 and +0.067 for a thin font, measured
against unmodified targets. It needs a wider test (the other two writers and the
ten fonts) and a guard so it only engages when a sample's strokes are thin
relative to the corpus, but it is a real option that costs no GPU time.

## Reproducing round 2

```bash
python scripts/slant.py --selfcheck
python scripts/slant_causal.py --measure --shear-curve --deslant --dilate --controls
python scripts/what_predicts_score.py
```

`python -m pytest tests -q` → 28 passed.
