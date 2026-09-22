# Adversarial review

Branch `review`, from `main @ a2e0c45`. No retraining, no re-rendering, no gate
or metric replaced. Every number below was measured on this machine with the
shipped checkpoint; where a sample is small the uncertainty is given.

**Scope warning, stated first:** four of the five subagents I dispatched for the
audit and the literature search were killed by a session rate limit before
reporting. I did the code audit myself and it is therefore narrower than
planned, and the few-shot font-generation literature review is **not done**.
What is here is what was actually measured. See DECISIONS.md §7.

---

## The three findings that matter most

### 1. The primary metric is blind at the stroke width real handwriting has

`scripts/metric_sensitivity.py`. Each perturbation is applied to the prediction
only, target fixed, averaged over all 30 of a writer's glyphs:

| writer 1 (1.3px strokes) | IoU | tol-F1 | clDice | ink |
|---|---|---|---|---|
| identical | 1.000 | 1.000 | 1.000 | 1.00 |
| shifted 1px | 0.606 | **1.000** | 0.945 | 1.00 |
| dilated 1px | 0.625 | **1.000** | 0.989 | 1.60 |
| dilated 2px | 0.457 | 0.907 | 0.978 | 2.19 |
| broken stroke | 0.918 | 0.985 | 0.954 | 0.92 |
| a different letter | 0.065 | 0.227 | 0.125 | 1.24 |

A 1px shift costs tol-F1 **0.0% of its range** at every stroke width tested
(Segoe Print and Times too). A glyph carrying 60% more ink scores *exactly the
same as a perfect one*. The reason is arithmetic: the tolerance is 1.5px and a
real hand's stroke is 1.3px, so the tolerance is wider than the object being
measured — the failure mode Cheng et al. name for Trimap IoU and boundary
F-measure ("ignores significant errors when the object size is comparable to
d", [Boundary IoU, CVPR 2021](https://arxiv.org/abs/2103.16562)).

Fattening a **wrong** glyph also inflates it, measured on real glyphs:
0.227 → 0.308 across 0–3px of dilation (+36% relative). IoU (0.065 → 0.110) and
clDice (0.125 → 0.190) inflate too, but less in absolute terms.

**Consequence:** every ranking this project has made on tol-F1 is weaker than
stated, including README's headline table, the run2-vs-run3 comparison, and my
own round-2 conclusion (§ "Corrections"). **Fix applied:** `cl_dice` added
alongside the existing metrics — nothing replaced, no gate touched.

### 2. The export path has a cliff just below where real hands land

`scripts/export_fidelity.py`, model removed from the loop: rasters → Béziers →
OTF → rasters, scored against the input.

| sample | stroke half-width | IoU | tol-F1 | ink kept |
|---|---|---|---|---|
| Times native | 2.56px | 0.980 | 1.000 | 1.00 |
| Segoe Print native | 2.13px | 0.976 | 1.000 | 0.99 |
| **writer 1 as intake delivers him** | **1.31px** | **0.925** | 1.000 | 0.99 |
| Bradley Hand native | 1.10px | 0.864 | 0.987 | 0.98 |
| Bradley Hand eroded | **1.02px** | **0.038** | 0.239 | **0.37** |

README's "the tracer is not the problem, 0.98 IoU" was measured on Times at
2.6px. At a real hand's weight the same path costs 0.05–0.11 IoU, and 0.2px
thinner it collapses entirely, losing 63% of the ink. This is a **ceiling on
the product that no model change can lift** — and it is invisible to tol-F1,
which reports 1.000 for the writer-1 round trip and 0.239 for the collapse.

### 3. An exported font contains two different stroke weights

`scripts/seed_vs_generated.py`, measured on the OTF itself. The ~30 letters the
writer wrote are passed through; the other ~45 are drawn by the model.

| writer | written | generated | gap | Welch t |
|---|---|---|---|---|
| 1 | 1.32px | 1.57px | **+19%** | 8.3 |
| 2 | 1.17px | 1.48px | **+27%** | 8.2 |
| 3 | 1.08px | 1.40px | **+29%** | 8.5 |

Unlike the correlations this project has relied on, this is well powered
(n = 30/45 glyphs per font, three fonts, same sign, t > 8). A word puts a light
written letter beside a heavy generated one. **I tried the obvious fix and
rejected it** — see "Tried and rejected".

---

## Corrections to earlier conclusions, including my own

| claim | status after this review |
|---|---|
| Round 2: "thinness is what costs writer 1; intervening recovers 0.19, ~0.11 genuine" | **Overstated by roughly 10×.** The intervention that cannot be inflated — dilate references only, thin the output back, score against *untouched* letters — gives **+0.014 tol-F1 / +0.005 clDice** for writer 1 and +0.067/+0.037 for Bradley Hand. The 0.19 came mostly from dilating the *target*, which widens the band a wrong stroke can land in. My round-2 "fatness control" (margin over baseline) was too weak to catch this. |
| Round 2: "LOO correlates with stroke width at r=+0.60" | **Underpowered.** 95% CI [−0.05, +0.89]; at n=10 nothing below \|r\|=0.63 is significant. |
| Round 1: "slant explains it, r=+0.50" | Already refuted in round 2; note the CI was [−0.19, +0.86] — it was never evidence. |
| Round 2: "slant does not correlate, r=−0.02" | The **refutation is also underpowered** (CI [−0.64, +0.62]). Round 2's conclusion survives only because of the *intervention* (15° of shear costs 0.32, with a resampling control showing +15°/−15° is free). Correlations at n=10 settled nothing in either direction. |
| README: "LOO predicts held-out quality, r=0.98" | **Survives its control.** I suspected range effects across categories; within HANDWRITING alone r = +0.974 [0.96, 0.98], n=56. Still untested for *photographed* hands, where no held-out ground truth exists. |
| README: "the excess ink is load-bearing" (Dice ablation) | **Unresolved, and now suspect.** The ablation ranked arms by tol-F1, which is blind to exactly the variable being manipulated. Re-scoring needs the three arms' checkpoints, which were on the reclaimed Colab runtime. Logged as blocked; the per-font re-analysis was assigned to a subagent that was rate-limited. |
| README: "the tracer is not the problem (0.98 IoU)" | **True only at typeset stroke weights.** See finding 2. |

---

## Audit (done by me; narrower than planned)

| decision | stated reason | verdict |
|---|---|---|
| tol-F1 as the fair metric for thin strokes | IoU punishes 1px offsets on thin strokes | **Half right, and the fix overshot.** IoU's harshness is real, but the replacement is blind rather than fair. Both should be read together, with clDice as the weight-blind third. |
| `leave_one_out` implementation | proxy for real-writer quality | **Correct as implemented** (`evaluate/report.py:62`): holds one glyph out, encodes style from the rest, scores against the real glyph. No leakage. |
| `clean_raster` prunes components < 1.5% of the largest before tracing | remove specks | **Safe.** An `i` dot is 7–13% of its stem at these sizes, well above the threshold. Checked because it could have been silently deleting dots; it is not. |
| Template intake uses a per-cell contrast window (0.25) while freehand uses 0.9 | freehand crops are letter-sized | **Not a bug.** Probed a 12px pen in a 200px template cell: both windows recover 100% of the ink. The freehand path needed 0.9 because its crops are ~99px, where the window approaches the stroke width. |
| Seed letters passed through unchanged | "a generated substitute could only be worse" | **Right for fidelity, wrong for consistency** — it is what creates finding 3. The premise is sound per-glyph and creates a defect per-font. |
| 128px canvas, one scale per font | proportions are the style | **This is the binding constraint.** It puts real hands at 15–25px x-height and 1.1–1.3px strokes, i.e. straight onto the export cliff in finding 2. |

Not audited, for lack of agent budget: the model internals (pooled style vector,
AdaIN, advance head, the bucketing-to-32 in the style encoder), the loss
interactions, the training schedule across warm starts, and `dataset.py`
sampling. These are the areas where I would look next.

---

## Research

Only the thin-structure-metrics search completed. Its central claims, **with my
verification against this project's real data**:

* **clDice** — Shit et al., CVPR 2021, [arXiv:2003.07311](https://arxiv.org/abs/2003.07311).
  Skeleton-of-each against body-of-the-other, harmonic mean. Adopted, alongside
  the existing metrics.
* **The fat-bias of tolerance metrics is a known pathology**, tracing to Mnih &
  Hinton's relaxed precision/recall (ECCV 2010) and named explicitly for Trimap
  IoU and boundary F-measure in
  [Boundary IoU, CVPR 2021](https://arxiv.org/abs/2103.16562). Confirmed here.
* **Recommended controls** — report ink statistics beside every score, stratify
  by stroke width, sweep the tolerance, and run a dilation adversary as a unit
  test. I have adopted the dilation adversary as a test
  (`test_cl_dice_sees_shape_not_stroke_weight`).
* **A claim that did NOT replicate.** The agent's synthetic probe reported a
  wrong glyph climbing 0.204 → 0.678 under dilation. On real glyphs the same
  test gives 0.227 → 0.308. The effect is real and one-directional but roughly
  four times smaller than the synthetic figure. Recorded because accepting the
  headline number would have overstated the case.
* **A warning worth keeping** — Heil & Breznik (2023,
  [arXiv:2303.02761](https://arxiv.org/abs/2303.02761)) found morphological
  *erosion* augmentation significantly hurt thin-stroke stenography. At 1.3px,
  erosion destroys the stroke. Any future stroke-weight augmentation should
  dilate only, or re-threshold a supersampled render.
* Font-generation papers report L1/RMSE/SSIM/LPIPS/FID and user studies; the
  survey ([arXiv:2508.06900](https://arxiv.org/abs/2508.06900)) records **no
  standard structural metric** and no standard dataset.

---

## Ranked weaknesses

| # | weakness | evidence | severity for a real person's font | category |
|---|---|---|---|---|
| 1 | Resolution/export cliff at real stroke widths | IoU 0.925 at 1.31px, 0.038 at 1.02px, model excluded | **High** — a hard ceiling; writer 3 is 1.08px, inside the danger band | (d) |
| 2 | Primary metric blind to sub-stroke error and to weight | 0.0% of range on a 1px shift; identical score at +60% ink | **High** — it is why defects survive review | (b), done: clDice added |
| 3 | Two stroke weights inside one exported font | +19/27/29%, t > 8 | **Medium-high**, visible in any word | (d) — the naive (b) fix was measured and rejected |
| 4 | Conclusions drawn from n≈10 correlations | every reported r has a CI spanning 0 | **Medium** — it has already produced one wrong headline (round 1) and one overstated one (round 2) | (c) — use interventions with controls, not correlations |
| 5 | Dice-ablation conclusion rests on the blind metric | arms ranked by tol-F1 while manipulating ink | **Medium** | (c), blocked: checkpoints gone |
| 6 | LOO unvalidated for photographed hands | r=0.98 holds on fonts, including within handwriting; no ground truth exists for a real hand's unwritten letters | **Medium** — it is the number shown to users | (c) |
| 7 | Model internals unaudited this round | — | unknown | (c) |

---

## What I changed

1. **`cl_dice` added to `evaluate/metrics.py`**, reported alongside IoU, tol-F1,
   SSIM and coverage. Nothing replaced. Test:
   `test_cl_dice_sees_shape_not_stroke_weight` (fails without it). Its docstring
   was corrected after measurement showed my first claim — that it punishes
   broken strokes — was false (0.954, barely below 1.000).
2. **Three measurement scripts** (`metric_sensitivity.py`, `export_fidelity.py`,
   `seed_vs_generated.py`, `recheck_thinness.py`), each stating the question it
   answers.

Tests: 29 passing at every commit. Lint clean.

## Tried and rejected

**Matching generated stroke weight to the writer's own** (a threshold remap, not
an erosion, measured through the builder's own conditioning so the correction
survives to trace time, with a floor at 1.15px half-width to stay off the export
cliff).

It worked as designed on the defect: the in-font gap fell from +19/27/29% to
**+6/15/19%**. But scored against the writers' real letters it made every glyph
worse:

| | tol-F1 off → on | IoU | clDice | coverage |
|---|---|---|---|---|
| writer 1 | 0.403 → 0.366 | 0.144 → 0.119 | 0.233 → 0.192 | 1.22 → 1.02 |
| writer 2 | 0.573 → 0.536 | 0.212 → 0.186 | 0.331 → 0.298 | 1.38 → 1.09 |
| writer 3 | 0.335 → 0.292 | 0.088 → 0.076 | 0.152 → 0.127 | 3.02 → 2.38 |

Mean **−0.039**, four times the −0.010 bar, and down on the weight-blind metric
too — so it is not a scoring artifact. The excess ink is shape error, as round 1
found when raising the trace level: thinning removes ink that was partly
overlapping the truth rather than moving the stroke to the right place.
**Reverted** (`git checkout src/hfont/generate.py`); the code on this branch is
unchanged from main.

The lesson generalizes: the weight mismatch cannot be post-processed away,
because the model is not drawing the right shape too fat — it is drawing a
slightly wrong shape whose extra ink happens to overlap.

---

## For the next round (needs training or re-rendering)

**P1. Render and train at 192px or 256px instead of 128px.**
*Prediction:* the export round trip at a real hand's proportions rises from
0.925 to >0.97 IoU, and the 1.02px collapse disappears, because the same
physical stroke lands on 2–2.6px instead of 1.3px. *Comparison:* re-run
`scripts/export_fidelity.py` at the new size — that part needs no model at all
and settles the ceiling question on its own, before any training. Then retrain
and compare on identical held-out fonts at matched steps. *Cost:* re-render
(~30 min) + ~2.3× the GPU time per step; ~4–6 T4-hours for a 14K-step warm
start, which is not possible from the current checkpoint (input size changes),
so budget a fresh 30K-step run.
*Do the no-model half first: it is free and it either confirms or kills P1.*

**P2. Stroke-weight augmentation, dilate-only.**
*Prediction:* thin-hand LOO improves and the LOO-vs-stroke-width relationship
flattens. *Caveat now measured:* the inference-time version of this buys only
+0.005–0.037 clDice, so the expected gain is small; and erosion must not be used
([arXiv:2303.02761](https://arxiv.org/abs/2303.02761)). *Cost:* ~1.5 T4-hours
warm start. **Lower priority than round 2 implied.**

**P3. Re-score the Dice ablation with clDice.** Needs the three arms'
checkpoints, which no longer exist. Cheapest honest version: re-run the ablation
at 3×4K steps (~1 GPU-hour) and score every arm with IoU, tol-F1 **and** clDice.
*Prediction:* if the Dice=0 arm's deficit shrinks or reverses under clDice, the
"excess ink is load-bearing" conclusion was a metric artifact.

**P4. Reference-to-target attention** — unchanged from the earlier report, and
still the only lever aimed at the dominant cause, but now explicitly *not*
justified by any correlation. Run it as an intervention with a control, not as a
hypothesis defended by an r value.
