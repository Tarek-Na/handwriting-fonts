# Adversarial review

Branch `review`, from `main @ a2e0c45`. No retraining, no re-rendering, no gate
or metric replaced. Every number below was measured on this machine with the
shipped checkpoint; where a sample is small the uncertainty is given.

**Scope warning, stated first:** four of the five subagents I dispatched for the
audit and the literature search were killed by a session rate limit before
reporting. I did the whole code audit myself instead — it is now complete,
including the model internals, losses, training schedule and `dataset.py` that
were listed as unreached in the first draft of this report — and I have since
done the few-shot font-generation literature review by hand as well, against
this project's open decisions rather than as a general sweep, reading the
survey in full and the other papers' abstracts only. What is here is what was
actually measured or actually read. See DECISIONS.md §7 and §14.

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

**Resolution buys much less of this back than I first claimed.** Running the
same round trip at larger canvases (no model, no training — the free half of
plan P1 below, run immediately after proposing it):

| sample | 128px | 192px | 256px |
|---|---|---|---|
| writer 1, photographed | 0.925 | 0.938 | **0.939** |
| Bradley Hand | 0.840 | 0.920 | **0.935** |
| Segoe Print | 0.974 | 0.977 | 0.976 |
| Times | 0.979 | 0.985 | 0.986 |

A thin *font* recovers most of the loss (+0.095), and the 1.02px collapse is a
pure quantization artifact that a bigger canvas removes. **A real hand recovers
+0.014 and stops at 0.939.** At 256px writer 1's strokes are 2.03px — thicker
than Segoe Print's 2.10px at 128px, which scores 0.974 — so the remainder is not
stroke width. Nor is it contour noise the tracer could be smoothed past:
pre-smoothing the photographed raster before tracing makes it monotonically
worse (0.925 → 0.880 at σ=0.6px → 0.809 at σ=1.0px at 128px), because it
destroys real detail along with the jitter.

So the export gap for a photographed hand is largely irreducible with this
tracer, and **P1's prediction is falsified before it cost a GPU-hour** (see
plans).

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
| `clean_raster` prunes components < 1.5% of the largest before tracing | remove specks | **Safe, now measured rather than reasoned.** Across the three writers' 90 photographed letters, their 135 generated glyphs and six rendered fonts' 180 glyphs, the only components pruned are 1–2px specks (5 photographed, 3 generated, **0 of 180 font glyphs**). No dot, bar or accent is ever dropped. |
| Template intake uses a per-cell contrast window (0.25) while freehand uses 0.9 | freehand crops are letter-sized | **Not a bug.** Probed a 12px pen in a 200px template cell: both windows recover 100% of the ink. The freehand path needed 0.9 because its crops are ~99px, where the window approaches the stroke width. |
| Seed letters passed through unchanged | "a generated substitute could only be worse" | **Right for fidelity, wrong for consistency** — it is what creates finding 3. The premise is sound per-glyph and creates a defect per-font. |
| 128px canvas, one scale per font | proportions are the style | **This is the binding constraint.** It puts real hands at 15–25px x-height and 1.1–1.3px strokes, i.e. straight onto the export cliff in finding 2. |

### The areas the dead agents never reached — audited since, outcomes below

| area | what I checked | verdict |
|---|---|---|
| Style vector out of distribution at inference | Training draws K ∈ [1,8] references; the product feeds all 29–30, and masked **max** pooling grows with K. Scored leave-one-out at K = 4, 8, 16 and all, five reference draws each. | **Not a defect — hypothesis falsified.** Writer 1's apparent advantage at K=4 is inside the seed spread (+0.005 tol-F1, 95% CI [−0.006, +0.016], p=0.39, paired per glyph). For both control fonts *more* references is reliably better: K=8 costs BRADHITC −0.024 [−0.031, −0.016] and segoepr −0.016 [−0.022, −0.011], both p < 0.001. Feeding everything is right. |
| Bucketing the encoded reference count to a multiple of 32 | Whether discarded filler slots can contaminate the kept ones | **Correct.** `encode_idx` is built valid-first, so `encoded[:n_valid]` lines up with `valid_idx`; the backbone is instance-norm throughout, so a subset encodes identically to the whole. At inference (b=1, K=29) no filler is added at all. |
| The advance head — a 2-layer MLP on (style, char) that never sees the glyph | Mean absolute advance error in em on the ten held-out Windows handwriting faces, generated glyphs only, against three baselines a person could write in an afternoon | **Earns its place.** head 0.0386 (r = 0.876) vs content-font-advance × one fitted constant 0.0571, ink width + bearings 0.0591, seed median 0.0686. Paired over fonts: −0.0185 [−0.0298, −0.0073] p=0.010, −0.0205 [−0.0294, −0.0115] p=0.002, −0.0300 [−0.0422, −0.0178] p=0.001. Wins on 9, 9 and 10 of 10 fonts. `scripts/advance_head.py` |
| Loss interactions | Whether the banded L1 still rewards over-inking, and what Dice does on a blank target | **As documented.** The band is a dilation of target ink, so both sides of every edge carry weight 4 and the asymmetry the docstring describes is genuinely removed; ink further than 2px from any stroke is the only place extra ink is cheap. `soft_dice` on a blank target is 0 loss via the epsilon, not a divide-by-zero. |
| `dataset.py` sampling | Whether the answer can leak into the references | **No leak.** `distinct_keys` excludes the target key *and* every glyph with the same pixel hash, so a reference can never be a copy of the answer even when two characters render identically. |
| The training schedule across warm starts | `warm_start` restores generator + EMA only; the discriminator is never carried between runs, and it takes no update before `adversarial_start` | **Real, and already reported.** In run 3 (`init_from` run 2, `adversarial` 0.1, `char_aux` 0.1, `adversarial_start` 8400 of 14000 — read off the shipped checkpoint's own config) a generator with ~24K cumulative steps starts taking adversarial and char-CE gradient from a discriminator initialised at random that same step. README:261 already says so and reports the consequence as measured: "with a fresh discriminator and 5,600 steps, the adversarial phase was close to a no-op." The mechanism explains the measurement. Category (d); the cheap fix next round is to let D train alone for a few hundred steps, or to save and restore it across warm starts. |
| `ShapingReport.overlapping_pairs` | Whether the declared field was ever computed | **Bug, fixed.** See "What I changed". |

Documented step counts check out: 10K + 16K + 14K = the 40K both READMEs claim,
and `hfont_step030000.pt` carries `step=14000` internally because the filename
counts from run 2. The shipped file is a slimmed export — generator, config and
charset only — which is why it has no discriminator in it.

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

### The few-shot font-generation review, done late and by hand

The agent assigned to this died on the rate limit, so I did it myself against
this project's four open decisions rather than as a general sweep. **Read:** the
survey's full HTML text and the abstracts of the papers named below — not each
paper's experimental section. The literature is overwhelmingly Chinese-glyph
work, which is a standing caveat for a Latin project: Chinese glyphs are far
denser, so anything driven by stroke density transfers badly.

**1. The pooled style vector is a named, documented failure mode — and it is
exactly this project's style encoder.** The survey states it directly: *"An
average operation is usually performed on the extracted features, which easily
weakens the local information and results in the loss of fine-grained details,"*
and that with only a global style representation the model *"is subject to its
capability to represent diverse partial font style shifts."* `StyleEncoder`
pools K references to one 256-d vector by masked mean ⊕ max
(`models/generator.py:105`). The concrete alternative is
[FS-Font, CVPR 2022](https://arxiv.org/abs/2205.09965): content glyph features
as queries, reference features as keys and values, so *"each spatial location in
the content glyph can be assigned with the right fine-grained style"* —
replacing global disentanglement rather than supplementing it. **This promotes
P4 from a hunch to the field's consensus diagnosis of exactly our architecture.**
Honest limit: FS-Font's abstract reports no numbers, only that it beats prior
work and wins user studies, so I cannot say how large the gain is, and I did not
read its tables.

**2. Resolution has no standard, but the one paper with our exact goal goes
far higher than us.** Reported sizes run 64×64 (FTransGAN), 320×320
(TE141K/Fonts-100), 64×64 fixed output (MF-Net), up to 1024×1024
(FontTransformer). [HFH-Font, arXiv:2410.06488](https://arxiv.org/abs/2410.06488)
is the closest match to this pipeline — raster synthesis *for the purpose of
vectorization* — and it generates at **1024×1024 or higher**, producing *"high-
fidelity, high-resolution raster images which can be vectorized into high-quality
vector fonts."* That is external support for finding 2: people who need vector
output do not try to get it from a small canvas. **It does not revive P1.** My
own measurement stands: for a real Latin hand the export round trip saturates by
192–256px (0.925 → 0.938 → 0.939). The 1024 figure is paid for by Chinese stroke
density, and nothing here suggests 8× more canvas would repay itself for Latin.

**3. Nothing in the field would have caught the metric problem.** The survey's
metric list is MAE, MSE, PSNR, SSIM, FID, LPIPS, plus visual comparison and user
studies. No structural or topological measure appears anywhere in it. So clDice
is a genuine addition rather than a standard this project had overlooked — and
the fat-bias in finding 1 would have survived the field's entire standard
toolkit unnoticed.

**4. Photographed handwriting has no baseline in this literature.** The survey
does cover personalised handwriting from user samples — Lian et al.'s system
decomposing handwriting into stroke shape style — but the input is clean glyph
samples. Camera-captured handwriting is not treated. `intake.py` is therefore
being judged against nothing, which cuts both ways: no competitor, and no
published failure mode to check ourselves against.

---

## Ranked weaknesses

| # | weakness | evidence | severity for a real person's font | category |
|---|---|---|---|---|
| 1 | Resolution/export cliff at real stroke widths | IoU 0.925 at 1.31px, 0.038 at 1.02px, model excluded | **High** — a hard ceiling; writer 3 is 1.08px, inside the danger band | (d) → **largely fixed without retraining**: percentile framing doubled glyph size and pen normalisation lifted faint strokes. Export IoU now 0.940 / 0.934 / 0.782 (was 0.925 / 0.889 / 0.551). See README "Intake". |
| 2 | Primary metric blind to sub-stroke error and to weight | 0.0% of range on a 1px shift; identical score at +60% ink | **High** — it is why defects survive review | (b), done: clDice added |
| 3 | Two stroke weights inside one exported font | +19/27/29%, t > 8 | **Medium-high**, visible in any word | (d) — the naive (b) fix was measured and rejected. **Worse after the framing fix** (writer 2: +29% → +48%). **Now fixed without training** by going the other way: thicken the writer's letters on a sub-pixel distance field to meet the model's weight, at font assembly. Gap +26/50/20% → −0/+1/+0%, skeletons intact (clDice ~0.99), LOO unchanged by construction. Cost: a light pen gets a bolder font. |
| 4 | Conclusions drawn from n≈10 correlations | every reported r has a CI spanning 0 | **Medium** — it has already produced one wrong headline (round 1) and one overstated one (round 2) | (c) — use interventions with controls, not correlations |
| 5 | Dice-ablation conclusion rests on the blind metric | arms ranked by tol-F1 while manipulating ink | **Medium** | (c), blocked: checkpoints gone |
| 6 | LOO unvalidated for photographed hands | r=0.98 holds on fonts, including within handwriting; no ground truth exists for a real hand's unwritten letters | **Medium** — it is the number shown to users | (c) |
| 7 | Discriminator restarts from random weights at every warm start, and takes no update before the generator first sees its gradient | run 3's own config: `adversarial_start` 8400 of 14000, `warm_start` loads generator + EMA only | **Low as configured** — README already measures the adversarial phase as "close to a no-op" at weight 0.1, which is what this mechanism predicts | (d) |
| 8 | Letter spacing is uniform where real hands vary | ink-to-ink gap, median 0.041 em across the three writers vs 0.057 em across six disconnected real faces (Mann-Whitney p=0.001, n=298 vs 599); zero pairs collide in our fonts against 1–12 per sample in theirs | **Low** — the text sets correctly; it is a flatness, not a fault | (c) |

---

## What I changed

1. **`cl_dice` added to `evaluate/metrics.py`**, reported alongside IoU, tol-F1,
   SSIM and coverage. Nothing replaced. Test:
   `test_cl_dice_sees_shape_not_stroke_weight` (fails without it). Its docstring
   was corrected after measurement showed my first claim — that it punishes
   broken strokes — was false (0.954, barely below 1.000).
2. **The collision check the shaping report only declared.**
   `ShapingReport.overlapping_pairs` was a field nothing ever assigned, and it
   was absent from `summary()`, so every font this project has exported
   reported zero colliding letter pairs — not because they did not collide but
   because nothing looked. `ink_collisions()` now shapes the sample strings
   with the real advances and compares neighbours' ink boxes, which is the one
   thing the per-glyph widening rule in `vectorize` cannot see.

   | | before | after |
   |---|---|---|
   | writer 1 | 0, unmeasured | **0 of 100 pairs**, worst 0.000 em |
   | writer 2 | 0, unmeasured | **0 of 100**, worst 0.000 em |
   | writer 3 | 0, unmeasured | **0 of 98**, worst 0.000 em |

   Control, so the zero means something: the same check on real faces gives
   Inkfree 1, calibri 1, times 5, BRADHITC 9, JUICE 11, segoepr 12, Gabriola
   27, segoesc 56, PRISTINA 59, RAGE 61, MISTRAL 72, FREESCPT 87 — firing on
   exactly the classic overhang pairs (`fo`, `PA`, `Wa`, `Th`). So the three
   shipped fonts genuinely have no collision defect; that is now a measured
   fact instead of an unset field. **Reported, never fatal** — overhang is
   normal typography and making it fail would change a gate. Scores are
   untouched: no generation code changed. Test:
   `test_shaping_report_counts_colliding_letters`, which against the old code
   fails with `[PASS] lop_report.otf … assert 0 > 0` on a font built to
   collide, and whose control asserts a well-spaced font still reports zero.
3. **Four measurement scripts** (`metric_sensitivity.py`, `export_fidelity.py`,
   `seed_vs_generated.py`, `recheck_thinness.py`, `advance_head.py`), each
   stating the question it answers.

Tests: 30 passing at every commit (29 before this round's second fix).
`python -m pyflakes src/hfont scripts tests` clean — that is the linter
available on this machine; ruff and flake8 are not installed here, so earlier
drafts of this report saying "lint clean" meant pyflakes.

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

**P1. Render and train at 192px or 256px instead of 128px. — PREDICTION
FALSIFIED, downgraded.**
*I predicted* the export round trip would rise from 0.925 to >0.97 IoU at a real
hand's proportions. *Measured:* **0.939 at 256px**, +0.014, and flat between 192
and 256. The prediction was wrong, and the free half of the experiment is what
showed it. What survives: the 1.02px collapse is a quantization artifact that a
bigger canvas does remove, which matters for writer 3 (1.08px) and for any very
thin hand, and thin *fonts* gain ~0.10.
*What is still untested:* whether the **model** draws better with more pixels
per stroke — a separate question from the export path, and the only remaining
reason to try it. If run: fresh 30K-step training (no warm start, the input size
changes), ~4–6 T4-hours, compared against run 3 on identical held-out fonts,
reporting IoU and clDice, not tol-F1. *Given the falsification above, I would
not spend that before P4.*

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

**P4. Reference-to-target attention — now the best-supported item on this list.**
Unchanged in substance from the earlier report, and still the only lever aimed at
the dominant cause, but no longer resting on a correlation: the survey names
average-pooled style as the cause of exactly the symptom this project has (lost
fine-grained local detail), and FS-Font is a worked instantiation. Run it as an
intervention with a control, not as a hypothesis defended by an r value — and
score it on IoU and clDice, since tol-F1 cannot see what it would change.
