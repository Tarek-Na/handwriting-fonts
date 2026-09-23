# The model is not imitating the writer

Asked after the adversarial review, in answer to a plain question: does this
actually match writer 2's and writer 3's handwriting?

**No. It produces close to the same handwriting whoever wrote the samples.**
Reproduce with `python scripts/style_collapse.py`.

---

## The observation

Generate the 45 letters nobody wrote, once from writer 2's samples and once from
writer 3's, and compare the two sets of *generated* letters to each other:

| | writer 2 vs writer 3 |
|---|---|
| style vectors, cosine similarity | **0.996** |
| generated letters, clDice | **0.943** |
| generated letters, IoU | 0.754 |

A cosine of 0.996 between two 256-d style codes means the encoder has placed two
different people at essentially the same point. The output follows: 0.943 clDice
is not "similar", it is the same letters. Look at row 2 of
`how_good_writer2.png` and `how_good_writer3.png` — they are interchangeable,
while row 1 of each (what the two people actually wrote) plainly is not.

## The control that makes it a finding

The same measurement across the four fonts the model was trained on:

| pair group | style-vector cosine | generated similarity (clDice) |
|---|---|---|
| the three photographed hands (n=3 pairs) | **0.982** | 0.561 |
| the four fonts (n=6 pairs) | **0.725** | 0.542 |

Individual font pairs spread from 0.421 (times vs Mistral) to 0.844. So the
encoder is not broken and the style path is not dead — given fonts, it
discriminates. Given photographs, it does not. The mechanism works on the
distribution it was trained on and collapses on the one the product actually
uses.

## What it is not

**Not stroke weight.** Photographed hands arrive at ~1.18px where fonts sit at
~2.11px, so thinness was the obvious suspect. It is not the cause:

| condition | stroke px | mean pairwise cosine |
|---|---|---|
| hands, as intake delivers them | 1.18 | 0.982 |
| **hands dilated toward font weight** | 1.58 | **0.970** |
| fonts, native weight | 2.11 | 0.725 |
| fonts eroded toward hand weight | 1.75 | 0.774 |

Dilating the hands most of the way to font weight moved their similarity by
0.012 — they stay collapsed. The erosion arm is weaker evidence and is reported
as such: `disk(1)` only took the fonts from 2.11px to 1.75px, about a third of
the gap, and eroding further destroys the strokes outright (Heil & Breznik 2023,
cited in REVIEW.md). The dilation direction is the one that carries the
argument, and it is clean.

**Not a small-sample correlation.** There are only three hands, so the 0.982 is
a mean over three pairs. But the claim does not rest on it: writer 2 versus
writer 3 is one pair, measured directly, at cosine 0.996 and output clDice
0.943. That is a single decisive observation, not a trend.

## Why nothing caught this

Every quality number this project reports — leave-one-out, the per-font tables,
the r=0.98 validation — is computed **within one style**: hold a letter out,
generate it from that same person's other letters, compare it to what that
person wrote. No comparison is ever made *across* styles. A model that ignored
the style input entirely would still score exactly what it scores today on every
one of those measurements. The evaluation is structurally incapable of seeing
this failure, which is why it survived a full adversarial review aimed at the
metrics.

The review did name the mechanism, in the literature section, without connecting
it to this: the survey ([arXiv:2508.06900](https://arxiv.org/abs/2508.06900))
states that *"an average operation is usually performed on the extracted
features, which easily weakens the local information and results in the loss of
fine-grained details."* `StyleEncoder` pools K references to one vector by
masked mean ⊕ max. This is that failure, observed.

## What it means for the three fonts already exported

They are not worthless, because 30 of the 75 glyphs are the person's real
handwriting passed through unchanged — `generate_rasters` substitutes the
written letters back over the generated ones. The remaining 45 are close to a
generic hand. So the exported font reads as *somewhat* personal in text that
happens to use the written letters, and drifts toward a stranger's handwriting
everywhere else. That also explains why the fonts looked more convincing than
their scores suggested.

## Consequence for what to do next

This does not change the ranking so much as collapse it. **P4
(reference-to-target attention) is no longer one option among four — it is the
only one that addresses this**, and the finding upgrades it from "the field's
stated diagnosis" to "the measured defect in this model". P1 was already
falsified; P2 (stroke-weight augmentation) and P3 (re-scoring the Dice ablation)
would both leave the model producing one hand for everybody.

Two cheaper things should be measured first, because they decide whether a
retrain is even the right lever:

1. **Is it intake or the encoder?** Render a font, put it through the freehand
   intake path as if photographed, and compare its style vector to the same
   font's vector taken directly. If intake is washing out the differences, the
   fix is in `intake.py` and costs no GPU at all. This is the first thing to run
   and it is cheap.
2. **Does the collapse exist in training too?** The corpus is on Colab, not here.
   Measuring cross-style similarity across held-out *corpus* fonts would say
   whether the model was ever learning style or whether the font-pair spread
   above is mostly the content path responding to different letterforms.

Neither was run in this round: the first needs a decision about whether
synthesising a fake photograph is a fair test, and the second needs the corpus.

---

# Follow-up: where the fault is, and what it will cost

Four candidate causes were tested. Three are ruled out, each in **both**
directions, which is what makes them ruled out rather than merely unsupported.

### 1. Intake is not washing the style away — the information survives it

Style features measured on exactly the images the encoder is fed:

| | stroke px | slant° | x-height/cap | aspect w/h | ink density |
|---|---|---|---|---|---|
| writer 2 | 1.154 | −3.6 | 0.546 | 0.843 | 0.303 |
| writer 3 | 1.066 | −6.7 | **0.721** | 0.742 | 0.183 |
| segoepr | 2.131 | −12.9 | 0.718 | 0.825 | 0.340 |
| times | 2.564 | −2.9 | 0.690 | 0.805 | 0.344 |

Writers 2 and 3 differ in measurable style by **3.31** (z-scored across all
seven styles) — *more* than segoepr vs times (2.56), times vs Gabriola (2.00)
or segoepr vs Gabriola (2.88). Their x-height-to-cap ratio differs by a third.
The encoder nonetheless rates them 0.996 while rating segoepr vs times 0.750.
**The style is in the images. The encoder discards it.** Across all 21 pairs,
correlation between measurable style distance and encoder cosine is only −0.293.

### 2. Not stroke weight, and not edge appearance

| condition | soft-edge fraction | mean pairwise cos |
|---|---|---|
| hands, as intake delivers them | 0.573 | 0.982 |
| **hands with edges hardened to render-like** | 0.217 | **0.981** |
| fonts, as rendered | 0.231 | 0.725 |
| **fonts blurred to photo-like** | 0.547 | **0.720** |

A double dissociation: making hands look like renders does not spread them, and
making fonts look like photographs does not collapse them. Together with the
weight control above (dilation moved cosine 0.982 → 0.970), every low-level
appearance explanation is exhausted. What is left is the encoder's
representation of shape.

### 3. No inference-time fix exists

Amplifying each hand's deviation from a generic style anchor (the mean style
vector of eight training-distribution fonts), at inference only:

| α | writer 1 | writer 2 | writer 3 | mean | writer 2 ≈ writer 3 |
|---|---|---|---|---|---|
| 1.00 (shipped) | 0.233 | 0.331 | 0.152 | 0.239 | 0.943 |
| 1.25 | 0.240 | 0.392 | 0.133 | **0.255** | 0.890 |
| 1.50 | 0.249 | 0.379 | 0.088 | 0.238 | 0.846 |
| 2.00 | 0.234 | 0.238 | 0.033 | 0.168 | 0.625 |
| 3.00 | 0.138 | 0.034 | 0.000 | 0.057 | 0.306 |

This is a useful negative result, not just a failed idea. The style code is
**not empty** — amplifying it separates the two writers monotonically, exactly
as it should if the direction carried real style. But the decoder only behaves
in a narrow neighbourhood of the training style distribution: push the code
further out and the letters disintegrate rather than becoming more personal.
α=1.25 buys +0.016 mean, and is still **rejected** — writer 3 loses 0.019,
which fails this project's own standing bar of 0.01.

### Conclusion

The defect is in what the style encoder learned, not in how it is fed or how it
is used. It cannot be fixed without training.

### The measurement that should gate any training run

This is the part that was missing before. `scripts/style_collapse.py` reports
cross-style output similarity — generate the same unwritten letters from two
different people's samples and compare the results. Every previous number in
this project is computed within a single style and would be unchanged by a model
that ignored style entirely. Any future run should be gated on **both**: the
per-writer leave-one-out score must not fall, and writer 2 ≈ writer 3 must come
down from 0.943.

---

# The precise defect, and why no inference-time fix exists

### It is not out-of-distribution collapse toward the mean

The obvious reading — the encoder returns its average output for inputs it does
not understand — is wrong. Photographed hands sit **farther** from the font
centroid than fonts do (2.26 vs 2.00) with **1.27×** the magnitude. These are
strong, confident codes, not a shrug.

### It is one shared axis

Measuring how parallel each style's *deviation* from the font centroid is:

| | mean pairwise cos of deviations |
|---|---|
| the ten fonts | **−0.103** (range −0.790 to +0.691) |
| the three hands | **+0.928** |
| writer 2 vs writer 3 alone | **+0.983** |

Different fonts deviate in unrelated directions, which is what a working style
space looks like. Every photographed hand deviates in very nearly the *same*
direction. So a hand's style code is approximately

    centroid  +  (a lot) × one shared axis  +  (very little) × who wrote it

The encoder responds far more strongly to "this is real handwriting" than to
"this is Antoine's handwriting". That single axis is the defect.

### The axis cannot simply be removed

Estimating the axis leave-one-writer-out (from the other two only) and
projecting it out of the held-out writer's code:

| held-out | β=0 (shipped) | β=0.5 | β=0.8 | β=1.0 |
|---|---|---|---|---|
| writer 1 | 0.233 | 0.230 | 0.254 | 0.257 |
| writer 2 | 0.331 | 0.236 | 0.189 | 0.161 |
| writer 3 | 0.152 | 0.109 | 0.069 | 0.059 |
| **mean** | **0.239** | 0.192 | 0.171 | 0.159 |

It gets worse, badly, for the two writers it was supposed to help. The shared
axis is not removable junk sitting on top of a good signal — the decoder needs
it to draw anything at all. Only writer 1, the least collapsed of the three,
gains.

### Five interventions, five dead ends

| intervention | result |
|---|---|
| blame intake | ruled out — the style survives intake, measurably |
| match stroke weight | ruled out in both directions |
| match edge appearance | ruled out in both directions |
| amplify deviation from the generic style | separates writers, destroys quality |
| project out the shared axis | worse for 2 of 3 writers |

**There is no fix that does not involve training.**

### What the diagnosis implies about *what* to train

The shared axis is not caused by how photographs *look* — blur and weight were
both eliminated. What is left is what real handwriting *is*: stroke weight that
varies within a single letter, baselines that wobble, the same letter shaped
differently each time. The training corpus contains handwriting-*styled* fonts,
but a designed handwriting font is regular in every way a real hand is not. The
encoder has therefore never been asked to tell two real hands apart, and it
cannot.

That points at the training data as much as the architecture, which changes the
ranking established in REVIEW.md and should be settled before any GPU time is
spent.

---

# What the axis actually is: glyph size

The irregularity hypothesis was tested **before** spending GPU time on it, and
**failed**. Synthesising handwriting-like irregularity onto fonts (elastic warp,
within-stroke weight variation, baseline wobble, lean) moved them to −0.10 on
the hand axis at every strength tried. The real hands are at **+2.22**. Wrong
direction, negligible magnitude. `src/hfont/data/augment.py` implements it and
is kept only because the pre-flight that rejected it is worth reproducing.

Probing one transform at a time against the same axis found the real one:

| transform applied to fonts | projection onto hand axis |
|---|---|
| **shrunk to 50%** | **+2.05** |
| shrunk to 70% | +1.21 |
| eroded 1px | +0.77 |
| unmodified | −0.00 |
| grey interior, noise, per-glyph jitter | ±0.14 |
| dilated 1px / enlarged 130% / dilated 2px | −0.82 / −1.42 / −1.43 |

The real hands sit at +2.22; shrinking a font to half size puts it at +2.05.
Across the ten fonts, **corr(x-height, projection) = −0.801, p = 0.005**.

**The "photograph axis" is a size axis.** Measured on the canvas the model is
actually fed: hands arrive at mean x-height **21.0px**, fonts at **38.9px** —
0.54×. All three hands are equally undersized, which is why they cluster, and
their strokes are thin (1.07–1.31px vs 1.74–2.56px) in the same proportion. The
thin-stroke finding in REVIEW.md and the export cliff are downstream of this,
not independent problems.

### Why, and it is not a coding bug

`frame_from_boxes` sets `ascent = max(...)` over the frame characters, so the
canvas must fit the single most extreme letter. Real hands sprawl more than
typefaces: (ascent+descent)/x-height is **3.40** for the three writers against
**2.60** for eight fonts, and the letter driving it is a capital `A` or a `g`
descender. That predicts 0.76× on its own; the measured 0.54× means sprawl is
most but not all of it. The framing code is correct and the `SEED_24` guard at
`intake.py:94` works — the rule itself is what disadvantages handwriting.

### A partial free fix, with its cost stated

Re-framing from the original photograph (resampling at full resolution, not
upsampling the 128px image):

| framing | x-height | writer 1 | writer 2 | writer 3 | mean | letters touching the canvas edge |
|---|---|---|---|---|---|---|
| shipped (max, margin 0.06) | 21.0 | 0.233 | 0.331 | 0.152 | 0.239 | 0 of 90 |
| 95th percentile | 24.8 | 0.232 | 0.342 | 0.163 | 0.246 | 2 |
| 90th percentile | 41.6 | 0.232 | 0.355 | 0.184 | 0.257 | 11 |
| 80th percentile | 51.0 | 0.288 | 0.350 | 0.204 | 0.281 | 19 |
| margin −0.30 | 55.1 | 0.326 | 0.359 | 0.237 | **0.307** | 27 |

**The control rules out metric inflation, in the right direction.** Bigger
glyphs could score better for free. Re-framing the *fonts* the same way makes
them score **−0.038 worse** on average, while the hands score **+0.068 better**.
The gain is real and runs opposite to the artifact.

But every setting that captures most of the gain clips letters, and the written
letters are passed through unchanged into the font, so a clipped `f` is a
visible defect. This is a genuine trade-off, not a free win: 95th percentile is
nearly free (+0.007, 2 letters) and nearly worthless.

### The corrected training plan

Scale augmentation, not irregularity augmentation. It passes the same pre-flight
that rejected the first plan: shrinking a font to 50% lands it at +2.05, inside
the hand region, so training with reference and target scale varied over that
range would cover exactly the part of style space real hands occupy and
currently sit outside of. Supporting evidence from the control run: the model
already scores worse on small-x-height fonts (times, 54.6px → LOO 0.836;
BRADHITC, 28.6px → 0.226), so this is visible inside the training distribution
too, not only on photographs.

---

# The scale-augmentation experiment: it failed

Trained on Colab, 23 Sep 2026. Two arms from identical init (seed 1234 set
before either model was built), the same freshly-rendered corpus (3031 fonts,
2000 families, 227,325 glyphs), the same 10,000 steps, the same losses. The
**only** difference between them is `DatasetConfig.scale_jitter`.

## The control reproduces the defect, so the experiment is valid

Before reading the treatment, check the control against the shipped run-3 model
it is standing in for. Run 3 had 40K cumulative steps on a different render of
the corpus; the control had 10K. They land in nearly the same place:

| leave-one-out, clDice | shipped run 3 | control (10K steps) |
|---|---|---|
| writer 1 | 0.233 | 0.231 |
| writer 2 | 0.331 | 0.353 |
| writer 3 | 0.152 | 0.139 |
| **writer 2 ≈ writer 3 output** | **0.943** | **0.897** |
| style-vector cosine, w2~w3 | 0.996 | 0.991 |

The control reproduces both the quality and the collapse. Whatever the treatment
shows is therefore attributable to the augmentation.

## The result

| | control (jitter 0) | treatment (jitter 1.0) | change |
|---|---|---|---|
| val IoU (best) | **0.4917** | 0.4503 | −0.041 |
| LOO clDice, writer 1 | 0.231 | 0.227 | −0.004 |
| LOO clDice, writer 2 | 0.353 | 0.283 | **−0.070** |
| LOO clDice, writer 3 | 0.139 | 0.086 | **−0.053** |
| LOO clDice, mean | 0.241 | 0.199 | **−0.042** |
| writer 2 ≈ writer 3 output | 0.897 | 0.804 | −0.093 |
| **style cosine, w2~w3** | 0.991 | **0.9998** | **+0.009** |

**Gate: failed.** The standing bar is that no writer may lose more than 0.01;
writer 2 lost 0.070 and writer 3 lost 0.053.

## Why the one apparently-good number is not good

Cross-style output similarity did fall, 0.897 → 0.804, which is the number the
gate was written around. It should not be read as success, for two reasons.

**The style vectors collapsed further, not less.** Writers 2 and 3 went from
cosine 0.991 to **0.9998** — indistinguishable. The augmentation was supposed to
force the encoder to separate hands; it did the exact opposite, and that is the
quantity the intervention was aimed at.

**The remaining divergence is inconsistent across pairs**, which is what noise
looks like and what a real effect does not:

| pair | control | treatment | |
|---|---|---|---|
| writer 1 ≈ writer 2 | 0.509 | 0.640 | *more* similar |
| writer 1 ≈ writer 3 | 0.607 | 0.449 | less similar |
| writer 2 ≈ writer 3 | 0.897 | 0.804 | less similar |

A model that had learned to tell hands apart would separate all three pairs. One
pair moving the wrong way, while the style codes become *more* identical, is
better explained by the treatment model simply being worse: its outputs differ
from each other because they are noisier, not because they are more personal.

## What this does and does not establish

**Established:** at equal step budget, scale augmentation costs 0.041 val IoU
and 0.042 mean LOO, and does not separate the style codes. As specified, it is
not the fix.

**Not established:** that scale coverage is irrelevant. The honest caveat is
that jitter makes the training task strictly harder — visible from step 200,
where the treatment's l1 was 0.514 against the control's 0.385 — so 10,000 steps
may simply be too few for it to repay its cost. This experiment shows it loses
at 10K steps, not that it loses at 40K. Testing that costs 4x the GPU time and
should not be spent until there is a reason beyond hope.

**What survives from the diagnosis:** the size finding itself is a measurement,
not a hypothesis, and it stands — hands arrive at 0.54x the size of fonts, and
corr(x-height, position on the collapse axis) = −0.80. What has been refuted is
the inference that covering that range during training would fix the collapse.
Two plausible readings remain, and this experiment cannot separate them: the
encoder may need an architectural change to represent style locally (P4), or
size may be a symptom of the collapse rather than its cause.
