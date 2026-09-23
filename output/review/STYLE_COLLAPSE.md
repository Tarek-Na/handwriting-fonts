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
