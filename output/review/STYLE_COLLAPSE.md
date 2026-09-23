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
