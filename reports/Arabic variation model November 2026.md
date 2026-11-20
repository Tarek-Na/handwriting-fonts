# Arabic handwriting variation model (steps A5–A6, 16 November 2026)

The course asks for a novel deep-learning idea. Building a font from a full
sheet needs no network (steps A1–A2 do it), so the question became what a
network can add. This report covers the method, the evidence that led to it,
and the results on held-out writers. Every number is from AHAWP (82 writers,
65 letter forms, 10 copies each; CC BY 4.0), split by writer: 49 train, 17
validation, 16 test. Choices were made on validation writers. The test writers
were scored once, at the end.

## 1. Representation: template skeleton plus a writer's moves (A5)

* For each of the 65 forms, the **template** is the pen path (centreline) of
  the training writer whose skeleton is nearest all the others: the medoid,
  by symmetric chamfer distance.
* A written copy is described as the template's points **moved** onto the
  copy's skeleton. This uses coherent point drift (Myronenko & Song 2010): an
  affine pass, then smooth local moves (β = 4 px, λ = 0.5, 1% outliers). It
  needs no point correspondence given in advance. Code:
  `src/hfont/skeleton.py`, `scripts/skeleton_moves.py`.
* Each form's moves are reduced to 24 principal components. Eight already
  hold 99.5% of the variance.

How much the representation can hold (val writers, joined forms, aligned
centreline F1 against the writer's typical copy):

| Drawn | F1 |
|---|---|
| writer's own second copy | 0.666 |
| template + moves fitted to that copy | 0.689 |
| template + moves fitted to the *second* copy | 0.551 |
| template + affine fit only | 0.619 |
| template alone, writer's pen | 0.486 |
| copy the nearest training writer (earlier best) | 0.435 |

## 2. Why one exact letter cannot be predicted

* **From isolated letters to joined forms.** Ridge regression from the
  isolated letters' fits to each joined form's affine fit reaches 0.496.
  Adding local shape components gives 0.480, which is worse. Neither beats
  the template alone (0.486) by more than noise.
* **Copy-to-copy wobble.** A form's size varies about as much between one
  writer's two copies (r = 0.28) as other letters can predict it (r = 0.37).
* **Donor choice.** Choosing a donor writer from the 49 by the writer's
  isolated letters gives 0.441, against 0.382 for a random donor. Choosing by
  all their other joined forms gives 0.469. The best donor would give 0.742,
  but nothing the writer hands in finds it.
* **What is stable.** Overall style is: averaged over letters, isolated and
  joined size correlate at r = 0.78.

**Conclusion:** a writer's letters vary greatly every time they are written,
around a stable overall style. So the model's job is a *distribution* of
versions in that style, and the font should carry several per letter.

## 3. Model (A6)

A conditional VAE over a form's 24 components (`src/hfont/models/variation.py`,
`scripts/variation.py`), with four parts:

* **Style encoder.** It reads the writer's isolated letters (one copy each, as
  on a sheet), embedding each with its letter identity. Attention pooling
  gives a 32-d style vector. During training, 30% of letters are hidden at
  random (letter dropout).
* **Prior.** A conditional prior p(z | form, style), with an 8-d latent.
* **Encoder.** q(z | x, form, style).
* **Decoder.** A Gaussian over x with **learned per-component variance**.

The loss is the Gaussian negative log-likelihood plus KL(q‖prior). Training
uses all 10 copies of every form of the 49 training writers (32,000 examples)
and takes 20,000 steps on CPU.

A fixed-variance decoder collapsed: every sampled version came out the same
(spread 0.01 against 0.3 in the data). A unit test catches this.

## 4. Evaluation

For each joined form, a method draws 5 versions. These are compared with the
writer's real copies (the even-numbered copies; the odd ones are the
ceiling's versions) by aligned centreline F1. There are three measures:

* **Realism:** each version's best match among the real copies.
* **Coverage:** each real copy's best match among the versions.
* **Writer specificity:** realism against this writer's copies minus realism
  against another writer's copies. Only a method that knows the writer can
  score above zero.

Comparisons are paired over writers, with 95% bootstrap intervals.

### Validation (17 writers, 765 forms, temperature 0.4)

| Method | Realism | Coverage | Specificity |
|---|---|---|---|
| own other copies (ceiling) | 0.726 | 0.706 | +0.204 |
| cVAE + style, seeds 0/1/2 | 0.632 / 0.634 / 0.626 | 0.539 / 0.538 / 0.540 | +0.027 / +0.032 / +0.029 |
| cVAE + style, letter dropout | 0.634 | 0.539 | +0.033 |
| cVAE, no style, seeds 0/1/2 | 0.612 / 0.612 / 0.610 | 0.535 / 0.534 / 0.533 | +0.001 / 0.000 / 0.000 |
| Gaussian shape model (no NN), T 0.4 | 0.600 | 0.531 | +0.001 |
| Gaussian + ridge from the writer's letters (no NN), T 0.4 / 0.7 / 1.0 | 0.612 / 0.557 / 0.499 | 0.531 / 0.531 / 0.506 | +0.007 / +0.007 / +0.006 |
| another writer's real copies | 0.523 | 0.491 | +0.007 |
| template only (1 version) | 0.645 | 0.451 | +0.001 |

### Test (16 writers, 712 forms, scored once)

| Method | Realism | Coverage | Specificity |
|---|---|---|---|
| own other copies (ceiling) | 0.713 | 0.692 | +0.208 |
| **cVAE + style (final)** | **0.605** | **0.518** | **+0.017** |
| cVAE, no style | 0.592 | 0.515 | +0.005 |
| Gaussian + ridge from the writer's letters, T 0.4 | 0.583 | 0.506 | −0.000 |
| Gaussian shape model | 0.572 | 0.506 | −0.004 |
| another writer's real copies | 0.503 | 0.471 | +0.002 |
| template only (1 version) | 0.616 | 0.429 | −0.002 |

Paired over the 16 test writers:

| Comparison | Gain | 95% CI | Writers better |
|---|---|---|---|
| realism vs Gaussian | +0.033 | [+0.026, +0.040] | 16/16 |
| coverage vs Gaussian | +0.012 | [+0.004, +0.021] | 12/16 |
| specificity vs Gaussian | +0.021 | [+0.010, +0.035] | 13/16 |
| realism vs no style | +0.013 | [+0.006, +0.021] | 11/16 |
| specificity vs no style | +0.012 | [+0.001, +0.025] | 10/16 |
| realism vs Gaussian + ridge | +0.021 | [+0.016, +0.028] | 15/16 |
| coverage vs Gaussian + ridge | +0.012 | [+0.005, +0.019] | 12/16 |
| specificity vs Gaussian + ridge | +0.017 | [+0.008, +0.029] | 14/16 |

**The fair non-network baseline.** The plain Gaussian ignores the writer, so
it cannot show whether *reading the writer* needs a network. The second
baseline does read the writer:

* For each joined form, ridge regression predicts the writer's mean components
  from their 18 isolated letters (copy 0, as on a sheet).
* Versions are drawn from a full Gaussian of the training copies around that
  prediction.
* The ridge strength is set by leave-one-writer-out on the training writers.
* The temperature was chosen on val (0.4 was best), then scored once on test.

On test it gains a little over the plain Gaussian: realism +0.011, 16/16
writers. But its writer specificity is zero (−0.000), and the network beats it
on all three measures. So a linear read of the letters does not carry a
writer's style into the joined forms; the network's style encoder does.

## 5. What can and cannot be claimed

* **The neural model beats the classical statistical shape model,** with or
  without the writer's letters. On the same template representation, its
  versions are more realistic than the plain Gaussian's for every test writer
  and than the ridge-conditioned Gaussian's for 15 of 16. They also cover the
  writer's range better than both.
* **The style encoder carries writer identity into joined forms.** Without
  it, specificity is zero on every seed. With it, specificity is reliably
  positive, but small: +0.012 to +0.03 against a ceiling of +0.2. The effect
  shrank from validation to test.
* **Limits.** It is far from the writer's own copies. It does not reproduce
  style that lives outside the template, such as a writer's long connecting
  strokes. AHAWP's 49 training writers bound how much style can be learned.

## 5b. Extension: several templates per form (A6b, 17 November)

With one template per form, the model could not produce a letter built
differently from that template. Two examples are a writer's three-tooth
medial seen and a closed-loop medial ain.

**Templates.** Each form now has up to 3 templates: k-medoids of the training
writers' skeletons, with clusters under 6 writers dropped (191 pairs in all).

**Registration.** Every copy is registered to the template it fits best
after an affine fit.

**Model.** A choice head, p(template | form, style), picks the template, and
each version draws its own template (`scripts/variation_mix.py`).

### Validation (17 writers)

| Method | Realism | Coverage | Specificity |
|---|---|---|---|
| single template + style (A6) | 0.634 | 0.539 | +0.033 |
| 3 templates + style | 0.626 | 0.555 | +0.027 |
| 3 templates + style, top template only | 0.631 | 0.536 | +0.028 |
| 3 templates, no style | 0.610 | 0.549 | −0.000 |
| 3-template Gaussian (no NN) | 0.595 | 0.545 | −0.002 |

### Test (16 writers, scored once)

| Method | Realism | Coverage | Specificity |
|---|---|---|---|
| 3 templates + style | 0.593 | 0.532 | +0.006 |
| 3-template Gaussian (no NN) | 0.572 | 0.522 | −0.002 |

For comparison, the single-template model on test: 0.605 / 0.518 / +0.017.

Paired over the 16 test writers:

| 3 templates + style, compared with | Realism | Coverage |
|---|---|---|
| single template | −0.012 [−0.021, −0.003], 5/16 | +0.014 [+0.008, +0.020], 14/16 |
| 3-template Gaussian | +0.021 [+0.014, +0.028], 15/16 | +0.010 [+0.004, +0.016], 13/16 |

**Reading.** More templates trade a little realism for coverage. The versions
now include the different ways people build a letter: the three-tooth seen
and the closed ain appear among the samples. Against its fair non-network
baseline, the network still wins on both measures.

**What did not work.** The choice head barely beats the training frequencies:
validation cross entropy is 0.90 with style and 0.91 without. So the model
does not pick *this writer's* structure; for one writer's medial ain it drew
the closed loop in 2 of 6 versions. This is the same data limit as in §2,
now seen at the level of letter structure. The single-template model stays
the primary result; the multi-template model is the option when variety
matters more than closeness.

## 5c. Training changes against the 49-writer limit (17 November)

Two changes were each trained in full (20,000 steps), alone and together,
and scored on the val writers. The current model (letter dropout, one
template) is the reference.

* **Synthetic writers (augmentation).** One random width, height and slant
  change (spread 0.1) is applied to all of a training writer's letters per
  example, done exactly in component space.
* **Writer-contrastive loss.** InfoNCE between the style vectors of two
  random draws of the same writer's isolated letters, weight 0.5. Other
  examples of the same real writer are left out of the negatives.

| Variant | Val loss | Realism | Coverage | Specificity |
|---|---|---|---|---|
| current model | 9.757 | 0.634 | 0.539 | +0.033 |
| synthetic writers | 9.665 | 0.630 | 0.539 | +0.040 |
| contrastive | 10.099 | 0.622 | 0.534 | +0.011 |
| both | 9.767 | 0.629 | 0.541 | +0.030 |

Paired against the current model (val writers, 95% CI):

| Variant | Realism | Specificity |
|---|---|---|
| synthetic writers | −0.005 [−0.009, −0.002] | +0.007 [−0.000, +0.013] |
| contrastive | −0.013 [−0.023, −0.004] | −0.022 [−0.038, −0.007] |

**Synthetic writers** trade a little realism for borderline specificity.

**The contrastive loss makes the style vector worse for generation.**
Telling writers apart is not the same as knowing how a writer forms the
letters the generator must draw: what identifies a person, such as their pen
and overall size, the generator cannot use.

The rule set before the runs was to keep a change only if it clearly
improves on val. Neither does, so the current model stays final, and the
variants were not scored on test.

Together with §2 and §5b, this points to the data limit. With 49 training
writers, and isolated letters that reveal little about joined forms, neither
more training nor these two training changes beat the simple model.

## 5d. Robustness checks (17 November)

Every check below is on the 17 val writers, paired over writers with 95%
bootstrap intervals; the test writers were not used. The final model
(`runs/v1_drop`), scored again, reproduces 0.634 / 0.539 / +0.033 exactly.

**Wrong-writer control.** The model is given the *next* writer's isolated
letters, the writer that specificity compares against.

| Style from | Realism | Coverage | Specificity |
|---|---|---|---|
| the writer's own letters | 0.634 | 0.539 | +0.033 |
| another writer's letters | 0.603 | 0.516 | −0.031 |
| no style (other model) | 0.612 | 0.535 | +0.001 |

Specificity flips sign: the versions now match the other writer. Own letters
against another's: specificity +0.064 [+0.030, +0.105], better for all 17
writers. The style encoder reads who wrote the letters, and the specificity
measure is not an artefact.

**How many letters the model needs.** It is shown a random subset of each
writer's 18 isolated letters.

| Letters given | 1 | 3 | 6 | 12 | 18 |
|---|---|---|---|---|---|
| Realism | 0.573 | 0.621 | 0.624 | 0.630 | 0.634 |
| Coverage | 0.498 | 0.527 | 0.533 | 0.537 | 0.539 |
| Specificity | −0.005 | +0.029 | +0.031 | +0.029 | +0.033 |

* **One letter is worse than no style:** realism −0.039 [−0.057, −0.020]
  against the no-style model. Training hides only 30% of letters, so a single
  letter is outside what the encoder saw.
* **Three letters already carry the writer:** specificity +0.029 [+0.001,
  +0.059] against no style. Going from 3 to 18 letters adds coverage
  (+0.012 [+0.008, +0.017]) but no specificity.
* **Consequence:** `photo_font` refuses fewer than 3 readable letters
  (`MIN_LETTERS`).

**Which copy is handed in.** The letters are taken from copies 1–4 instead of
copy 0. Realism ranges 0.636–0.643 and specificity +0.030 to +0.034, so the
result does not hinge on one sheet.

**Metric tolerance.** Rescored at 2 and 5 px instead of 3:

| Tolerance | Gaussian | Gaussian + ridge | no style | **model** |
|---|---|---|---|---|
| 2 px, realism / specificity | 0.507 / +0.002 | 0.518 / +0.004 | 0.518 / +0.002 | **0.539 / +0.030** |
| 5 px, realism / specificity | 0.733 / +0.000 | 0.746 / +0.007 | 0.741 / −0.002 | **0.763 / +0.030** |

Against Gaussian + ridge, the model gains realism +0.021 [+0.007, +0.037] at
2 px and +0.017 [+0.006, +0.030] at 5 px, and specificity +0.026 and +0.024,
both with intervals above zero. The ordering does not depend on the
tolerance.

**Temperature and number of versions.** Temperature trades realism for
coverage, for every method.

| Temperature | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| model, realism / coverage | 0.653 / 0.514 | 0.645 / 0.529 | 0.634 / 0.539 | 0.621 / 0.545 | 0.587 / 0.544 | 0.531 / 0.526 |
| Gaussian + ridge | 0.639 / 0.507 | 0.628 / 0.521 | 0.612 / 0.531 | — | 0.557 / 0.531 | 0.499 / 0.506 |

* **The model is ahead at every temperature tried.** Its specificity stays
  +0.022 to +0.033.
* **Versions per form.** With 3 versions it scores 0.632 / 0.514 against
  0.614 / 0.502 for Gaussian + ridge; with 10, 0.632 / 0.569 against
  0.612 / 0.565.

**Training writers (the data curve).** The model was retrained on random
subsets of the training writers, 2 seeds each. Templates and components
still come from all 49.

| Training writers | 12 | 25 | 49 |
|---|---|---|---|
| Best val loss | 10.53 / 10.56 | 10.23 / 10.22 | 9.76 |
| Realism | 0.612 / 0.611 | 0.619 / 0.629 | 0.634 |
| Coverage | 0.523 / 0.529 | 0.535 / 0.534 | 0.539 |
| Specificity | +0.008 / +0.008 | +0.018 / +0.022 | +0.033 |

* **Specificity grows with every step:** 49 against 12 writers gives +0.025
  [+0.009, +0.043], and 49 against 25 gives +0.011 [+0.001, +0.021].
* **At 12 writers, the model is no better than the no-style model** (0.612).
* **The curve has not flattened at 49.** This is the direct evidence for §2's
  data limit: the style signal is limited by the number of writers, and more
  writers would raise it.

## 5e. Stroke ends: the fits were too narrow (18 November)

**How it was found.** A sentence set in a test writer's real handwriting
was much wider than the same sentence set by the model. The writer's long
medial seen accounts for part of it: their copy was 1.7× the typical width,
against about 1.1× on most of their other copies. The rest is a bias in the
representation:

* the fitted joined forms were **0.89–0.92 of the real copies' width** for
  every writer;
* the worst forms were kaf and seen, at 0.74–0.80;
* `register` keeps moves smooth, so a template stroke stops short of a long
  tail or connecting stroke.

The font itself copies each scan faithfully.

**Fix.** `skeleton.extend_ends` carries each fitted stroke end out along the
writer's stroke to where it ends, keeping point count, order and even
spacing. On 540 val copies:

* width: 0.92 → 1.00;
* the fit's F1 to its own copy: 0.637 → 0.723 (medial kaf 0.611 → 0.847).

**Pipeline.** With `HFONT_VARIATION=ends`, every step uses its own files;
the original stays as tag `a6-final-v1`. The first retrain chose an
undertrained checkpoint (step 6,000): a few copies whose fit had failed,
such as a mangled scan or an alef with a stray line under it, made the val
KL explode. So both representations were retrained with checkpoints chosen
by the mean of the best 99% of val examples (`--select trimmed`), seeds 0
and 1.

| Val, paired, fixed minus original | Realism | Coverage | Specificity |
|---|---|---|---|
| seed 0 | −0.007 [−0.015, +0.001] | +0.008 [+0.000, +0.016] | +0.010 [−0.005, +0.026] |
| seed 1 | −0.008 [−0.013, −0.002] | +0.006 [−0.000, +0.013] | +0.006 [−0.008, +0.021] |

* **Width.** The model's letters, as a share of the writer's real width,
  rise from a median of 0.94 (initial kaf 0.68) to 1.00 (narrowest form
  0.81).
* **Against its fair baseline,** on the same representation, the fixed
  model still wins: realism +0.027 [+0.008, +0.049], specificity +0.031
  [+0.012, +0.051].
* **Verdict on the letter-level measures:** a trade of realism for coverage,
  not a clear gain. By the rule set beforehand, it does not replace the final
  model on these alone.
* **Whole words decide it.** Width and connecting strokes matter most there,
  and those measures are blind to them (see the word-level evaluation).

## 5f. Whole words (19 November)

`scripts/word_eval.py` sets 15 test words for each val writer:

* **Reference:** fonts made from the writer's full sheet, one per
  even-numbered copy.
* **Each method:** a font made from their isolated letters only.
* **Width error:** |log(word width / median real width)|.
* **Shape:** centreline F1 after stretching the word to the real width.
* **Ceiling:** fonts from the writer's odd-numbered copies.

| Val, 17 writers × 15 words | Width error | Width vs writer's | Shape |
|---|---|---|---|
| writer's own other copies | 0.336 | ×1.13 | 0.535 |
| template only | 0.373 | ×1.00 | 0.548 |
| final model (A6) | 0.387 | ×0.94 | 0.551 |
| original, seeds 0 / 1 (trimmed) | 0.385 / 0.397 | ×0.92 / ×0.94 | 0.531 / 0.531 |
| stroke ends, seeds 0 / 1 | 0.373 / 0.367 | ×1.00 / ×1.03 | 0.539 / 0.543 |

* **The measure itself fails its check.** The writer's own other copies do
  not beat the template: shape −0.013 [−0.047, +0.020], width error 0.037
  lower [−0.067, +0.135]. At word length, how one person writes a word
  varies from one copy to the next more than writers differ, so this
  measure cannot rank methods. It is the copy-to-copy variation of §2,
  seen at word level.
* **The stroke-end fix** is better on both seeds and both measures. It
  gives width error 0.011 and 0.031 lower, and shape +0.008 and +0.012.
  But every interval includes zero.
* **Decision:** by the rule set beforehand, the final model stays A6
  (`a6-final-v1`, `runs/v1_drop`). The stroke-end representation is kept as
  an option (`HFONT_VARIATION=ends`): its words run as wide as the
  writer's (×1.00 against ×0.92).
* **Claims stay at letter level.** There the network beats both Gaussian
  baselines on held-out writers. At word level, neither it nor the template
  can be told apart from the writer's own repeats.

## 5g. Stress test of the short-sheet pipeline (19 November)

`scripts/stress_test.py` writes each val writer's real isolated letters into
the 20-box sheet with a change, simulates the photo, reads it back, and lets
the model draw every joined form. The drawings are scaled back by the size
change and scored against the writer's real copies.

| Change (17 val writers) | Realism vs unchanged, paired |
|---|---|
| harsh photo (strong tilt, dim uneven light, shadow, blur, JPEG 50) | +0.000 [−0.002, +0.002] |
| faint pale-blue ink | +0.001 [−0.001, +0.003] |
| thinner / thicker pen | +0.001 / −0.002 |
| letters ×1.25 | −0.001 [−0.008, +0.006] |
| letters ×0.8 | −0.016 [−0.023, −0.008] |
| letters ×0.6 | −0.060 [−0.079, −0.041] |
| letters ×1.5 (7 letters a sheet overflow their box) | −0.020 [−0.039, −0.003] |

Every box was read in every condition. Only size mattered.

**Fix.** Letters whose median height lies outside the training writers'
range (49–112 px, 1st–99th percentile) are rescaled into it before the
model reads them, and its drawings are scaled back (`variation.photo_style`,
`SIZE_RANGE`). Sheets inside the range are untouched. With the fix:

* ×0.6: −0.028 [−0.039, −0.019] (13 of 17 sheets rescaled);
* ×1.5: −0.010 [−0.022, +0.003];
* ×0.8: −0.013 (mostly in range, so mostly unchanged).

The demo (`scripts/demo_app.py`) also tells the writer when:

* letters touch their box's edge and may be cut off;
* the writing was small and has been enlarged;
* fewer than 15 letters were read.

## 6. In a font

Glyphs may now have alternates (`GlyphSpec.variant`, named `<glyph>.altN`).
They are kept out of the cmap and cycled by `calt`, so each letter shows the
version after its neighbour's. `rand` was tried first and HarfBuzz did not
apply it.

`python scripts/variation.py preview WRITER --out output/ahawp/<name>.png`
sets the same words in three fonts:

1. the writer's real full sheet;
2. their 18 isolated letters with template skeletons;
3. the same letters with 3 model versions per joined form.

The preview is an image of real handwriting. It stays in `output/` and is
never committed.

**Lam-alef in final position.** Arabic always joins lam and alef into one
ligature. The font's rule for the final position (after a joining letter,
as in سلام) had no glyph to use, because a sheet gives only the isolated
ligature, so سلام was set as a separate lam and alef.

* The final ligature is now made from the isolated one written
  (`ahawp_font.with_final_lam_alef`).
* It gets a connecting stroke at its base. Joined forms take their
  connection at their rightmost point, and a writer's alef that leans out
  to the right would otherwise be joined half way up.

**The live demo's medial ain and kaf.** The first real writer found the
demo's medial ain and medial kaf hard to read. They are drawn from each
form's template, the medoid training writer's construction:

* medial ain: a crossed loop with a spur (21 of 49 training writers);
* medial kaf: an S-shape.

The demo now draws these two forms from the multi-template model's
alternative templates (§5b), still in the writer's style: a closed-loop
ain (1 of 49 writers) and an upright kaf with a hook (15 of 49). This is
a product choice for legibility, not a data-driven one
(`variation.CONSTRUCTIONS`). The evaluation and every number in this
report use the single-template model.

## 7. Commits

51d5e94, 60db8f4 (A5 and its diagnostics), ed4e197 (cVAE), 216ac48 (letter
dropout), 0a1e2fe (font alternates), f5a6572 (fonts from versions, test
score).
