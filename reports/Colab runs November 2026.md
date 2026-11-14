# Colab runs, November 2026: rebuilt baseline and G3

Run on Colab Pro with `notebooks/colab_train.ipynb` (commit e616b9e). The
laptop's hw40k baseline was lost, so it was rebuilt: the recommended recipe,
16K steps, `handwriting_weight=6.0`, Google Fonts cache only. G3 was then
warm-started from the rebuilt baseline's `best.pt` and trained 10K steps on the
same cache plus SD19's writers (`sd19_weight=1.0`).

All five models were scored in one session by the same cell, so the rows are
comparable with each other. The nearest-font floor here retrieves from the
Google Fonts cache (0.609), which is not the 0.576 / 0.590 of earlier machines.

## Results

| Model | LOO clDice (10 photographed writers) | Alike across writers | Attributed | Floor: model vs nearest font (2,511 SD19 letters) | Model closer |
|---|---|---|---|---|---|
| shipped (phase1c 30K) | 0.373 | 0.768 | 13.7% | 0.598 vs 0.609 | 44% |
| base_best | 0.368 | 0.599 | 16.0% | 0.599 vs 0.609 | 46% |
| base_last | 0.364 | 0.554 | 16.7% | 0.586 vs 0.609 | 42% |
| **g3_best** | **0.430** | 0.800 | 14.0% | **0.672** vs 0.609 | **63%** |
| g3_last | 0.392 | 0.690 | 13.0% | 0.632 vs 0.609 | 54% |

Pass thresholds (instruction.md §4): clDice ≥ 0.36, alike ≤ 0.67 (real
writers: 0.287), attributed ≥ 25% (chance: 10%).

## What it shows

1. **Real handwriting makes much better letters.** G3's LOO clDice rose from
   0.368 to 0.430 on the ten photographed writers, the largest gain the
   project has measured. Those writers were never trained on, so this is a
   held-out result. For the first time, the model clearly beats the nearest
   font on SD19's held-out writers (0.672 vs 0.609, closer for 63% of letters);
   every earlier model only tied it.
2. **But it does not personalise more.** Attribution stays at chance level
   (13–14%), and the outputs are *more* alike across writers (0.800, against
   0.599 for the baseline). G3 learned how people in general write each
   letter, not how this writer does. Style collapse is unchanged: the
   typical letter has moved from typeset towards handwriting.
3. **The rebuilt baseline is less collapsed than the shipped model** (alike
   0.554–0.599 against 0.768, which passes the 0.67 threshold) at the same
   clDice and floor. Attribution rose only to 16–17%, still failing.
4. **best vs last.** `g3_best` beats `g3_last` on every shape number but is
   more collapsed. Caveat: `best.pt` is chosen by IoU on the first 40 val
   fonts, which can include SD19 val writers, and the floor scores SD19 val and
   test writers. So `g3_best`'s floor may be slightly flattered. `g3_last`,
   which involved no selection, still beats the nearest font (0.632 vs 0.609).
   The LOO number is not affected by this.

## Verdict

G3 fails the style gate (alike 0.800 > 0.67; attribution 14.0% < 25%) and
passes clDice by a wide margin. It is kept as the new starting point for
letter shape: it is the best model the project has for letter accuracy.

## Next

- Use `g3_best` as the warm start for the personalisation arms, which target
  writer-specific copying rather than letter quality: G4
  (`generator.reference_weighting`) and G5 (`loss.projection`), each one
  variable on top of G3, scored by the same notebook cell.
- Make the best-checkpoint selection exclude SD19 writers, so the floor is
  measured on writers that played no part in choosing the checkpoint.

## Second session: G4, G5, G6 on top of G3

Each arm warm-started from `g3/best.pt`, 10K steps, one switch each:
G4 `generator.reference_weighting=true`; G5 `loss.projection=0.1
loss.adversarial=0.5 loss.char_aux=1.0 adversarial_start=0`; G6
`loss.ref_swap=10` (new: each target also drawn from another writer's
references; the own-reference error must be 10% below the swapped one).

### Photographed writers (test) and nearest-font floor

| Model | LOO clDice | Alike across writers | Attributed | Floor: model vs nearest font |
|---|---|---|---|---|
| g3_best (reference) | 0.430 | 0.800 | 14.0% | 0.672 vs 0.609 |
| g4_best | 0.423 | 0.762 | 14.7% | 0.667 |
| g4_last | 0.376 | 0.638 | 16.0% | 0.602 |
| g5_best | 0.401 | 0.726 | 15.0% | 0.641 |
| g5_last | 0.378 | 0.703 | 13.7% | 0.619 |
| g6_last | 0.390 | 0.682 | 13.7% | 0.620 |

`g6/best.pt` was lost: Google Drive was full (15 of 15 GB) and the upload
was dropped silently. Later runs write checkpoints to the runtime's disk.

### SD19 validation writers (model selection; `scripts/sd19_style.py`)

48 held-out SD19 writers, 4 groups of 10 (chance 10%), every letter not on
the seed sheet. Real hands are 0.460 alike.

| Model | clDice | Alike | Attributed |
|---|---|---|---|
| shipped | 0.600 | 0.890 | 14.2% |
| base_best | 0.593 | 0.783 | 17.8% |
| base_last | 0.582 | 0.747 | 18.6% |
| g3_best | 0.671 | 0.860 | 17.4% |
| g3_last | 0.629 | 0.753 | 19.2% |
| g4_best | 0.665 | 0.834 | 18.0% |
| g4_last | 0.601 | 0.704 | 17.8% |
| g5_best | 0.638 | 0.772 | 18.4% |
| g5_last | 0.616 | 0.712 | 18.1% |
| g6_last | 0.615 | 0.726 | 17.6% |

### What it shows

- **G4, G5 and G6 fail.** No arm moves attribution beyond noise on either
  test set (17-19% on SD19, 13.7-16% on the photographs). Where "alike"
  falls, letter shape falls with it: the letters get noisier, not more
  personal.
- **G6 learned its task only on training hands.** Its swap penalty fell from
  0.028 to 0.003 while val IoU fell, and held-out attribution did not move:
  it recognises training writers and recalls their letters instead of
  reading a new hand.
- **`best.pt` is not the arm.** In every fine-tune val IoU peaks at step
  2,000 and then declines (G4: 0.615 -> 0.598), because the schedule
  restarts at full learning rate. So `*_best` is G3 plus 2,000 steps, and
  `*_last` shows the arm with drift mixed in.
- **The two test sets agree**, so SD19 validation can stand in for the
  photographs when choosing between models.

Losses and attention variants on this architecture have run their course.
Next: retrieve-and-refine (G7, `data.content_source=nearest`): the content
image becomes the target letter from the training hand most like the
writer, so the content path, which the model ignores today, carries shape.

## G7: retrieve-and-refine

`data.content_source=nearest` (commit f5cbafa): the content image is the
target letter from one of the 4 training hands (fonts or SD19 writers) whose
seed letters look most like the style hand (`python -m hfont.data.neighbors`;
blurred 16x16 thumbnails). At inference the nearest *training* hand is
retrieved from the writer's seed letters (`--retrieve`, `--retrieve-cache`);
leave-one-out retrieves without the held-out letter. Warm-started from
`g3/best.pt`, 10K steps, checkpoints on the runtime's disk.

Val IoU (fonts): 0.720 at 2K, **0.732 at 4K**, 0.706 at 10K, against a 0.61
peak for every earlier arm. The content path, which the model ignored, now
carries shape.

| G7 | Photographs: clDice / alike / attributed | SD19 val: clDice / alike / attributed |
|---|---|---|
| g3_best (reference) | 0.430 / 0.800 / 14.0% | 0.671 / 0.860 / 17.4% |
| g7_best | 0.413 / **0.638** / **17.3%** | 0.653 / 0.766 / 17.2% |
| g7_last | 0.362 / 0.472 / 15.3% | 0.590 / 0.610 / 17.2% |

G7 is the first change that lowers how alike the letters are across writers
at near-unchanged letter shape (photographs 0.800 -> 0.638 at 0.430 ->
0.413). Attribution on the photographs rose 14.0% -> 17.3%, within the noise
of 10 writers (about +/-2 points per model). On SD19 it did not move. The
letters now vary as the retrieved neighbour writes, not yet as the writer
does. `g7_best` is kept as `runs/g7/best_slim_fp16.pt` (EMA weights, half
precision, 64 MB; same SD19 score as the full checkpoint).

## How far can attribution go? (ceilings)

Run on SD19 val writers (`scripts/sd19_ceiling.py`, `sd19_style.py --oracle
--only`, `scripts/retrieval_eval.py`):

| On the unseen digits 1245789 | Attributed |
|---|---|
| models (base_last, g3, g7) | 17.1-21.1% |
| nearest training writer, from seed thumbnails | 18.6% |
| oracle: the training writer whose unseen letters match best | 36.1% |
| **ceiling: the writer's own second copy** (100 writers) | **38.0%** |

| On all 33 unseen characters (114 writers, 3,182-writer bank) | Attributed |
|---|---|
| retrieval by thumbnails / by g3 or g7 style code | 15.1% / 14.4-15.5% |
| oracle other writer | 21.9% |

- The test tops out far below 100%: a writer's own second copy is
  attributed only 38% of the time. The 25% pass threshold is two thirds of
  the way to that ceiling, not a quarter.
- Someone in SD19 writes almost exactly like each held-out writer (oracle
  36% on digits), but neither the models nor retrieval from the 29 seed
  letters find them; the style code ranks the oracle's pick at a median 680
  of 3,182, no better than chance. Seed letters say little about how the
  same person writes the other letters, or the embeddings miss it.
- On letters, even the oracle other writer reaches only 21.9%, and the
  models are at 17-19%. SD19 attribution is nearly saturated as a guide; the
  photographed writers (real hands 0.287 alike against SD19's 0.46) and
  human judgement (gate 3) have the headroom.

## Recommendation

1. Gate-3 pilot (one rater, the owner) on `g7_best` vs `g3_best`: the
   numbers cannot separate them further.
2. If G7 holds up: learned retrieval, an embedding of the seed letters
   trained so writers with similar *unseen* letters land together (labels
   are the oracle similarities over 3,182 training writers). Its upside on
   SD19 is bounded by the oracle (36% digits, 22% letters).
3. G7 at a lower fine-tuning learning rate, so `best.pt` reflects the arm
   rather than G3 plus 2K steps.
4. Free Google Drive space before more runs (15 of 15 GB used).

## Third session: looking at the sheets (14 November)

The letter sheets had not been looked at since the baseline. Looking at
`g3_best` for three writers (Antoine, Mark, Elias) changes the reading of
everything above:

- **The three sheets are nearly the same.** G3 draws one set of heavy,
  soft-edged marker letters for every writer; the writers used thin pens and
  write very differently. The collapse is total: not even pen weight follows
  the writer.
- **G7's lower "alike" is mostly garbled letters** (an `a` drawn as a `3`, a
  `6` as a `U`, a `5` as a `6`), not personal style.
- **Every score rewarded this.** clDice, IoU and tolerant F1 forgive a
  misplaced stroke when it is drawn heavy (a ring 2 px off: clDice 0.49 thin,
  0.86 heavy). A model that hedges with weight and blur wins on all three.

Two changes follow (commits 785f677, 196075c):

1. **Pen redraw** (`eval_writers.py --pen`, `generate_rasters(pen=True)`):
   each generated letter's centreline is traced and redrawn at the writer's
   measured pen weight; written letters stay as written. On the sheets the
   generated letters now look written with the same pen as the originals.
2. **Centreline F1** (`evaluate.metrics.centreline_f1`): centreline against
   centreline, 3 px tolerance, the same whatever the pen.

| Ten writers, leave-one-out | clDice | Centreline F1 (weight-fair) |
|---|---|---|
| shipped | 0.373 | 0.379 |
| shipped + pen | 0.336 | 0.378 |
| g3_best | 0.430 | 0.396 |
| g3_best + pen | 0.355 | 0.395 |
| g7_best + pen | 0.342 | 0.383 |

- The pen redraw moves no centreline (0.396 -> 0.395); its clDice drop is
  the metric's bias, not a loss.
- Most of G3's clDice gain over the shipped model (+0.057) was heavier ink:
  on the fair score it is +0.017. G3 is still the best model, by less.
- G7 is below G3 on the fair score: rejected.

### Where this leaves the project

Every model so far draws the average letter; training changes moved it by
hundredths. What makes a generated letter sit beside the writer's own is
first what can be *measured* from the 30 letters and imposed exactly: size
(already done, `match_seed_proportions`), pen weight (now done), then slant
and line quality. Letter-shape personalisation is the smallest and least
predictable part (own-second-copy ceiling 38%, oracle other writer 22-36%).

Next, in order:
1. The owner compares before/after sheets (pen redraw on or off in the font).
2. Slant: measure the writer's against the model's and shear the generated
   letters, as size and pen are matched now.
3. Re-score with centreline F1 on SD19 validation (`sd19_style.py` still uses
   clDice) before trusting any further model comparison.
4. Only then more training; the candidate is a pen-normalised domain (every
   training glyph and every input redrawn at one thin pen), which removes
   the weight gap between fonts, SD19 and photographs and stops the model
   hedging with weight.
