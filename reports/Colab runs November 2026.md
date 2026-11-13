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
