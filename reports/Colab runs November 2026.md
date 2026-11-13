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
