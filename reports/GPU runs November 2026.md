# GPU runs, November 2026: baseline, G1, G2 (G3 paused)

Run on a laptop RTX 3060 (6 GB), following [SETUP_GPU.md](../SETUP_GPU.md) and
[instruction.md](../instruction.md) §5.1-5.3. Every model was judged on the
pass thresholds fixed in §4 before any run: similarity across writers
**≤ 0.67**, attribution **≥ 25%**, mean leave-one-out clDice **≥ 0.36**, and it
must clearly beat the nearest-font floor. No run trained or tuned on the ten
photographed writers. Text and numbers only; the letter sheets stay on disk.

## Summary

| model | LOO clDice | alike across writers | attributed | SD19: model vs nearest font | fonts IoU / tol-F1 | verdict |
|---|---|---|---|---|---|---|
| shipped phase1c | 0.373 | 0.768 | 13.7% | 0.598 vs 0.590 | 0.606 / 0.828 (§4) | — |
| hw40k best (step 14k) | 0.368 | 0.665 | 13.7% | 0.601 vs 0.590 | 0.609 / 0.820 | baseline |
| hw40k last (step 40k) | 0.349 | 0.565 | 15.7% | not run | — | — |
| G1 best (step 2k) | 0.368 | 0.642 | 12.7% | 0.600 vs 0.590 | 0.610 / 0.821 | **fail** |
| G1 last (step 10k) | 0.360 | 0.613 | 14.7% | 0.576 vs 0.590 | — | **fail** |
| G2 best (step 4k) | 0.370 | 0.665 | 13.7% | 0.601 vs 0.590 | 0.615 / 0.825 | **fail** |
| G2 last (step 10k) | 0.367 | 0.651 | 17.0% | 0.591 vs 0.590 | — | **fail** |

Real letters are 0.287 alike across writers; chance attribution is 10%.
Attribution gains under ~5 points are noise (§4).

**What it means.** Every arm makes the blind attempts differ more between
writers (0.768 → 0.61-0.67) without making them more *right* for the writer:
attribution stays at 13-17%, and on SD19's real letters no model beats the
nearest typeface by more than 0.011. The sheets agree: the blind attempts are
still typeset letters in the writer's weight and slant (Bahaa's two-stroke
`k` and Mark's barred `i` are not reproduced). Lower similarity alone is not
progress; G1's full run gets there by garbling letters.

## Per writer, leave-one-out clDice

| writer | phase1c | hw40k best | G1 best | G1 last | G2 best | G2 last |
|---|---|---|---|---|---|---|
| Alans | 0.323 | 0.338 | 0.344 | 0.331 | 0.340 | 0.337 |
| Bahaa | 0.300 | 0.356 | 0.354 | 0.357 | 0.351 | 0.351 |
| Elias | 0.398 | 0.361 | 0.361 | 0.369 | 0.350 | 0.363 |
| Elissas | 0.556 | 0.559 | 0.567 | 0.553 | 0.577 | 0.572 |
| Karims | 0.291 | 0.309 | 0.303 | 0.271 | 0.332 | 0.307 |
| Layanes | 0.453 | 0.434 | 0.442 | 0.433 | 0.414 | 0.429 |
| Mark | 0.468 | 0.366 | 0.367 | 0.348 | 0.361 | 0.343 |
| My | 0.283 | 0.297 | 0.300 | 0.293 | 0.310 | 0.295 |
| Antoine | 0.381 | 0.383 | 0.372 | 0.393 | 0.403 | 0.412 |
| Jad | 0.279 | 0.281 | 0.274 | 0.252 | 0.265 | 0.260 |

Mark loses 0.10 under every new model; Bahaa gains 0.05.

## The runs

**Baseline (hw40k, §5.1).** `--recipe recommended --set steps=40000 --set
data.handwriting_weight=6.0`, batch 32 (it fits in 6 GB). Validation IoU on
fonts: 0.469 at 2k, 0.570 at 8k, peak **0.585 at step 14k**, then a slow
decline to 0.572 at 40k while training loss kept falling: it over-fits the
training fonts in the second half. `best.pt` is therefore the step-14k model.
Its final step is more varied across writers (0.565) but worse on them (LOO
0.349).

**G1** (`generator.attention_levels=[0,1]`, warm start from hw40k best,
batch 32). Font IoU peaked at its step 2k (0.5848) and declined to 0.577, so
`best.pt` holds attention trained for only 2,000 steps. The full run copies
stroke fragments from other references into the target: Bahaa's `b` and `D`
come out mangled, `W` garbled, `N`/`M` broken. Failed on attribution and the
floor at both checkpoints; the full run falls below the nearest font.

**G2** (`loss.self_recon=1.0`, from hw40k best because G1 failed, without
G1's switch). **Batch 16**: at batch 32 the extra pass overflowed 6 GB (see
below). Font IoU in training read ~0.545, but the test-split report is the
best of all runs (0.615), so the training-time number is not comparable
across batch sizes. Attribution 17.0% at the full run, +3.3 points over the
baseline: under the noise level. Ties the nearest font.

**G3** (`--cache data/cache_sd19 --set data.sd19_weight=1.0`, from hw40k
best, batch 32). Paused by the owner at step 1,315 of 10,000;
`runs/g3/last.pt` resumes it with the same command. G4 and G5 not started.

## Training, step by step

From each run's `console.log`. "Val IoU" is the trainer's check on 40
held-out fonts every 2,000 steps (it chooses `best.pt`); L1 and Dice are the
training losses, lower is better.

**Baseline hw40k** (batch 32, 40,000 steps, about 8 hours; ~4 would have done
without the memory spill after step 34k):

| step | val IoU | val L1 | train L1 | train Dice | speed |
|---|---|---|---|---|---|
| 2,000 | 0.469 | 0.0686 | 0.314 | 0.496 | 44 img/s* |
| 4,000 | 0.514 | 0.0579 | | | 100 |
| 6,000 | 0.551 | 0.0497 | | | 99 |
| 8,000 | 0.570 | 0.0458 | | | 93 |
| 10,000 | 0.580 | 0.0439 | 0.227 | 0.353 | 95 |
| 12,000 | 0.585 | 0.0431 | 0.199 | 0.313 | 95 |
| **14,000** | **0.585** (best) | 0.0427 | 0.188 | 0.293 | 95 |
| 20,000 | 0.583 | 0.0424 | 0.145 | 0.236 | 96 |
| 30,000 | 0.577 | 0.0428 | 0.096 | 0.163 | 95 |
| 34,000 | 0.574 | 0.0430 | | | 48 (spill starts) |
| 40,000 | 0.572 | 0.0432 | 0.078 | 0.140 | 11 |

\* slowed while the SD19 cache was being built on the same disk.

Training L1 keeps falling four-fold after step 14k while held-out IoU slips
by 0.013: the second half memorises the training fonts.

**G1** (batch 32, from hw40k best; 2.5 hours including a restart):

| step | val IoU | val L1 | train L1 | speed |
|---|---|---|---|---|
| **2,000** | **0.585** (best) | 0.0427 | 0.180 | 75 img/s |
| 4,000 | 0.584 | 0.0428 | | 48 (spill) |
| 6,000 | 0.583 | 0.0425 | | 49 |
| 8,000 | 0.579 | 0.0427 | | 42, then 7 |
| 10,000 | 0.577 | 0.0427 | 0.109 | 45 after restart |

**G2** (batch 16, from hw40k best; 45 minutes, no slowdown):

| step | val IoU | val L1 | train L1 | speed |
|---|---|---|---|---|
| 2,000 | 0.544 | 0.0377 | 0.206 | 60 img/s |
| **4,000** | **0.546** (best) | 0.0373 | | 60 |
| 6,000 | 0.543 | 0.0372 | | 60 |
| 8,000 | 0.541 | 0.0375 | | 60 |
| 10,000 | 0.539 | 0.0376 | 0.137 | 60 |

G2's val IoU starts 0.04 below where hw40k ended although it warm-starts from
it, and its val L1 is lower than any other run; the batch size changes this
number, so it is not compared across runs (the test-split report, 0.615, is
G2's best). The aborted batch-32 attempt ran 200 steps at 16-17 img/s.

**G3** (batch 32, SD19 cache): paused at step 1,315, 75-77 img/s, train L1
0.32-0.34 (higher than the others because real SD19 letters are now in the
batches). Its validation set includes SD19 writers (1,815 glyphs, not
2,040), so its val IoU will not compare with the runs above.

## Choices made, so they are not mistaken for accidents

- **SD19 lives in a separate cache.** `DatasetConfig.sd19_weight` defaults to
  1.0, so adding SD19 to `data/cache` would silently put it into the baseline,
  G1 and G2, and would also change the validation set (it takes every k-th val
  record) and the test report. `data/cache` stays fonts only; `data/cache_sd19`
  is a copy with 3,403 SD19 writers added (`sd19_cache.py` counts digits toward
  its 40-character minimum, hence more than the ~2,900 with 40 letters).
- **`data.handwriting_weight=6.0` in every arm.** The §5.3 command lines omit
  it, which would drop to the recipe's 3.0 and change a second variable.
- **Both checkpoints scored.** `best.pt` is chosen by font IoU, which peaks
  early in every run, so it often holds a barely-trained arm. The verdicts
  use `best.pt` as specified; `last.pt` is reported beside it.
- **The nearest-font floor here is 0.590, not 0.576.** The floor retrieves
  from `data/cache_local`, the old machine's 361 Windows fonts. 305 of them
  exist on this laptop (the rest are CAD/GOST fonts, Cascadia and Droid), so
  the cache was rebuilt from those (`data/manifest_local.json` filtered to
  existing paths). The shipped model reproduces its recorded 0.598 exactly on
  the same 80 writers and 2,511 letters, so the test is the same; only the
  font pool differs. Arms are judged against both numbers, and none clears
  either clearly.
- **Corpus drift.** Google Fonts has changed since the old machine fetched it:
  3,057 fonts (238 test) here against 237 test fonts before; content-copy
  baseline tol-F1 on the test split reads 0.692 here.

## Practical notes for a 6 GB Windows GPU

- **A full GPU does not raise out-of-memory on Windows; it crawls.** The
  driver spills GPU memory into system RAM: the GPU shows 100% busy but draws
  ~60 W instead of ~128 W, and training drops from ~95 to 7-12 images/s. It
  hit the baseline after step ~34k (4 hours became 8), G1 after its first
  evaluation, and G2 at batch 32 from the start (17 images/s). Watch
  `nvidia-smi --query-gpu=memory.used,power.draw`. Plain recipe and G1 fit at
  batch 32; arms with an extra pass need batch 16 (60 images/s, 3.9 GB).
- Resuming a batch-32 run reloads optimizer and EMA state onto the GPU and
  can then hit a real CUDA out-of-memory error; a second attempt went through.
- On a connection that drops, `git clone` of google/fonts restarts from zero
  each time. What worked: a tree-only partial clone
  (`--filter=blob:none --no-checkout --sparse`), `git checkout`, then
  `git sparse-checkout add` 40 font folders at a time, retrying each batch.
  `curl -C - --retry` for `by_write.zip`.
- Speeds: plain recipe 95-100 images/s at batch 32 (T4 on Colab: 77); G1
  75; G2 at batch 16, 60.
