# Judgment calls made while you were away

Every choice I made on your behalf, with the reasoning, so you can overrule any
of them. Newest at the bottom. The standing instruction was "pick the more
conservative option", which I read as: prefer adding over replacing, measuring
over assuming, and leaving existing artifacts untouched.

---

### 1. There was no git repository, so I created one

`git rev-parse` reported "not a git repository". The instruction was to branch
from the current state and never touch main, which presupposes a repo.

**Decision:** `git init`, commit the entire current tree as `main`
(`a2e0c45 Snapshot before adversarial review`), then branch `review` and work
only there. Main is therefore an exact snapshot of the state you left, and
`git diff main..review` shows everything I changed.

**Conservative because:** it makes every change reversible and reviewable, and
it cannot lose work — nothing was modified before the snapshot commit.

### 2. Large binaries are git-ignored, not deleted

Checkpoints (61MB), the 445MB local glyph cache, fonts and built OTFs would
make the repository unusable.

**Decision:** a `.gitignore` covering `*.pt`, `*.otf`, `*.ttf`, `*.npy`,
`*.npz`, `data/cache_local/`, `data/fonts/`, `data/google-fonts/`. **Nothing was
deleted**; those files are all still on disk exactly where they were.

**Cost:** the review branch does not carry the checkpoint, so a fresh clone
cannot reproduce the numbers without it. Recorded here rather than solved,
since committing a 61MB binary is worse.

### 3. The Google Fonts corpus is not on this machine — I substituted Windows fonts

`data/cache_local` turns out to be a *local Windows-font* cache rendered with
the **old** framing (margin 0.08, no `frame_mode`), not the Google Fonts corpus.
The real corpus and its family-level splits lived on the Colab runtime, which
has been reclaimed. Re-downloading is ~4GB plus a 20-minute re-render, and
re-rendering the corpus is explicitly forbidden this round.

**Decision:** where an experiment needs "held-out fonts", use the ten
handwriting-style faces shipped with Windows that are not in Google Fonts —
the same stand-ins `scripts/handwriting_e2e.py` already uses. Every result
carrying this substitution says so.

**Cost:** n = 10, not 461, and they are commercial faces rather than the corpus
distribution. All correlations on them are reported with that caveat and a
confidence interval.

### 4. clDice was added beside the existing metrics, never in place of one

The instruction allows adding a metric but not replacing one. Measurement
showed tolerant F1 is blind to a 1px shift (0.0% of its range) and scores a
glyph dilated by 1px exactly 1.000 — at real handwriting stroke widths it
cannot see the project's known defect.

**Decision:** add `cl_dice` to `evaluate/metrics.py`, report it *alongside*
IoU, tol-F1, SSIM and coverage everywhere, and change no existing gate,
threshold or reported number.

### 5. I corrected my own docstring after it failed its own test

I first wrote that clDice "punishes breaking a stroke". Measured: a stroke
broken by a 3px gap scores 0.954, barely below 1.000. The claim was wrong.

**Decision:** rewrite the docstring to state only what was measured — weight
invariance, not topology — and say explicitly what it does *not* buy. Recorded
because it is precisely the failure mode this review exists to catch, and it
happened in my own new code.

### 6. Five subagents were used for the audit and the literature search

The audit spans ~20 modules and the research spans a literature I would
otherwise read serially.

**Decision:** three read-only audit agents (model/losses/training;
data/framing/intake/export; evaluation/conclusions) and two research agents
(few-shot font generation; thin-structure metrics and augmentation). All were
instructed to change no files and to give file:line evidence. I verify their
claims against the code myself before acting on any of them — an agent's report
is a lead, not a finding.

### 7. Four of five subagents died on a session rate limit — I continued without them

The audit was split across three read-only agents (model/losses/training;
data/framing/intake/export; evaluation/conclusions) and two research agents.
All but the metrics researcher were terminated by an HTTP 429 session limit
before reporting.

**Decision:** do the code audit myself, narrower and focused on the paths where
I could measure rather than opine, and mark the few-shot font-generation
literature review as **not done** rather than pretending a thin version of it.
REVIEW.md says so in its first paragraph and names what went unaudited (model
internals, loss interactions, the training schedule, `dataset.py`).

**Not retried** because the limit is session-wide: respawning would consume the
remaining budget and fail again.

### 8. I verified the surviving agent's headline number and it did not replicate

The metrics researcher reported that fattening a *wrong* glyph lifts tolerant F1
from 0.204 to 0.678 — two-thirds of the way to perfect. On this project's real
glyphs the same test gives 0.227 → 0.308.

**Decision:** report my number, keep the agent's qualitative point (the effect
is real, one-directional and a known pathology with good citations), and record
the discrepancy in REVIEW.md. Its probe was a synthetic shape; real glyphs
overlap less when fattened.

**Conservative because:** the direction of the error is against my own interest
— it makes the metric critique *weaker* than the agent claimed.

### 9. I added clDice but not Centerline F1, Betti numbers or a width ratio

The researcher recommended Centerline F1 (tolerant F1 computed skeleton to
skeleton) over raw clDice, on the grounds that clDice's implicit tolerance
collapses to the ground-truth stroke radius and scores 0.287 for a 1px shift.

**Decision:** adopt clDice only. On this project's real glyphs I measured a 1px
shift at **0.945**, not 0.287 — the brittleness that motivated their
recommendation does not appear here. Adding a fourth and fifth metric on the
strength of a number that did not replicate is the opposite of conservative.
Betti numbers are attractive (they would catch a missing `i` dot directly) and
are recorded in REVIEW.md as available, unadopted.

### 10. The stroke-weight fix was reverted rather than kept behind the visible-defect exception

Your rule allows keeping a fix that lowers the score if it repairs a visible
defect in the exported font, as the E bar did. The weight mismatch (+19/27/29%,
t > 8) is exactly such a defect, so the exception was available.

**Decision:** revert anyway. The E-bar exception applied because the score cost
there was an artifact of a *more complete* target being harder to match. Here
the cost is real: −0.039 mean, and down on clDice, which is blind to weight and
therefore cannot be blamed. The letters genuinely got less like the writer's.
Shipping it would have traded a defect a reader might notice for a defect every
metric can see.

**Recorded for you to overrule:** if you judge the visual consistency worth more
than 0.039 of fidelity, the implementation is in the branch history and can be
restored from this description; it is the only change I reverted for a
measured reason rather than a correctness one.

### 11. `output/review/*.otf` were rebuilt after the revert

The three writer fonts in `output/review/` were first built with the rejected
fix. I rebuilt them from the reverted code so the artifacts match the shipped
behaviour (LOO 0.403 / 0.573 / 0.335, unchanged from main). Nothing under
`output/myhand_diag/`, `output/myhand/`, `output/AntoineHand/`,
`output/JadHand/` or `runs/` was touched.
