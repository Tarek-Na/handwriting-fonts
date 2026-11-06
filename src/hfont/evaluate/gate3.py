"""Gate 3, the human test: would someone take a generated letter for the writer's own?

The question asked of a rater is yes/no: here are letters by one person; is
this next letter by the same person? Asked of three kinds of probe letter,
always the same character so only the hand differs:

- ``generated``: the model's blind attempt at a letter the writer wrote,
  drawn from their other letters (what the font would contain);
- ``ceiling``: the writer's own letter, held out of the references. A perfect
  generator scores this;
- ``floor``: the same letter written by a different writer. A model that
  ignores the writer scores about this.

Reporting all three is what makes the number mean something. Published
handwriting tests sit near 50% "real" with no anchor, which cannot tell good
fakes from raters who cannot tell hands apart at all. Here the generated rate
is read against the ceiling and the floor, as ``(generated - floor) /
(ceiling - floor)``: 0 means "no more like the writer than a stranger", 1
means "as convincing as their own hand".

Protocol after HYPE (Zhou et al. 2019): balanced conditions, randomised order
per rater, no time limit, bootstrap intervals. At ~100 judgements a rate has a
95% interval of about +-9 points, so several hundred per condition (a few
raters) are needed to separate 70% from 60%.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np

CONDITIONS = ("generated", "ceiling", "floor")


@dataclass(frozen=True)
class Trial:
    trial_id: int
    writer: str            # whose letters are shown as references
    char: str              # the probe's character
    condition: str         # generated / ceiling / floor
    source: str            # who wrote the probe (the writer, "model", or another writer)
    references: tuple[str, ...]  # characters shown as the writer's letters


def plan_trials(written: dict[str, set[str]], per_writer: int = 4, n_refs: int = 8,
                seed: int = 0) -> list[Trial]:
    """The same balanced trial set for every rater (order is shuffled per rater).

    ``written`` maps each writer to the characters they wrote. For each writer,
    ``per_writer`` characters are probed in each of the three conditions; the
    probe character is never among the references, and the floor's letter comes
    from another writer who also wrote that character.
    """
    rng = random.Random(seed)
    trials: list[Trial] = []
    writers = sorted(written)
    for w in writers:
        shared = sorted(c for c in written[w]
                        if any(c in written[v] for v in writers if v != w))
        chars = rng.sample(shared, min(per_writer, len(shared)))
        for ch in chars:
            pool = sorted(written[w] - {ch})
            refs = tuple(rng.sample(pool, min(n_refs, len(pool))))
            stranger = rng.choice([v for v in writers if v != w and ch in written[v]])
            for condition, source in (("generated", "model"), ("ceiling", w), ("floor", stranger)):
                trials.append(Trial(len(trials), w, ch, condition, source, refs))
    return trials


def trials_as_dicts(trials: list[Trial]) -> list[dict]:
    return [asdict(t) for t in trials]


def _bootstrap(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(responses: list[dict], seed: int = 0) -> dict:
    """Yes-rates per condition with 95% intervals, the anchored score, and agreement.

    ``responses`` are rows ``{"rater", "trial_id", "condition", "same": bool}``
    (one per judgement, pooled over raters). Agreement is the fraction of
    trial pairs judged by two raters on which they gave the same answer.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for condition in CONDITIONS:
        yes = np.array([r["same"] for r in responses if r["condition"] == condition], dtype=float)
        lo, hi = _bootstrap(yes, rng)
        out[condition] = {"n": int(yes.size), "yes": float(yes.mean()) if yes.size else float("nan"),
                          "ci95": (lo, hi)}
    g, c, f = (out[k]["yes"] for k in CONDITIONS)
    out["anchored"] = (g - f) / (c - f) if c - f > 1e-9 else float("nan")

    by_trial: dict[int, dict[str, bool]] = {}
    for r in responses:
        by_trial.setdefault(int(r["trial_id"]), {})[r["rater"]] = bool(r["same"])
    same = pairs = 0
    for answers in by_trial.values():
        values = list(answers.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs += 1
                same += values[i] == values[j]
    out["raters"] = len({r["rater"] for r in responses})
    out["agreement"] = same / pairs if pairs else float("nan")
    return out
