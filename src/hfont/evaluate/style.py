"""Does the model draw *this* writer's letters, or everyone's average letter?

Leave-one-out clDice cannot answer that: a model that draws one typeset letter
for every hand still scores about the same as one that follows the writer,
because the writer's real letter is itself close to the typeset one. Three
numbers here measure the thing directly, all from the same inputs: each
writer's real letters, and the model's blind attempt at each of them (drawn
from that writer's other letters).

- ``cross_writer_similarity`` of the *blind attempts*: how alike the model's
  letters are across writers. Near 1.0 means the references are ignored. The
  model's output depends only on (letter, references), so this is also the
  "style sensitivity" test of swapping one writer's references for another's.
- the same on the *real* letters: how alike the hands really are. This is the
  target; a model that follows the writer should come down towards it.
- ``attribution_accuracy``: for each writer and letter, is the blind attempt
  closer to that writer's real letter than to every other writer's? Chance is
  1 / (number of writers). This is the held-out shape check: it only rises if
  the model draws a shape it was never shown, in the writer's own way.

All three are clDice, so stroke weight (which AdaIN does carry) barely counts;
what is scored is where the strokes run.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .metrics import cl_dice

Letters = dict[str, np.ndarray]


def _shared(letters: dict[str, Letters]) -> list[str]:
    keys = None
    for per_writer in letters.values():
        keys = set(per_writer) if keys is None else keys & set(per_writer)
    return sorted(keys or ())


def cross_writer_similarity(letters: dict[str, Letters]) -> float:
    """Mean clDice between different writers' versions of the same letter."""
    writers = sorted(letters)
    keys = _shared(letters)
    scores = [cl_dice(letters[a][k], letters[b][k])
              for a, b in combinations(writers, 2) for k in keys]
    return float(np.mean(scores)) if scores else float("nan")


def attribution(blind: dict[str, Letters], real: dict[str, Letters]) -> dict[str, float]:
    """How often each writer's blind attempts are nearest that writer's real letters.

    Returns per-writer accuracy (fraction of their letters attributed to them)
    plus ``"mean"`` over writers and ``"chance"``. A tie with another writer
    counts as a miss. A model that draws the same image for everyone can only be
    nearest one writer per letter, so it scores at most chance: whoever's hand
    happens to look like its average letter.
    """
    writers = sorted(set(blind) & set(real))
    keys = _shared({w: blind[w] for w in writers} | {f"real:{w}": real[w] for w in writers})
    result: dict[str, float] = {}
    for w in writers:
        hits = []
        for k in keys:
            own = cl_dice(blind[w][k], real[w][k])
            others = max(cl_dice(blind[w][k], real[v][k]) for v in writers if v != w)
            hits.append(own > others)
        result[w] = float(np.mean(hits)) if hits else float("nan")
    result["mean"] = float(np.mean([result[w] for w in writers])) if writers else float("nan")
    result["chance"] = 1.0 / len(writers) if writers else float("nan")
    return result


def style_report(blind: dict[str, Letters], real: dict[str, Letters]) -> dict[str, float]:
    """The three numbers together, as eval_writers.py reports them."""
    acc = attribution(blind, real)
    return {
        "generated_similarity": cross_writer_similarity(blind),
        "real_similarity": cross_writer_similarity(real),
        "attribution": acc["mean"],
        "attribution_chance": acc["chance"],
        "attribution_per_writer": {w: v for w, v in acc.items() if w not in ("mean", "chance")},
    }


def grouped_style_report(blind: dict[str, Letters], real: dict[str, Letters],
                         group_size: int = 10) -> dict[str, float]:
    """``style_report`` over many writers, in groups of ``group_size``.

    Attribution among 80 writers has a chance of 1/80, which cannot be set
    beside the ten photographed writers' 1/10. Splitting the writers (sorted,
    so the groups are the same for every model) into groups of ten keeps
    chance at 10% and averages the numbers over the groups. A last group
    smaller than ``group_size`` is dropped.
    """
    writers = sorted(set(blind) & set(real))
    groups = [writers[i:i + group_size]
              for i in range(0, len(writers) - group_size + 1, group_size)]
    reports = [style_report({w: blind[w] for w in g}, {w: real[w] for w in g}) for g in groups]
    if not reports:
        return {"groups": 0}
    mean = lambda key: float(np.mean([r[key] for r in reports]))  # noqa: E731
    return {"groups": len(reports), "writers": len(groups) * group_size,
            "generated_similarity": mean("generated_similarity"),
            "real_similarity": mean("real_similarity"),
            "attribution": mean("attribution"),
            "attribution_chance": 1.0 / group_size}
