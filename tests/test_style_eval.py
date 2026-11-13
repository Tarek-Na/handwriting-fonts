"""The style measures must tell a model that follows the writer from one that does not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.evaluate.style import (  # noqa: E402
    attribution,
    cross_writer_similarity,
    grouped_style_report,
    style_report,
)


def _stroke(angle: float, offset: int, size: int = 64) -> np.ndarray:
    """A thick straight stroke through the canvas at an angle: one writer's 'letter'."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32) - size / 2
    t = np.deg2rad(angle)
    dist = np.abs(-np.sin(t) * (xx - offset) + np.cos(t) * yy)
    return (dist < 3).astype(np.float32)


def _hands(n_writers: int = 4, letters: str = "abc") -> dict[str, dict[str, np.ndarray]]:
    # Each writer draws each letter at their own slant and position, far enough
    # apart that clDice can tell the hands apart.
    return {f"w{w}": {k: _stroke(-60 + 40 * w + 10 * i, 4 * i - 4) for i, k in enumerate(letters)}
            for w in range(n_writers)}


def test_a_model_that_copies_the_hand_is_attributed_to_it():
    real = _hands()
    acc = attribution(real, real)
    assert acc["mean"] == 1.0
    assert acc["chance"] == 0.25


def test_one_letter_for_everyone_is_not_attributed_and_looks_identical():
    """The collapse this project has: the same image whoever wrote the references."""
    real = _hands()
    average = {w: {k: _stroke(0, 0) for k in letters} for w, letters in real.items()}
    assert cross_writer_similarity(average) == 1.0
    # Identical output can only ever be nearest one writer per letter, so a
    # collapsed model scores at most chance however its average letter falls.
    acc = attribution(average, real)
    assert acc["mean"] <= acc["chance"]
    report = style_report(average, real)
    assert report["real_similarity"] < report["generated_similarity"]


def test_similarity_uses_only_letters_every_writer_has():
    real = _hands()
    real["w0"]["z"] = np.zeros((64, 64), np.float32)   # only one writer has a z
    assert cross_writer_similarity(real) == cross_writer_similarity(
        {w: {k: v for k, v in d.items() if k != "z"} for w, d in real.items()})


def test_many_writers_are_scored_in_groups_with_a_fixed_chance():
    """80 SD19 writers must give numbers comparable with the ten photographed ones."""
    real = _hands(n_writers=9)
    copied = grouped_style_report(real, real, group_size=4)
    assert copied["groups"] == 2 and copied["writers"] == 8   # the ninth is dropped
    assert copied["attribution"] == 1.0 and copied["attribution_chance"] == 0.25
    average = {w: {k: _stroke(0, 0) for k in letters} for w, letters in real.items()}
    collapsed = grouped_style_report(average, real, group_size=4)
    assert collapsed["attribution"] <= 0.25 and collapsed["generated_similarity"] == 1.0
