"""Positional-form completion: the score and the writer split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.evaluate.metrics import aligned_centreline_f1, centreline_f1  # noqa: E402


def _hook(h=60, w=90, x0=10, y0=10, pen=3):
    im = np.zeros((h, w), np.float32)
    im[y0:y0 + 30, x0:x0 + pen] = 1                     # an upright
    im[y0 + 30 - pen:y0 + 30, x0:x0 + 40] = 1           # and a stroke along the line
    return im


def test_where_a_letter_sits_in_its_image_does_not_count():
    a = _hook()
    b = _hook(h=80, w=140, x0=70, y0=35)               # same letter, other crop
    assert centreline_f1(a, b[:60, :90]) < 0.5, "unaligned, the plain score misses it"
    assert aligned_centreline_f1(a, b) == 1.0


def test_shape_and_size_still_count():
    a = _hook()
    longer = np.zeros((60, 120), np.float32)
    longer[10:40, 10:13] = 1
    longer[37:40, 10:90] = 1                            # the stroke twice as long
    assert aligned_centreline_f1(a, longer) < 0.85
    other = np.zeros((60, 90), np.float32)
    other[30:33, 10:80] = 1                             # no upright at all
    assert aligned_centreline_f1(a, other) < 0.8
    assert aligned_centreline_f1(a, np.zeros_like(a)) == 0.0


def test_every_writer_is_on_exactly_one_side():
    from hfont.data.ahawp import split_of

    writers = [f"user{n:03d}" for n in range(1, 83)]
    sides = {s: [w for w in writers if split_of(w) == s] for s in ("train", "val", "test")}
    assert [len(sides[s]) for s in ("train", "val", "test")] == [49, 17, 16]
    assert sum(len(v) for v in sides.values()) == 82
