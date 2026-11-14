"""Pen strokes recovered from glyph images, and drawn back (step A3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont import strokes  # noqa: E402
from hfont.evaluate.metrics import centreline_f1  # noqa: E402

FONT = ROOT / "data/fonts/NotoSans[wdth,wght].ttf"


@pytest.fixture(scope="module")
def letters():
    if not FONT.is_file():
        pytest.skip("content font not present")
    from hfont.charset import LATIN_CORE
    from hfont.generate import content_images_from_font

    return content_images_from_font(FONT, LATIN_CORE)


def test_every_glyph_survives_the_trip_through_pen_strokes(letters):
    scores = {}
    for key, image in letters.items():
        seq = strokes.glyph_sequence(image)
        assert seq is not None, key
        scores[key] = centreline_f1(strokes.render(strokes.from_sequence(seq), 128, 2.0), image)
    worst = min(scores, key=scores.get)
    assert np.mean(list(scores.values())) > 0.98
    assert scores[worst] > 0.9, (worst, scores[worst])


def test_the_dot_of_an_i_is_a_stroke_of_its_own(letters):
    seq = strokes.glyph_sequence(letters["0069.isol"])
    pieces = strokes.from_sequence(seq)
    assert len(pieces) == 2
    assert pieces[0][:, 1].max() < pieces[1][:, 1].min(), "the dot, above, comes first"


def test_a_crossing_bar_is_its_own_stroke():
    image = np.zeros((128, 128), np.float32)
    image[20:100, 62:66] = 1                     # stem
    image[50:54, 30:98] = 1                      # bar crossing it
    from hfont.generate import centreline

    pieces = strokes.trace_strokes(centreline(image))
    assert len(pieces) == 2
    stem, bar = pieces                           # the stem starts higher, so it is first
    assert np.ptp(stem[:, 1]) > 60 and np.ptp(stem[:, 0]) < 6
    assert np.ptp(bar[:, 0]) > 50 and np.ptp(bar[:, 1]) < 6
    assert stem[0, 1] < stem[-1, 1], "strokes run from their upper end"


def test_sequences_round_trip_and_mark_each_lift():
    a = np.array([[10, 10], [10, 40]], float)
    b = np.array([[30, 20], [60, 20]], float)
    seq = strokes.to_sequence([a, b], step=3.0)
    assert seq[:, 2].sum() == 2 and seq[-1, 2] == 1
    back = strokes.from_sequence(seq)
    assert len(back) == 2
    assert np.allclose(back[0][[0, -1]], a) and np.allclose(back[1][[0, -1]], b)
    assert np.abs(np.diff(back[0], axis=0)).max() <= 3.0 + 1e-6, "resampled every step"
