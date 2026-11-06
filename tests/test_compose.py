"""Missing letters built from the writer's own strokes land where the letter belongs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.compose import compose  # noqa: E402

SIZE, LINE, XTOP = 128, 96, 70          # canvas, baseline row, x-height top


def _b() -> np.ndarray:
    """A b: stem on the left from the ascender to the line, bowl on the right in the x band."""
    im = np.zeros((SIZE, SIZE), np.float32)
    im[40:LINE, 50:54] = 1                  # stem
    im[XTOP:XTOP + 4, 50:76] = 1            # bowl top
    im[LINE - 4:LINE, 50:76] = 1            # bowl bottom
    im[XTOP:LINE, 72:76] = 1                # bowl right side
    return im


def _n() -> np.ndarray:
    im = np.zeros((SIZE, SIZE), np.float32)
    im[XTOP:LINE, 50:54] = 1                # left stem, full height
    im[XTOP:XTOP + 4, 50:76] = 1            # arch
    im[XTOP:LINE, 72:76] = 1                # right stem
    im[LINE - 4:LINE, 66:76] = 1            # a foot on the right only
    return im


def _o() -> np.ndarray:
    im = np.zeros((SIZE, SIZE), np.float32)
    im[XTOP:LINE, 52:76] = 1
    im[XTOP + 4:LINE - 4, 56:72] = 0
    return im


def _extent(im):
    rows = np.nonzero((im > 0.5).any(axis=1))[0]
    cols = np.nonzero((im > 0.5).any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _stem_col(im, rows):
    """Column of the vertical stroke that runs through all of ``rows``."""
    full = (im[rows] > 0.5).all(axis=0)
    return float(np.mean(np.nonzero(full)[0]))


def test_b_gives_d_p_q_with_stem_and_bowl_in_place():
    out = compose({"b": _b(), "o": _o()}, LINE)
    assert set(out) == {"d", "p", "q"}
    b_top, b_bot, b_left, b_right = _extent(_b())

    d_top, d_bot, *_ = _extent(out["d"])
    assert (d_top, d_bot) == (b_top, b_bot), "d keeps b's height and sits on the line"
    assert _stem_col(out["d"], slice(45, 65)) > 64, "d's stem is on the right"

    for ch, side in (("p", "left"), ("q", "right")):
        top, bot, *_ = _extent(out[ch])
        assert abs(top - XTOP) <= 1, f"{ch}'s bowl starts at the x-height"
        assert bot > LINE + 10, f"{ch} hangs below the line"
        assert bot <= SIZE, f"{ch}'s tail fits the canvas"
        col = _stem_col(out[ch], slice(LINE + 2, LINE + 8))
        assert (col < 64) if side == "left" else (col > 64), f"{ch}'s stem is on the {side}"


def test_n_turned_over_is_a_u_in_the_same_band():
    out = compose({"n": _n()}, LINE)
    top, bot, *_ = _extent(out["u"])
    assert (top, bot) == (XTOP, LINE), "u stays between x-height and baseline"
    # n's foot was bottom right; turned over it is top left.
    assert out["u"][XTOP + 1, 50:54].mean() > 0.5 and out["u"][XTOP + 1, 72:76].mean() > 0.5
    assert out["u"][LINE - 2, 52:60].mean() > 0.5, "u's cup closes at the bottom"


def test_digit_turns_about_itself_and_written_letters_are_kept():
    six = np.zeros((SIZE, SIZE), np.float32)
    six[50:LINE, 50:54] = 1
    six[70:LINE, 50:76] = 1
    out = compose({"6": six, "u": _n(), "n": _n()}, LINE)
    assert "u" not in out, "a written letter is never replaced"
    assert _extent(out["9"])[:2] == _extent(six)[:2], "9 occupies 6's rows"
    assert "W" not in out, "no donor, no letter"
