"""Letters that share a body get the writer's own dots (step A2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.fontbuild import rasm  # noqa: E402

LINE = 90


def _body() -> np.ndarray:
    """A beh-like body: a stroke on the line with a tooth rising at its right."""
    im = np.zeros((128, 100), np.float32)
    im[LINE - 3:LINE + 1, 10:90] = 1
    im[LINE - 25:LINE, 80:84] = 1
    return im


def _dot(im: np.ndarray, row: int, col: int, size: int = 5) -> np.ndarray:
    im[row:row + size, col:col + size] = 1
    return im


def _pieces_in(im: np.ndarray, rows: slice) -> int:
    """Pieces of ink in ``rows``, left of the tooth (columns 80-84)."""
    _, boxes, _ = rasm._pieces(im[rows, :78])
    return len(boxes)


def _sheet() -> tuple[dict, dict]:
    """beh (one dot below), noon (one above), qaf (two above), jeem (one below), yeh (two below)."""
    letters = {}
    for form in (Form.INIT, Form.FINA):
        letters[GlyphSpec("ب", form)] = _dot(_body(), LINE + 8, 40)
        letters[GlyphSpec("ن", form)] = _dot(_body(), LINE - 16, 45)
        two = _dot(_dot(_body(), LINE - 16, 30), LINE - 16, 40)
        letters[GlyphSpec("ق", form)] = two
        letters[GlyphSpec("ج", form)] = _dot(_body(), LINE + 8, 60)
    letters[GlyphSpec("ي", Form.FINA)] = _dot(_dot(_body(), LINE + 8, 30), LINE + 8, 40)
    letters[GlyphSpec("ت", Form.FINA)] = _body()       # one the writer did write
    lines = {spec: float(LINE) for spec in letters}
    return letters, lines


def test_dots_are_split_from_the_body_but_other_strokes_stay():
    im = _dot(_body(), LINE + 8, 40)
    im[20:60, 20:24] = 1                                # a separate upright, not a dot
    body, dots = rasm.split_dots(im, dot_size=12)
    assert dots[LINE + 8:LINE + 13, 40:45].all() and dots.sum() == 25
    assert body[20:60, 20:24].all(), "a long separate stroke is not a dot"
    assert not body[LINE + 8:, :].any()


def test_two_dots_written_as_one_dash_are_dots():
    im = _body()
    im[LINE + 8:LINE + 11, 30:52] = 1                   # 22 px wide, 3 high: a dash
    body, dots = rasm.split_dots(im, dot_size=12)
    assert dots[LINE + 8:LINE + 11, 30:52].all()
    upright = _body()
    upright[20:60, 20:24] = 1                           # tall: never a dash
    assert rasm.split_dots(upright, dot_size=12)[1].sum() == 0


def test_teh_is_behs_body_with_two_dots_above_where_behs_dot_was():
    letters, lines = _sheet()
    out, out_lines = rasm.compose(letters, lines, dot_size=12, tolerance=6)
    teh = out[GlyphSpec("ت", Form.INIT)]
    shift = int(out_lines[GlyphSpec("ت", Form.INIT)] - LINE)
    assert not teh[LINE + shift + 4:].any(), "beh's dot below is gone"
    above = teh[:LINE + shift - 4, :78]
    assert _pieces_in(teh, slice(0, LINE + shift - 4)) == 2, "two dots above"
    cols = np.nonzero(above.any(axis=0))[0]
    assert abs(cols.mean() - 42) <= 6, "centred where beh's dot was"


def test_theh_and_sheen_get_three_dots_and_hah_none():
    letters, lines = _sheet()
    letters[GlyphSpec("س", Form.INIT)] = _body()
    lines[GlyphSpec("س", Form.INIT)] = float(LINE)
    out, out_lines = rasm.compose(letters, lines, dot_size=12, tolerance=6)
    theh = GlyphSpec("ث", Form.INIT)
    assert _pieces_in(out[theh], slice(0, int(out_lines[theh]) - 4)) == 3
    sheen = GlyphSpec("ش", Form.INIT)                  # no dots of its own: over what rises
    tooth_top = int(out_lines[sheen]) - 25              # above the line, here the tooth
    _, boxes, _ = rasm._pieces(out[sheen][:tooth_top])
    assert len(boxes) == 3 and all(80 - 8 <= b[1].start <= 84 for b in boxes[1:2])
    hah = out[GlyphSpec("ح", Form.INIT)]
    assert hah.sum() == _body().sum(), "hah is jeem without its dot"


def test_alef_maksura_is_yeh_without_dots_and_written_letters_are_kept():
    letters, lines = _sheet()
    out, _ = rasm.compose(letters, lines, dot_size=12, tolerance=6)
    assert out[GlyphSpec("ى", Form.FINA)].sum() == _body().sum()
    assert out[GlyphSpec("ت", Form.FINA)] is letters[GlyphSpec("ت", Form.FINA)]
    assert GlyphSpec("ى", Form.INIT) not in out, "only the forms the rule lists"
