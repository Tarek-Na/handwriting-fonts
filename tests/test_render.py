"""The preview renderer draws a font as a text engine would."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1] / "scripts")]
pytest.importorskip("uharfbuzz")


def _font(advance: int, hole: bool) -> bytes:
    """One glyph 'a': a 600-unit square (with a square hole in it when ``hole``), ``advance`` wide."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    pen = TTGlyphPen(None)
    for x0, y0, x1, y1, clockwise in [(0, 0, 600, 600, True)] + ([(200, 200, 400, 400, False)] if hole else []):
        pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        for i, p in enumerate(pts if clockwise else pts[::-1]):
            (pen.moveTo if i == 0 else pen.lineTo)(p)
        pen.closePath()
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "a"])
    fb.setupCharacterMap({ord("a"): "a"})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "a": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "a": (advance, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "T", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    buffer = io.BytesIO()
    fb.save(buffer)
    return buffer.getvalue()


def test_glyphs_that_overlap_at_a_join_are_drawn_solid_and_a_letters_hole_stays_open():
    from ahawp_font import render

    over = render(_font(400, hole=False), ["aa"], px_per_em=100) < 128      # two squares overlapping by 200 units
    cols = np.nonzero(over.any(0))[0]
    row = over[np.nonzero(over.any(1))[0].mean().astype(int), cols[0]:cols[-1] + 1]
    assert row.all(), "no hole where two glyphs overlap"
    holed = render(_font(800, hole=True), ["a"], px_per_em=100) < 128
    rows, cols = np.nonzero(holed)
    assert not holed[int(rows.mean()), int(cols.mean())], "the hole inside a letter stays open"
