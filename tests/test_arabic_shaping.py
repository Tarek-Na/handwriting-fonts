"""Gate 4: Arabic export works before any Arabic model exists.

Builds a font of placeholder shapes, gives it GSUB tables generated from glyph
names alone, and checks with HarfBuzz — the shaper behind Chrome, Firefox,
Android and LibreOffice — that real Arabic words select the right positional
form of every letter and take the mandatory lam-alef ligature.

Each placeholder is a baseline bar plus a stem, with the bar carried to the
edge of the glyph on whichever side that form joins. Arabic runs right to left,
so a form that connects to the previous letter reaches the right edge and one
that connects to the next letter reaches the left. A correctly shaped word
therefore renders as one unbroken bar — the crude beginning of the seam metric
Phase 2 needs — and a wrongly chosen form shows up as a gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("uharfbuzz")

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.fontbuild.arabic import add_arabic_shaping  # noqa: E402

BEH, TEH, NOON, MEEM, LAM = "ب", "ت", "ن", "م", "ل"
ALEF, DAL = "ا", "د"
DUAL_JOINING = [BEH, TEH, NOON, MEEM, LAM]
RIGHT_JOINING = [ALEF, DAL]  # join to the previous letter only

UPEM, ADVANCE, BAR = 1000, 500, (0, 80)  # bar spans y 0..80


def _name(char: str, form: Form = Form.ISOL) -> str:
    return GlyphSpec(char, form).name


def _outline(pen, joins_prev: bool, joins_next: bool, stem_h: int) -> None:
    """Bar reaching the edges it joins on, plus a stem to tell glyphs apart."""
    right = ADVANCE if joins_prev else ADVANCE - 60
    left = 0 if joins_next else 60
    y0, y1 = BAR
    for x0, x1, top in ((left, right, y1), (220, 280, stem_h)):
        pen.moveTo((x0, y0))
        pen.lineTo((x1, y0))
        pen.lineTo((x1, top))
        pen.lineTo((x0, top))
        pen.closePath()


@pytest.fixture(scope="module")
def arabic_font(tmp_path_factory):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.ttLib import TTFont

    glyphs: dict[str, tuple[bool, bool]] = {}  # name -> (joins_prev, joins_next)
    cmap = {}
    for char in DUAL_JOINING:
        cmap[ord(char)] = _name(char)
        glyphs[_name(char)] = (False, False)
        glyphs[_name(char, Form.INIT)] = (False, True)
        glyphs[_name(char, Form.MEDI)] = (True, True)
        glyphs[_name(char, Form.FINA)] = (True, False)
    for char in RIGHT_JOINING:
        cmap[ord(char)] = _name(char)
        glyphs[_name(char)] = (False, False)
        glyphs[_name(char, Form.FINA)] = (True, False)
    glyphs["lam_alef"] = (False, False)
    glyphs["lam_alef.fina"] = (True, False)
    glyphs["space"] = None

    order = [".notdef"] + sorted(glyphs)
    cmap[0x20] = "space"
    fb = FontBuilder(UPEM, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)

    charstrings, metrics = {}, {}
    for i, name in enumerate(order):
        pen = T2CharStringPen(ADVANCE, None)
        if glyphs.get(name):
            _outline(pen, *glyphs[name], stem_h=300 + 20 * i)
        charstrings[name] = pen.getCharString()
        metrics[name] = (ADVANCE, 0)
    fb.setupCFF("ArabicTest-Regular", {"FullName": "Arabic Test"}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Arabic Test", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()

    add_arabic_shaping(fb.font, lam=_name(LAM),
                       alefs={_name(ALEF): "lam_alef"})
    path = tmp_path_factory.mktemp("arabic") / "arabic_test.otf"
    fb.save(str(path))
    return path, TTFont(str(path)).getGlyphOrder()


def _shape(font, text: str) -> list[str]:
    """Glyph names in *logical* (reading) order."""
    from hfont.evaluate.shaping import shape_text

    path, order = font
    shaped = shape_text(path, text)
    # HarfBuzz returns right-to-left runs in visual order; clusters restore the
    # reading order the assertions are written in.
    shaped.sort(key=lambda g: g[1])
    return [order[gid] for gid, _, _ in shaped]


@pytest.mark.parametrize(
    "word, expected",
    [
        # beh-noon-teh: initial, medial, final.
        (BEH + NOON + TEH, [(BEH, Form.INIT), (NOON, Form.MEDI), (TEH, Form.FINA)]),
        # A single letter stands alone.
        (MEEM, [(MEEM, Form.ISOL)]),
        # Alef joins only backwards, so the beh after it starts over as isolated.
        (BEH + ALEF + BEH, [(BEH, Form.INIT), (ALEF, Form.FINA), (BEH, Form.ISOL)]),
        # Dal likewise breaks the connection; meem then begins a new word-part.
        (MEEM + DAL + MEEM + NOON,
         [(MEEM, Form.INIT), (DAL, Form.FINA), (MEEM, Form.INIT), (NOON, Form.FINA)]),
    ],
)
def test_positional_forms(arabic_font, word, expected):
    assert _shape(arabic_font, word) == [_name(c, f) for c, f in expected]


def test_lam_alef_is_ligated(arabic_font):
    """Lam followed by alef must become one glyph, in both contexts."""
    assert _shape(arabic_font, LAM + ALEF) == ["lam_alef"]
    # After a joining letter the ligature takes its final form.
    assert _shape(arabic_font, BEH + LAM + ALEF) == [_name(BEH, Form.INIT), "lam_alef.fina"]


def test_isolated_forms_only_in_cmap(arabic_font):
    """Contextual forms must be reachable only through GSUB."""
    from fontTools.ttLib import TTFont

    cmap = TTFont(str(arabic_font[0])).getBestCmap()
    assert not [n for n in cmap.values() if "." in n], "a contextual form is in the cmap"


def _bar_row(image: np.ndarray) -> np.ndarray:
    """The rendered baseline bar's row: the lowest row with ink across the word."""
    rows = np.nonzero((image > 0.5).sum(axis=1) > image.shape[1] * 0.3)[0]
    return image[rows[-1]] > 0.5


def _gaps(row: np.ndarray) -> int:
    """Number of breaks inside the inked span of a row."""
    ink = np.nonzero(row)[0]
    span = row[ink[0]: ink[-1] + 1]
    return int(np.count_nonzero(np.diff(span.astype(int)) == -1))


def test_joined_word_renders_without_seams(arabic_font):
    """A fully joined word must draw as one continuous baseline."""
    from hfont.evaluate.shaping import render_string

    image = render_string(arabic_font[0], BEH + NOON + MEEM + TEH, size=48)
    assert _gaps(_bar_row(image)) == 0, "joined letters left a seam"


def test_non_joiner_breaks_the_line(arabic_font):
    """...and a word with a non-joining letter must break exactly once."""
    from hfont.evaluate.shaping import render_string

    image = render_string(arabic_font[0], BEH + ALEF + BEH, size=48)
    assert _gaps(_bar_row(image)) == 1
