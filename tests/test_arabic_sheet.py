"""The Arabic sheet: its layout, its labels, and reading a page back."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont import arabic_sheet as sheet  # noqa: E402
from hfont.charset import Form, GlyphSpec  # noqa: E402


def test_every_form_has_one_box_on_one_a4_page():
    layout = sheet.SheetLayout()
    specs = sheet.sheet_specs()
    assert len(specs) == 66 and len(set(specs)) == 66
    width, height = layout.size
    assert width <= 2480 and height <= 3508, "fits A4 at 300 dpi"
    boxes = [layout.cell_box(s) for s in specs]
    assert len(set(boxes)) == len(boxes), "no two forms share a box"
    isol, fina = layout.cell_box(GlyphSpec("ب")), layout.cell_box(GlyphSpec("ب", Form.FINA))
    assert isol[0] > fina[0], "isolated on the right, as Arabic reads"


def test_labels_are_the_shaped_forms():
    assert sheet.presentation_form(GlyphSpec("ب", Form.INIT)) == "ﺑ"
    assert sheet.presentation_form(GlyphSpec("ه", Form.MEDI)) == "ﻬ"
    assert sheet.presentation_form(GlyphSpec("ﻻ")) == "ﻻ"


def test_writing_in_a_box_is_read_back_and_empty_boxes_are_not():
    page, layout = sheet.build_sheet()
    page = page.copy()
    spec = GlyphSpec("ب", Form.MEDI)
    x0, y0, x1, y1 = layout.cell_box(spec)
    page[y0 + 60:y0 + 66, x0 + 100:x0 + 400] = (30, 45, 120)          # a blue stroke
    page[y0 + 30:y0 + 66, x0 + 250:x0 + 256] = (30, 45, 120)
    read = sheet.read_sheet(page, layout)
    assert list(read) == [spec], "only the box written in"
    ink = read[spec]
    assert 0.8 < ink.sum() / (6 * 300 + 6 * 36) < 1.2
