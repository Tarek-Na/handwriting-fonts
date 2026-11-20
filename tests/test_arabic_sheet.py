"""The Arabic sheet: its layout, its labels, and reading a page back."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_a_box_not_holding_its_form_is_flagged(tmp_path):
    import torch

    from hfont.models.formclass import ClassConfig, FormClassifier

    names = ["uni0628.init", "uni0646.init"]
    model = FormClassifier(ClassConfig(classes=2, width=8))
    with torch.no_grad():                               # a classifier that always says noon
        model.head[1].weight.zero_()
        model.head[1].bias.copy_(torch.tensor([-5.0, 5.0]))
    path = tmp_path / "c.pt"
    torch.save(model.checkpoint(names), path)
    beh, noon, tah = GlyphSpec("ب", Form.INIT), GlyphSpec("ن", Form.INIT), GlyphSpec("ط", Form.INIT)
    ink = np.zeros((128, 200), np.float32)
    ink[60:66, 40:160] = 1
    flags = sheet.check_boxes({beh: ink, noon: ink, tah: ink}, path)
    assert list(flags) == [beh], "beh's box flagged; noon's matches; tah.init is not checked"
    assert flags[beh][1] == "uni0646.init"


def test_a_shadow_over_a_corner_is_neither_a_missing_square_nor_ink():
    from scipy.ndimage import gaussian_filter

    page, layout = sheet.build_sheet()
    page = page.astype(np.float64)
    spec = GlyphSpec("م", Form.FINA)                    # bottom left: under the shadow
    x0, y0, x1, y1 = layout.cell_box(spec)
    page[y0 + 60:y0 + 66, x0 + 100:x0 + 400] = (30, 45, 120)
    h, w = page.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    shade = gaussian_filter(np.where((xx < 0.5 * w) & (yy > 0.6 * h), 0.4, 1.0), 25)
    photo = (page * shade[..., None]).astype(np.uint8)
    read = sheet.read_sheet(photo, layout)
    assert list(read) == [spec], f"only the written box, not the shadow: {[s.name for s in read]}"
    assert 0.8 < read[spec].sum() / (6 * 300) < 1.3


def test_a_photo_stored_sideways_is_turned_upright(tmp_path):
    from PIL import Image

    upright = np.zeros((30, 50, 3), np.uint8)
    upright[:5, :] = 255                                 # a white band along the top
    stored = Image.fromarray(np.rot90(upright, 1).copy())   # as a phone stores it...
    exif = stored.getexif()
    exif[0x0112] = 6                                     # ...with "turn 90 clockwise"
    path = tmp_path / "p.jpg"
    stored.save(path, quality=95, exif=exif)
    photo = sheet.load_photo(path)
    assert photo.shape[:2] == (30, 50)
    assert photo[:5].mean() > 200 and photo[10:].mean() < 50


def test_the_boxes_image_marks_flagged_and_empty_boxes():
    layout = sheet.SheetLayout()
    spec, other = GlyphSpec("ب", Form.MEDI), GlyphSpec("ب", Form.INIT)
    ink = np.zeros((138, 548), np.float32)
    ink[60:66, 100:400] = 1
    image = sheet.boxes_image({spec: ink, other: ink}, {spec: (0.01, "uni0646.medi")}, layout)
    assert image.shape == (len(layout.rows) * 75, 4 * 280, 3)
    x0, y0, _, _ = layout.cell_box(spec)
    gx, gy = layout.grid_origin
    x, y = (x0 - gx) // 2, (y0 - gy) // 2
    assert tuple(image[y + 1, x + 1]) == (220, 30, 30), "flagged: framed red"
    ex0, ey0, _, _ = layout.cell_box(GlyphSpec("ج", Form.MEDI))
    assert tuple(image[(ey0 - gy) // 2 + 20, (ex0 - gx) // 2 + 20]) == (225, 225, 225), "empty: grey"


def test_a_tall_letter_reaching_into_the_label_corner_keeps_its_top():
    page, layout = sheet.build_sheet()
    page = page.copy()
    spec = GlyphSpec("ا")
    x0, y0, x1, y1 = layout.cell_box(spec)
    lx0, _, lx1, _ = layout.label_box(spec)
    col = x0 + lx1 - 18                                  # inside the label's corner, right of its glyph
    page[y0 + 10:y0 + 95, col:col + 5] = (30, 45, 120)   # an alef from near the top of the box
    ink = sheet.read_sheet(page, layout)[spec]
    rows = np.nonzero(ink.any(axis=1))[0]
    assert rows[0] <= 10 - 6 + 2, "its top is read, not blanked with the label's corner"


def test_the_short_sheet_holds_each_letter_once_isolated_four_to_a_row():
    layout = sheet.short_layout()
    specs = layout.specs()
    assert len(specs) == 20 and all(s.form == Form.ISOL for s in specs)
    assert {s.char for s in specs} == {char for char, _ in sheet.ROWS}
    boxes = [layout.cell_box(s) for s in specs]
    assert len(set(boxes)) == 20 and layout.grid_rows == 5
    width, height = layout.size
    assert all(0 <= x0 and x1 <= width and y1 <= height for x0, _, x1, y1 in boxes)
    first, second, fifth = boxes[0], boxes[1], boxes[4]
    assert first[0] > second[0] and first[1] == second[1], "alphabet order runs right to left"
    assert fifth[1] > first[1] and fifth[0] == first[0], "then the next row"


def test_the_short_sheet_reads_back_only_its_written_boxes():
    page, layout = sheet.build_sheet(sheet.short_layout())
    page = page.copy()
    spec = GlyphSpec("ع")
    x0, y0, _, _ = layout.cell_box(spec)
    page[y0 + 60:y0 + 66, x0 + 100:x0 + 400] = (30, 45, 120)
    read = sheet.read_sheet(page, layout)
    assert list(read) == [spec]


# ---------------------------------------------------------------- a real photo (20 November 2026)
# The first real photo of a filled-in short sheet showed three things the simulated photos had
# not: two squares printed grey and lit from the side, red guide lines that photographed a dull
# red (as dark in the red channel as ink), and slivers of line left beside blue strokes by the
# phone's JPEG. Each test below builds that situation from the printed sheet itself.
DULL_RED = (115, 74, 72)            # the guide line as photographed on the real sheet
VIOLET_PEN = (23, 9, 86)            # the real sheet's pen


def _short_page():
    page, layout = sheet.build_sheet(sheet.short_layout())
    page = page.copy()
    red = (page[..., 0] > 150) & (page[..., 1] < 120)              # the printed red, as the sheet draws it
    page[red] = DULL_RED
    return page, layout


def test_dull_red_guide_lines_are_not_read_as_strokes():
    page, layout = _short_page()
    assert sheet.read_sheet(page, layout) == {}, "an empty sheet with dull red lines reads as empty"
    spec = GlyphSpec("ل")
    x0, y0, _, _ = layout.cell_box(spec)
    base = y0 + int(layout.cell_h * layout.baseline)
    page[base - 50:base + 2, x0 + 260:x0 + 265] = VIOLET_PEN        # an upright stroke down to the line
    read = sheet.read_sheet(page, layout)
    assert list(read) == [spec]
    cols = np.nonzero(read[spec].any(axis=0))[0]
    assert cols.max() - cols.min() < 20, "the stroke alone, not the line through the box"


def test_a_stroke_crossing_the_line_stays_whole_and_a_line_under_paper_stays_paper():
    g = np.ones((9, 9))
    g[:, 4] = 0.1                                                    # a stroke, top to bottom
    m = np.zeros((9, 9), bool)
    m[4, :] = True                                                   # the guide line, masked
    g[4, :] = 0.5
    f = sheet.fill_across(g, m)
    assert f[4, 4] == 0.1 and f[4, 0] == 1.0
    g = np.ones((9, 30))
    g[3, :] = 0.1                                                    # a stroke resting on the line...
    m = np.zeros((9, 30), bool)
    m[4, :] = True                                                   # ...the line just under it
    assert (sheet.fill_across(g, m)[4] == 1.0).all(), "the line under a stroke is not thickened into it"
    g = np.ones((9, 30))
    g[4, :] = 0.1
    m = np.zeros((9, 30), bool)
    m[4, 12:15] = True                                               # a short break along the line
    assert (sheet.fill_across(g, m, short=5)[4, 12:15] == 0.1).all(), "a stroke along the line is not broken"


def test_a_sliver_of_line_is_dropped_but_a_dot_and_the_letter_stay():
    ink = np.zeros((100, 200), np.float32)
    ink[20:60, 100:104] = 1                                          # the letter
    ink[78:81, 60:110] = 1                                           # a sliver on the line (row 79)
    ink[85:90, 90:95] = 1                                            # a dot just under the line
    out = sheet.drop_line_slivers(ink, line_row=79)
    assert out[78:81, 60:90].sum() == 0 and out[85:90, 90:95].all() and out[20:60, 100:104].all()


def test_faint_squares_are_still_found_and_a_dark_table_is_not_taken_for_one(monkeypatch):
    """The real photo's squares sat at about 0.57 of the paper around them, above the old cut
    (0.55): with only that cut they are not found; with the looser cuts they are, in place."""
    page, layout = _short_page()
    photo = np.full((page.shape[0] + 200, page.shape[1] + 200, 3), 60, np.uint8)   # a dark table around it
    photo[100:-100, 100:-100] = page
    square = np.zeros(photo.shape[:2], bool)
    for cx, cy in layout.marker_centers() + 100:
        half = layout.marker // 2
        square[int(cy) - half:int(cy) + half + 1, int(cx) - half:int(cx) + half + 1] = True
    photo[square] = 150                                              # printed grey: 0.59 of the paper
    monkeypatch.setattr(sheet, "MARKER_CUTS", (0.55,))
    with pytest.raises(sheet.TemplateError):
        sheet.find_markers_local(photo, layout)
    monkeypatch.undo()
    found = sheet.find_markers_local(photo, layout)
    assert np.abs(found - (layout.marker_centers() + 100)).max() < 3
