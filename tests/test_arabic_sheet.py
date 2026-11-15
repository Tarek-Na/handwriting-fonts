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
