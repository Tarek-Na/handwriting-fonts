"""The Arabic handwriting sheet: one printed page, every letter form in its own box.

A writer fills in one A4 page and photographs it; ``read_sheet`` returns one
ink image per letter form, framed like the AHAWP scans (a cell about 130 px
high at 300 dpi), so the joining-font builder (``scripts/ahawp_font.py``)
takes them as they are.

Layout: a row per letter, in alphabetical order, with the forms right to
left as Arabic reads: isolated, initial, medial, final. A form the letter
does not have (dal joins only on its right) is left out. The sheet holds
the 18 letter bodies AHAWP holds (the other ten letters are the same
bodies with other dots, composed by ``fontbuild.rasm``), plus tah's initial
form, alef with hamza and lam-alef: 66 boxes.

It reuses the Latin template's tricks (``hfont.intake``): four black squares
fix the perspective, and everything printed in the box is red, so reading
only the red channel leaves the writer's dark ink and nothing else. Each
box is labelled with the form to write, as its Unicode presentation-form
character (``ﺑ`` for initial beh), so no text shaping is needed to print it.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .charset import Form, GlyphSpec
from .intake import INK_DROPOUT_RGB, TemplateError, _label_font, find_markers

#: The letters in the order of the alphabet, with the forms each is written in.
ROWS: list[tuple[str, tuple[Form, ...]]] = [
    ("ا", (Form.ISOL, Form.FINA)),
    ("أ", (Form.ISOL,)),
    ("ب", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ج", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("د", (Form.ISOL, Form.FINA)),
    ("ر", (Form.ISOL, Form.FINA)),
    ("س", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ص", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ط", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ع", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ف", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ق", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ك", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ل", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ﻻ", (Form.ISOL,)),
    ("م", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ن", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("ه", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
    ("و", (Form.ISOL, Form.FINA)),
    ("ي", (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)),
]
#: Column of each form, counted from the right.
COLUMN = {Form.ISOL: 0, Form.INIT: 1, Form.MEDI: 2, Form.FINA: 3}
_FORM_WORD = {Form.ISOL: "ISOLATED", Form.INIT: "INITIAL", Form.MEDI: "MEDIAL", Form.FINA: "FINAL"}


def presentation_form(spec: GlyphSpec) -> str:
    """The Unicode character showing ``spec`` already shaped (``ﺑ`` for beh.init)."""
    if len(spec.char) == 1 and 0xFB50 <= ord(spec.char) <= 0xFEFF:     # already one (lam-alef)
        return spec.char
    base = unicodedata.name(spec.char)
    return unicodedata.lookup(f"{base} {_FORM_WORD[spec.form]} FORM")


def sheet_specs() -> list[GlyphSpec]:
    return [GlyphSpec(char, form) for char, forms in ROWS for form in forms]


@dataclass(frozen=True)
class SheetLayout:
    """Page geometry in pixels at 300 dpi (A4: 2480 x 3508)."""

    cell_w: int = 560
    cell_h: int = 150
    margin: int = 120
    header: int = 200
    marker: int = 56
    baseline: float = 0.62          # of the box height, from its top
    label_size: int = 40
    dpi: int = 300
    rows: tuple = field(default_factory=lambda: tuple(ROWS))

    @property
    def grid_origin(self) -> tuple[int, int]:
        return self.margin, self.margin + self.header

    @property
    def size(self) -> tuple[int, int]:
        return (2 * self.margin + 4 * self.cell_w,
                2 * self.margin + self.header + len(self.rows) * self.cell_h)

    def cell_box(self, spec: GlyphSpec) -> tuple[int, int, int, int]:
        row = next(i for i, (char, forms) in enumerate(self.rows) if char == spec.char and spec.form in forms)
        gx, gy = self.grid_origin
        x0 = gx + (3 - COLUMN[spec.form]) * self.cell_w
        y0 = gy + row * self.cell_h
        return x0, y0, x0 + self.cell_w, y0 + self.cell_h

    def marker_centers(self) -> np.ndarray:
        """TL, TR, BR, BL registration squares, (x, y), as ``intake.TemplateLayout``."""
        gx, gy = self.grid_origin
        gw, gh = 4 * self.cell_w, len(self.rows) * self.cell_h
        off = self.marker
        return np.array([[gx - off, gy - off], [gx + gw + off, gy - off],
                         [gx + gw + off, gy + gh + off], [gx - off, gy + gh + off]], np.float64)

    def label_box(self, spec: GlyphSpec) -> tuple[int, int, int, int]:
        """Where the label sits in its box (top right), relative to the box."""
        w, h = int(self.label_size * 1.6), int(self.label_size * 1.5)
        return self.cell_w - w, 0, self.cell_w, h


def build_sheet(layout: SheetLayout | None = None) -> tuple[np.ndarray, SheetLayout]:
    """The printable page as an RGB image, and its layout."""
    from PIL import Image, ImageDraw

    layout = layout or SheetLayout()
    width, height = layout.size
    page = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    red = INK_DROPOUT_RGB
    draw.text((layout.margin, layout.margin // 2),
              "Write each letter form in its box, resting on the red line.",
              fill=(0, 0, 0), font=_label_font(40))
    draw.text((layout.margin, layout.margin // 2 + 56),
              "Dark pen (blue or black). Print in COLOUR. Photograph flat with all four black squares in view.",
              fill=(60, 60, 60), font=_label_font(26))
    draw.text((layout.margin, layout.margin // 2 + 96),
              "Joined forms: write the connecting stroke out to the edge of the box on the side it joins.",
              fill=(60, 60, 60), font=_label_font(26))
    half = layout.marker // 2
    for cx, cy in layout.marker_centers():
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(0, 0, 0))
    font = _label_font(layout.label_size)
    for spec in sheet_specs():
        x0, y0, x1, y1 = layout.cell_box(spec)
        draw.rectangle([x0, y0, x1, y1], outline=red, width=2)
        base = y0 + int(layout.cell_h * layout.baseline)
        draw.line([x0 + 8, base, x1 - 8, base], fill=red, width=3)
        lx0, ly0, _, _ = layout.label_box(spec)
        draw.text((x0 + lx0 + 6, y0 + ly0 + 2), presentation_form(spec), fill=red, font=font)
    return np.asarray(page), layout


def save_sheet(path: str | Path, layout: SheetLayout | None = None) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image, layout = build_sheet(layout)
    Image.fromarray(image).save(path, dpi=(layout.dpi, layout.dpi))
    return path


def find_markers_local(photo_rgb: np.ndarray) -> np.ndarray:
    """The four registration squares, found against their surroundings, not the whole photo.

    ``intake.find_markers`` cuts dark from light with one threshold for the
    whole photo, so a square under a hand's shadow sinks into the shadow and
    is not found. Here each pixel is compared with the light around it (a
    wide blur), which a shadow darkens as much as the square: the square
    stays far darker than its surroundings. The same size, solidity and
    squareness tests then pick the squares.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.measure import label, regionprops

    gray = photo_rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    h, w = gray.shape
    surround = gaussian_filter(gray, 0.04 * min(h, w)) + 1.0
    dark = gray / surround < 0.55
    min_area, max_area = (min(h, w) * 0.012) ** 2, (min(h, w) * 0.12) ** 2
    candidates = []
    for region in regionprops(label(dark)):
        if not (min_area <= region.area <= max_area):
            continue
        minr, minc, maxr, maxc = region.bbox
        if region.solidity < 0.9 or not (0.6 < (maxc - minc) / max(maxr - minr, 1) < 1.66):
            continue
        cy, cx = region.centroid
        candidates.append((cx, cy))
    if len(candidates) < 4:
        raise TemplateError(f"found {len(candidates)} registration squares, need 4. "
                            "Is the whole page in frame?")
    pts = np.asarray(candidates)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    chosen = np.asarray([pts[np.argmin(np.linalg.norm(pts - c, axis=1))] for c in corners])
    if len({tuple(p) for p in chosen}) < 4:
        raise TemplateError("registration squares are ambiguous; retake the photo straighter")
    return chosen


def flatten(photo_rgb: np.ndarray, layout: SheetLayout) -> np.ndarray:
    """The photo's red channel, perspective removed, at the page's own size (1 = paper)."""
    from skimage.transform import ProjectiveTransform, warp

    try:
        found = find_markers(photo_rgb)
    except TemplateError:
        found = find_markers_local(photo_rgb)            # a shadow over a square
    transform = ProjectiveTransform()
    if not transform.estimate(layout.marker_centers(), found):
        raise TemplateError("could not fit a perspective transform to the markers")
    red = photo_rgb[..., 0].astype(np.float64)
    if red.max() > 1.5:
        red /= 255.0
    width, height = layout.size
    return even_light(warp(red, transform, output_shape=(height, width), order=1, cval=1.0))


def even_light(page: np.ndarray, size: int = 25) -> np.ndarray:
    """The page divided by its paper's brightness around each point, so a shadow does not read as ink.

    The paper is what is left when strokes (narrower than ``size`` px) are
    closed over: a grey closing, smoothed. Where a shadow's edge crosses a
    box, one threshold for the box would take the shadowed half for ink.
    """
    from scipy.ndimage import gaussian_filter, grey_closing

    paper = gaussian_filter(grey_closing(page, size=(size, size)), size / 2)
    return np.clip(page / np.maximum(paper, 0.05), 0.0, 1.0)


def cell_ink(crop: np.ndarray, min_piece: int = 6, contrast: float = 0.25) -> np.ndarray:
    """A box's writing as binary ink (1 = pen); all zero when the box is empty.

    Ink is what is darker than the paper by ``contrast`` of the paper's
    brightness. A box with no such pixels, or only specks, is empty.
    """
    from scipy.ndimage import label

    paper = float(np.percentile(crop, 90))
    dark = crop < paper * (1 - contrast)
    pieces, n = label(dark, np.ones((3, 3), int))
    sizes = np.bincount(pieces.ravel())
    for i in range(1, n + 1):
        if sizes[i] < min_piece:
            dark[pieces == i] = False
    return dark.astype(np.float32) if dark.sum() >= 20 else np.zeros(crop.shape, np.float32)


def read_sheet(photo_rgb: np.ndarray, layout: SheetLayout | None = None,
               inset: int = 6) -> dict[GlyphSpec, np.ndarray]:
    """One ink image per box of a photographed sheet (empty boxes left out).

    Each is the box less ``inset`` px all round, with the label corner
    blanked (a faded or monochrome print may not drop out completely).
    """
    layout = layout or SheetLayout()
    flat = flatten(photo_rgb, layout)
    out = {}
    for spec in sheet_specs():
        x0, y0, x1, y1 = layout.cell_box(spec)
        crop = flat[y0 + inset:y1 - inset, x0 + inset:x1 - inset].copy()
        lx0, ly0, lx1, ly1 = layout.label_box(spec)
        crop[max(ly0 - inset, 0):ly1 - inset, max(lx0 - inset, 0):lx1 - inset] = float(np.percentile(crop, 90))
        ink = cell_ink(crop)
        if ink.any():
            out[spec] = ink
    return out


#: A box is flagged when the form classifier gives its label less than this
#: (``scripts/train_formclass.py``: about 2 correct boxes a sheet flagged, 9% of
#: boxes holding the most alike other form passed, on the AHAWP val writers).
FLAG = 0.2


def check_boxes(letters: dict[GlyphSpec, np.ndarray], checkpoint: str | Path) -> dict[GlyphSpec, tuple[float, str]]:
    """Boxes whose writing does not look like their label: spec -> (p(label), likeliest form name).

    Uses the form classifier (``hfont.models.formclass``). A form it was not
    trained on (tah's initial form) is not checked.
    """
    import torch

    from .models.formclass import FormClassifier
    from .models.formgen import to_canvas

    model, names = FormClassifier.from_checkpoint(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    specs = [s for s in letters if s.name in names]
    if not specs:
        return {}
    x = torch.from_numpy(np.stack([to_canvas(letters[s]) for s in specs])).float()[:, None]
    with torch.no_grad():
        prob = torch.softmax(model(x), 1).numpy()
    flagged = {}
    for spec, p in zip(specs, prob):
        own = float(p[names.index(spec.name)])
        if own < FLAG:
            flagged[spec] = (own, names[int(p.argmax())])
    return flagged
