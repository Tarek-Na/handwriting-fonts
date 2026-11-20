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

``short_layout()`` is the 20-box sheet: only the isolated letters, four to a
row. The joined forms then come from the variation model
(``scripts/variation.py``), not from the writer.

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
    wrap: bool = False              # boxes four to a row in order, not a row per letter

    def specs(self) -> list[GlyphSpec]:
        return [GlyphSpec(char, form) for char, forms in self.rows for form in forms]

    @property
    def grid_rows(self) -> int:
        return -(-len(self.specs()) // 4) if self.wrap else len(self.rows)

    @property
    def grid_origin(self) -> tuple[int, int]:
        return self.margin, self.margin + self.header

    @property
    def size(self) -> tuple[int, int]:
        return (2 * self.margin + 4 * self.cell_w,
                2 * self.margin + self.header + self.grid_rows * self.cell_h)

    def cell_box(self, spec: GlyphSpec) -> tuple[int, int, int, int]:
        """The box's page pixels; ValueError (wrapped) or StopIteration for a form not on the sheet."""
        if self.wrap:
            row, column = divmod(self.specs().index(spec), 4)
        else:
            row = next(i for i, (char, forms) in enumerate(self.rows) if char == spec.char and spec.form in forms)
            column = COLUMN[spec.form]
        gx, gy = self.grid_origin
        x0 = gx + (3 - column) * self.cell_w
        y0 = gy + row * self.cell_h
        return x0, y0, x0 + self.cell_w, y0 + self.cell_h

    def marker_centers(self) -> np.ndarray:
        """TL, TR, BR, BL registration squares, (x, y), as ``intake.TemplateLayout``."""
        gx, gy = self.grid_origin
        gw, gh = 4 * self.cell_w, self.grid_rows * self.cell_h
        off = self.marker
        return np.array([[gx - off, gy - off], [gx + gw + off, gy - off],
                         [gx + gw + off, gy + gh + off], [gx - off, gy + gh + off]], np.float64)

    def label_box(self, spec: GlyphSpec) -> tuple[int, int, int, int]:
        """Where the label sits in its box (top right), relative to the box."""
        w, h = int(self.label_size * 1.6), int(self.label_size * 1.5)
        return self.cell_w - w, 0, self.cell_w, h


def short_layout() -> SheetLayout:
    """The 20-box sheet: each letter once, isolated (the 18 bodies, alef with hamza, lam-alef)."""
    return SheetLayout(rows=tuple((char, (Form.ISOL,)) for char, _ in ROWS), wrap=True, header=160)


def build_sheet(layout: SheetLayout | None = None) -> tuple[np.ndarray, SheetLayout]:
    """The printable page as an RGB image, and its layout."""
    from PIL import Image, ImageDraw

    layout = layout or SheetLayout()
    width, height = layout.size
    page = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    red = INK_DROPOUT_RGB
    draw.text((layout.margin, layout.margin // 2),
              "Write each letter on its own, as you write it alone, resting on the red line."
              if layout.wrap else "Write each letter form in its box, resting on the red line.",
              fill=(0, 0, 0), font=_label_font(40))
    draw.text((layout.margin, layout.margin // 2 + 56),
              "Dark pen (blue or black). Print in COLOUR. Photograph flat with all four black squares in view.",
              fill=(60, 60, 60), font=_label_font(26))
    if not layout.wrap:
        draw.text((layout.margin, layout.margin // 2 + 96),
                  "Joined forms: write the connecting stroke out to the edge of the box on the side it joins.",
                  fill=(60, 60, 60), font=_label_font(26))
    half = layout.marker // 2
    for cx, cy in layout.marker_centers():
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(0, 0, 0))
    font = _label_font(layout.label_size)
    for spec in layout.specs():
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


#: How dark a registration square must be against the paper around it, tried in turn: a square
#: printed faint or lit from the side (a real photo: 0.57 of its surroundings) needs a looser cut.
MARKER_CUTS = (0.55, 0.65, 0.75)


def _markers_plausible(chosen: np.ndarray, areas: np.ndarray, layout: SheetLayout | None) -> bool:
    """Do four blobs look like the sheet's squares: alike in size, and at the corners of a
    quadrilateral shaped like the layout's (allowing for perspective)?"""
    if len({tuple(p) for p in chosen}) < 4 or areas.max() > 4.0 * areas.min():
        return False
    tl, tr, br, bl = chosen
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if min(width, height) < 1:
        return False
    if layout is not None:
        m = layout.marker_centers()
        expected = np.linalg.norm(m[1] - m[0]) / np.linalg.norm(m[3] - m[0])
        if not 0.65 < (width / height) / expected < 1.5:
            return False
    cross = [np.cross(chosen[(i + 1) % 4] - chosen[i], chosen[(i + 2) % 4] - chosen[(i + 1) % 4]) for i in range(4)]
    return all(c > 0 for c in cross) or all(c < 0 for c in cross)      # convex, in corner order


def find_markers_local(photo_rgb: np.ndarray, layout: SheetLayout | None = None) -> np.ndarray:
    """The four registration squares, found against their surroundings, not the whole photo.

    ``intake.find_markers`` cuts dark from light with one threshold for the
    whole photo, so a square under a hand's shadow sinks into the shadow and
    is not found. Here each pixel is compared with the light around it (a
    wide blur), which a shadow darkens as much as the square: the square
    stays far darker than its surroundings. The same size, solidity and
    squareness tests then pick the squares. A square printed faint, or lit
    from the side, is only somewhat darker than its paper, so looser cuts
    (``MARKER_CUTS``) are tried in turn; a set is taken only when the four
    blobs are alike in size and lie at the corners of a quadrilateral shaped
    like the sheet's (``layout``), so a looser cut cannot pick a dark patch of
    the table instead.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.measure import label, regionprops

    gray = photo_rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    h, w = gray.shape
    ratio = gray / (gaussian_filter(gray, 0.04 * min(h, w)) + 1.0)
    min_area, max_area = (min(h, w) * 0.012) ** 2, (min(h, w) * 0.12) ** 2
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    found = 0
    for cut in MARKER_CUTS:
        candidates = []
        for region in regionprops(label(ratio < cut)):
            if not (min_area <= region.area <= max_area):
                continue
            minr, minc, maxr, maxc = region.bbox
            if region.solidity < 0.9 or not (0.6 < (maxc - minc) / max(maxr - minr, 1) < 1.66):
                continue
            cy, cx = region.centroid
            candidates.append((cx, cy, region.area))
        found = max(found, len(candidates))
        if len(candidates) < 4:
            continue
        pts = np.asarray(candidates)
        pick = [int(np.argmin(np.linalg.norm(pts[:, :2] - c, axis=1))) for c in corners]
        if _markers_plausible(pts[pick, :2], pts[pick, 2], layout):
            return np.asarray([_refine_square(gray, x, y, area) for x, y, area in pts[pick]])
    raise TemplateError(f"found {found} registration squares, need 4. Is the whole page in frame?")


def _refine_square(gray: np.ndarray, x: float, y: float, area: float) -> tuple[float, float]:
    """The centre of the square found near (x, y), measured again in a window around it: a loose
    cut, or dark surroundings, can leave only part of a square found, and its centre off by many
    pixels. The window is cut halfway between the square's own grey (at its found centre) and
    the paper's (the window's bright end), so a darker table beside it is a separate piece."""
    from skimage.measure import label

    r = int(2.5 * np.sqrt(area)) + 2                     # the whole square, even from a partial find
    y0, x0 = max(int(y) - r, 0), max(int(x) - r, 0)
    win = gray[y0:int(y) + r + 1, x0:int(x) + r + 1]
    if win.size < 16:
        return x, y
    cy, cx = min(int(y) - y0, win.shape[0] - 1), min(int(x) - x0, win.shape[1] - 1)
    own = float(np.median(win[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]))
    paper = float(np.percentile(win, 90))
    if paper - own < 10:
        return x, y
    pieces = label(win < (own + paper) / 2)
    piece = pieces[cy, cx]
    if piece == 0:
        return x, y
    rows, cols = np.nonzero(pieces == piece)
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    if not (0.6 < w / h < 1.66 and rows.size >= 0.8 * h * w and rows.size >= 0.5 * area):
        return x, y                                      # not a solid square: keep the first guess
    if rows.size > 0.9 * win.size:
        return x, y
    return x0 + cols.mean(), y0 + rows.mean()


#: A pixel whose red exceeds its green and blue by this share of its red is the printed red, not ink.
RED_HUE = 0.15


def printed_red(rgb: np.ndarray) -> np.ndarray:
    """Where the photo shows the sheet's red print (guide lines, labels), as a boolean mask.

    Reading the red channel drops the red print only when it photographs bright. A real print
    under room light came out a dull red, about half the paper's brightness in red (115-144
    against 216), and every guide line was read as a stroke through its letter. Its hue still
    gives it away: red well above green and blue. Blue, violet and black pens are not; where a
    pen crosses a line, the pen's colour dominates the mix and the stroke stays.
    """
    return hue_redness(rgb) > RED_HUE


def hue_redness(rgb: np.ndarray) -> np.ndarray:
    """How much red exceeds green and blue, as a share of red: about 0.3 on the printed guides,
    near 0 on paper and black ink, below 0 on blue ink."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r - np.maximum(g, b)) / np.maximum(r, 1e-3)


def _nearest_unmasked(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each pixel, the row of the nearest unmasked pixel above it and below it (its own
    row when unmasked; the edge when there is none)."""
    h = mask.shape[0]
    rows = np.arange(h)[:, None].repeat(mask.shape[1], 1)
    above = np.maximum.accumulate(np.where(mask, 0, rows), axis=0)
    below = (h - 1 - np.maximum.accumulate(np.where(mask, 0, h - 1 - rows)[::-1], axis=0))[::-1]
    return above, below


def fill_across(gray: np.ndarray, mask: np.ndarray, short: int | None = None) -> np.ndarray:
    """``gray`` with each masked pixel (a guide line) filled from the pixels beside it.

    Up and down: the lighter of the nearest unmasked pixels above and below, so ink only where
    both are ink: a pen stroke crossing the line joins up again, while under a stroke that rests
    on the line (ink above, paper below) the line stays paper. Along the line: a masked run of at
    most ``short`` px (default 0.6% of the photo's short side) takes the lighter of its two ends,
    so a stroke that runs along the line for a moment (the bottom of a bowl, on a real sheet) is
    not broken there; longer runs (the line between two letters) are left to the up-and-down
    fill. Either way giving ink is enough.
    """
    h, w = gray.shape
    short = short if short is not None else max(4, int(0.006 * min(h, w)))
    above, below = _nearest_unmasked(mask)
    cols = np.arange(w)[None, :].repeat(h, 0)
    vertical = np.maximum(gray[above, cols], gray[below, cols])
    left, right = (a.T for a in _nearest_unmasked(mask.T))
    rows = np.arange(h)[:, None].repeat(w, 1)
    along = np.where(right - left - 1 <= short, np.maximum(gray[rows, left], gray[rows, right]), np.inf)
    return np.where(mask, np.minimum(vertical, along), gray)


def flatten(photo_rgb: np.ndarray, layout: SheetLayout) -> np.ndarray:
    """The photo's red channel, perspective removed, at the page's own size (1 = paper)."""
    from skimage.transform import ProjectiveTransform, warp

    try:
        found = find_markers(photo_rgb)
    except TemplateError:
        found = find_markers_local(photo_rgb, layout)    # a shadow over a square, or a faint one
    transform = ProjectiveTransform()
    if not transform.estimate(layout.marker_centers(), found):
        raise TemplateError("could not fit a perspective transform to the markers")
    rgb = photo_rgb[..., :3].astype(np.float64)
    if rgb.max() > 1.5:
        rgb /= 255.0
    # The guides, wherever they print dark. Along the line only on the short sheet: there every
    # letter stands alone, and a real stroke running along the line was broken without it (sad,
    # qaf, noon); on the full sheet joined forms' loops sit on the line and would be closed.
    red = fill_across(rgb[..., 0], printed_red(rgb), short=None if layout.wrap else 0)
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


def label_masks(layout: SheetLayout, grow: int = 3) -> dict[GlyphSpec, np.ndarray]:
    """Each box's printed label as a mask over the box (page pixels), widened by ``grow`` px."""
    from scipy.ndimage import binary_dilation

    page, _ = build_sheet(layout)
    printed = page[..., 1] < 150                      # the red label is dark in green, paper is not
    masks = {}
    for spec in layout.specs():
        x0, y0, x1, y1 = layout.cell_box(spec)
        lx0, ly0, lx1, ly1 = layout.label_box(spec)
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        mask[ly0 + 3:ly1, lx0:lx1 - 3] = printed[y0 + ly0 + 3:y0 + ly1, x0 + lx0:x0 + lx1 - 3]
        masks[spec] = binary_dilation(mask, iterations=grow)
    return masks


def drop_line_slivers(ink: np.ndarray, line_row: float, near: int = 8, thick: int = 6) -> np.ndarray:
    """``ink`` without detached thin, flat pieces lying on the box's writing line (``line_row``).

    Beside a pen stroke a phone's JPEG mixes the stroke's colour into the red line, which then no
    longer looks red and survives as a short bar under or beside the letter (a real sheet: under
    dal and seen). Such a piece is at most ``thick`` px tall, at least 3 times as wide, and lies
    within ``near`` px of the line. A dot is compact and a letter's own stroke is joined to it, so
    neither is taken. Only on the short sheet: on the full sheet a joined form's connecting stroke
    runs along the line and can be a piece of its own (a medial yeh is mostly that stroke), and
    without the fill along the line a waw's tail lying on the line can be one too.
    """
    from scipy.ndimage import label

    pieces, n = label(np.asarray(ink) > 0.5, np.ones((3, 3), int))
    if n < 2:
        return ink
    out = np.asarray(ink).copy()
    sizes = np.bincount(pieces.ravel())
    biggest = int(np.argmax(sizes[1:])) + 1
    for i in range(1, n + 1):
        if i == biggest:
            continue
        rows, cols = np.nonzero(pieces == i)
        h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        if h <= thick and w >= 3 * h and abs(rows.mean() - line_row) <= near:
            out[pieces == i] = 0
    return out


def read_sheet(photo_rgb: np.ndarray, layout: SheetLayout | None = None,
               inset: int = 6) -> dict[GlyphSpec, np.ndarray]:
    """One ink image per box of a photographed sheet (empty boxes left out).

    Each is the box less ``inset`` px all round. The label's own strokes are
    blanked (a faded or monochrome print may not drop out completely), not
    its whole corner: a writer starting at the right edge, as Arabic is
    written, may reach into it with a tall alef or lam.
    """
    layout = layout or SheetLayout()
    flat = flatten(photo_rgb, layout)
    masks = label_masks(layout)
    out = {}
    for spec in layout.specs():
        x0, y0, x1, y1 = layout.cell_box(spec)
        crop = flat[y0 + inset:y1 - inset, x0 + inset:x1 - inset].copy()
        crop[masks[spec][inset:-inset, inset:-inset]] = float(np.percentile(crop, 90))
        ink = cell_ink(crop)
        if layout.wrap:                                  # the short sheet: letters alone (see fill_across in flatten)
            ink = drop_line_slivers(ink, layout.cell_h * layout.baseline - inset)
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


def load_photo(path: str | Path) -> np.ndarray:
    """A photo as RGB, turned upright as the phone recorded (EXIF orientation).

    Phones often store a portrait photo sideways with a tag saying how to
    turn it; read as stored, the page would be on its side.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def read_upright(photo_rgb: np.ndarray, checkpoint: str | Path | None,
                 layout: SheetLayout | None = None):
    """(letters, flagged, turned): ``read_sheet`` of the photo, or of it turned 180 degrees
    when that leaves fewer boxes flagged by the form classifier.

    The four squares are alike, so an upside-down page reads without an error,
    as nonsense; the classifier tells which way up the writing is.
    """
    letters = read_sheet(photo_rgb, layout)
    if checkpoint is None:
        return letters, {}, False
    flagged = check_boxes(letters, checkpoint)
    if len(flagged) <= len(letters) // 2:
        return letters, flagged, False
    turned = read_sheet(photo_rgb[::-1, ::-1].copy(), layout)
    turned_flags = check_boxes(turned, checkpoint)
    if len(turned_flags) < len(flagged):
        return turned, turned_flags, True
    return letters, flagged, False


def boxes_image(letters: dict[GlyphSpec, np.ndarray], flagged: dict,
                layout: SheetLayout | None = None, scale: float = 0.5) -> np.ndarray:
    """What was read, laid out as the sheet: flagged boxes framed red, empty ones grey (RGB)."""
    from PIL import Image, ImageDraw

    layout = layout or SheetLayout()
    w, h = int(layout.cell_w * scale), int(layout.cell_h * scale)
    out = Image.new("RGB", (4 * w, layout.grid_rows * h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    gx, gy = layout.grid_origin
    for spec in layout.specs():
        x0, y0, _, _ = layout.cell_box(spec)
        x, y = int((x0 - gx) * scale), int((y0 - gy) * scale)
        if spec in letters:
            ink = Image.fromarray((255 * (1 - letters[spec])).astype(np.uint8))
            ink = ink.resize((max(1, int(ink.width * scale)), max(1, int(ink.height * scale))))
            out.paste(ink.convert("RGB"), (x + 2, y + 2))
        else:
            draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(225, 225, 225))
        colour, width = ((220, 30, 30), 4) if spec in flagged else ((180, 180, 180), 1)
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=colour, width=width)
    return np.asarray(out)
