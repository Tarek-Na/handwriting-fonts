"""Handwriting sample intake.

Two jobs: produce a template for someone to write on, and turn the photograph
they send back into glyph images in exactly the coordinate system the model was
trained in.

The second job is where the brief's "most likely to fail quietly" risk lives.
The corpus is clean typeset outlines; the inference input is ink on paper, shot
on a phone, with uneven lighting, a grey cast and a stroke weight that has
nothing to do with any font. Normalization is what closes most of that gap, and
it has to arrive at the same place the renderer does:

* the baseline goes on a fixed canvas row;
* one scale for the whole sample, derived from the ink of all the glyphs
  together, never per glyph — relative proportions *are* the handwriting;
* ink as a continuous [0, 1] coverage field, not a hard threshold, because the
  tracer reads sub-pixel edge position out of the grey values.

Both of those are the same rules :mod:`hfont.data.render` follows, for the same
reason: the model must not be able to tell which source an image came from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .charset import SEED_24, Charset, GlyphSpec, agl_name, seed_charset

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #

_UNICODE_RE = re.compile(r"^(?:u\+?|uni)([0-9a-f]{4,6})$", re.IGNORECASE)


def parse_glyph_name(stem: str, charset: Charset) -> GlyphSpec | None:
    """Resolve a filename stem to a glyph.

    Accepts ``A``, ``a``, ``comma``, ``U+0041`` and ``uni0041``. Case matters
    for single letters and must not be normalized away — ``A.png`` and
    ``a.png`` are different glyphs, and on a case-insensitive filesystem the
    template writer is responsible for disambiguating (the generated template
    uses ``A_upper``/``a_lower`` suffixes for exactly this reason).
    """
    stem = stem.strip()
    if stem.endswith("_upper"):
        stem = stem[: -len("_upper")].upper()
    elif stem.endswith("_lower"):
        stem = stem[: -len("_lower")].lower()

    by_char = {g.char: g for g in charset}
    if len(stem) == 1 and stem in by_char:
        return by_char[stem]

    match = _UNICODE_RE.match(stem)
    if match:
        char = chr(int(match.group(1), 16))
        return by_char.get(char)

    by_name = {agl_name(g.char): g for g in charset}
    return by_name.get(stem)


# --------------------------------------------------------------------------- #
# image preparation
# --------------------------------------------------------------------------- #

@dataclass
class IntakeConfig:
    size: int = 128
    baseline: float = 0.75
    margin: float = 0.06
    #: Window for local thresholding, as a fraction of the image's short side.
    #: Local rather than global because phone photographs of paper are almost
    #: never evenly lit, and one global threshold either drops the pale corner
    #: or floods the shadowed one.
    threshold_window: float = 0.25
    #: Ink is assumed darker than paper. Set False for white-on-black scans.
    dark_ink: bool = True
    #: Drop connected components smaller than this fraction of the largest one;
    #: removes pencil smudges, paper speckle and JPEG noise.
    min_component_ratio: float = 0.02
    #: Characters whose ink defines the frame. This must match the renderer's
    #: ``RenderConfig.frame_chars``, because the scale it produces is what the
    #: model was trained to expect. Framing on whatever letters happen to be
    #: available instead makes the scale depend on the seed set: `seed30` adds
    #: `f` and `j`, whose ascender and descender stretch the frame and land
    #: every letter on the canvas ~40% too small.
    frame_chars: str = SEED_24
    #: Rescale each sheet so its strokes reach full ink, as a dark pen's do.
    #: See normalize_pen_darkness. Only ever scales up.
    normalize_pen: bool = True


def to_ink_field(image: np.ndarray, cfg: IntakeConfig) -> np.ndarray:
    """Greyscale photo -> ink coverage in [0, 1], 1 = ink.

    Deliberately *not* binarized. The tracer recovers sub-pixel edge positions
    from intermediate values, and thresholding here would throw that away and
    leave stair-stepped outlines in the exported font.
    """
    from skimage.filters import threshold_local

    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = gray[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if gray.max() > 1.5:
        gray /= 255.0

    if not cfg.dark_ink:
        gray = 1.0 - gray

    # Local threshold estimates the paper brightness at each point; ink is how
    # far below the local paper level a pixel sits.
    block = max(3, int(min(gray.shape) * cfg.threshold_window) | 1)
    try:
        local = threshold_local(gray, block_size=block, method="gaussian")
    except ValueError:
        local = np.full_like(gray, float(np.median(gray)))

    ink = np.clip((local - gray) / np.maximum(local * 0.6, 1e-3), 0.0, 1.0)
    # A soft floor removes the faint background texture that local thresholding
    # leaves behind without eating the antialiased stroke edges.
    ink = np.clip((ink - 0.12) / 0.88, 0.0, 1.0)
    return ink.astype(np.float32)


#: A detached stroke within this many median letter-heights of exactly one
#: letter's body is taken to be part of that letter. See _attach_marks.
TOUCH_GAP = 0.15

#: Below this median, a sheet is treated as written in a light pen and lifted.
PEN_FULL_INK = 1.0
#: Never amplify by more than this, so a near-empty sheet cannot blow up noise.
PEN_MAX_GAIN = 3.0


def normalize_pen_darkness(inks: dict) -> dict:
    """Lift a light pen's strokes to full ink, leaving a dark pen untouched.

    ``to_ink_field`` scores a pixel as full ink only when it is at least 60% darker
    than the paper around it. That is calibrated for a dark pen, and a coloured
    one never gets there: orange (255, 140, 0) has a luminance near 0.62 against
    paper near 0.9, which scores about 0.52. One of the three test writers used
    exactly such a pen, and his strokes came through with a median ink of 0.548 --
    sitting on the tracer's 0.5 threshold, so export discarded 44% of his visible
    ink against 17-20% for the others, and the model was fed references fainter
    than anything it trained on.

    How dark a pen is is not part of anyone's handwriting: a font is binary, ink
    or paper. So the whole sheet is rescaled by one factor -- one pen wrote every
    letter -- chosen so the median of its visible ink reaches the level a dark
    pen gives. Pooling across the sheet rather than per letter keeps a genuinely
    lighter letter lighter, and stops one faint letter being amplified alone.

    It only ever scales **up**: a dark pen's median is already ~1.0, so its factor
    is 1 and the sheet passes through unchanged.
    """
    visible = [v[v > 0.1] for v in inks.values() if (v > 0.1).any()]
    if not visible:
        return inks
    median = float(np.median(np.concatenate(visible)))
    gain = min(max(PEN_FULL_INK / max(median, 1e-6), 1.0), PEN_MAX_GAIN)
    if gain <= 1.0 + 1e-3:
        return inks
    log.info("light pen: stroke median %.2f, lifting the sheet by %.2fx", median, gain)
    return {k: np.clip(v * gain, 0.0, 1.0).astype(np.float32) for k, v in inks.items()}


def largest_components(ink: np.ndarray, min_ratio: float) -> np.ndarray:
    """Zero out connected components far smaller than the biggest one."""
    from skimage.measure import label

    mask = ink > 0.25
    if not mask.any():
        return ink
    labels = label(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if counts.max() == 0:
        return ink
    keep = np.nonzero(counts >= counts.max() * min_ratio)[0]
    return np.where(np.isin(labels, keep), ink, 0.0).astype(np.float32)


def _ink_bounds(ink: np.ndarray, threshold: float = 0.15):
    rows = np.nonzero((ink > threshold).any(axis=1))[0]
    cols = np.nonzero((ink > threshold).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


@dataclass
class SampleSet:
    """Handwriting samples sharing one normalization frame."""

    images: dict[str, np.ndarray]
    scale: float
    baseline_row: float


def normalize_samples(
    raw: dict[GlyphSpec, np.ndarray],
    cfg: IntakeConfig | None = None,
) -> SampleSet:
    """Place every sample on the shared canvas the model expects.

    Framing goes through data/frame.py, the same code the corpus renderer
    uses, so a photographed letter lands exactly where the model saw that
    letter during training. The baseline is estimated from the bottoms of
    non-descending lowercase letters, which is where a writer's baseline really
    is; the overall ink box would drag it down by however deep the descenders
    happen to go.

    All samples must share one coordinate frame — cells cut from the same
    template, or images cropped identically. Comparing bottom edges across
    independently cropped images would be comparing unrelated numbers.
    """
    from .data.frame import canvas_scale, frame_from_boxes

    cfg = cfg or IntakeConfig()
    baseline_row = cfg.size * cfg.baseline

    inks = {spec: largest_components(to_ink_field(image, cfg), cfg.min_component_ratio)
            for spec, image in raw.items()}
    if cfg.normalize_pen:
        inks = normalize_pen_darkness(inks)

    prepared: dict[GlyphSpec, tuple[np.ndarray, tuple[int, int, int, int]]] = {}
    for spec, ink in inks.items():
        bounds = _ink_bounds(ink)
        if bounds is None:
            log.warning("sample for %s has no ink; skipped", spec.name)
            continue
        prepared[spec] = (ink, bounds)

    if not prepared:
        return SampleSet({}, 1.0, baseline_row)

    framing = {
        spec.char: b for spec, (_, b) in prepared.items() if spec.char in cfg.frame_chars
    }
    if not framing:  # a seed set with none of the frame letters; better than failing
        framing = {spec.char: b for spec, (_, b) in prepared.items()}
        log.warning("framing on all %d samples: none of the frame letters are present",
                    len(framing))
    frame = frame_from_boxes(framing)
    scale = canvas_scale(frame, cfg.size, cfg.baseline, cfg.margin)

    out: dict[str, np.ndarray] = {}
    for spec, (ink, bounds) in prepared.items():
        out[spec.key] = _place(ink, bounds, frame.baseline, scale, cfg)
    return SampleSet(out, scale, baseline_row)


def _place(
    ink: np.ndarray,
    bounds: tuple[int, int, int, int],
    sample_baseline: float,
    scale: float,
    cfg: IntakeConfig,
) -> np.ndarray:
    """Resample one sample onto the canvas with a single exact affine warp.

    The glyph's ink box is centred horizontally and its baseline goes to the
    canvas baseline row, both to sub-pixel precision. An earlier version
    rounded the placement and the resized dimensions to whole pixels, which is
    up to half a pixel of error — invisible in a large letter, and a real
    fraction of a 3-4px stroke, where the model is learning stroke weight.

    Coordinates are pixel *edges* (pixel i spans [i, i+1]), matching the
    rasterizer; skimage indexes pixel centres, hence the 0.5 terms.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.transform import AffineTransform, warp

    size = cfg.size
    x0, _, x1, _ = bounds
    ink_center = (x0 + x1) / 2.0

    source = ink.astype(np.float64)
    if scale < 1.0:
        # Downscaling without a prefilter aliases thin strokes into dashes.
        source = gaussian_filter(source, sigma=max((1.0 / scale - 1.0) / 2.0, 0.0))

    # Output centre-index -> input centre-index.
    inv = 1.0 / scale
    to_input = AffineTransform(
        matrix=np.array(
            [
                [inv, 0.0, ink_center - 0.5 + (0.5 - size / 2.0) * inv],
                [0.0, inv, sample_baseline - 0.5 + (0.5 - size * cfg.baseline) * inv],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    canvas = warp(source, to_input, output_shape=(size, size), order=1, cval=0.0)
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def load_sample_directory(
    directory: str | Path, charset: Charset, size: int = 128
) -> dict[str, np.ndarray]:
    """Read a directory of per-glyph images into normalized model inputs."""
    from PIL import Image

    directory = Path(directory)
    raw: dict[GlyphSpec, np.ndarray] = {}
    unmatched: list[str] = []

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        spec = parse_glyph_name(path.stem, charset)
        if spec is None:
            unmatched.append(path.name)
            continue
        raw[spec] = np.asarray(Image.open(path).convert("L"))

    if unmatched:
        log.warning(
            "%d file(s) did not match a glyph and were ignored: %s",
            len(unmatched), ", ".join(unmatched[:8]),
        )
    return normalize_samples(raw, IntakeConfig(size=size)).images


# --------------------------------------------------------------------------- #
# template: layout, rendering, and extraction from a photograph
# --------------------------------------------------------------------------- #
#
# The template and the extractor share one geometry object, so the extractor
# knows exactly where every cell, guide and marker is without detecting any of
# them. It only has to find the four corner markers and undo the perspective.
#
# Guides and letter labels are printed in red. Read through the red channel, a
# red print is as bright as the paper and disappears, while pen ink (blue or
# black) is dark in every channel and survives. That is the "dropout colour"
# trick from form scanning, and it removes the one thing that otherwise
# contaminates every cell: a baseline rule running straight through the bottom
# of each letter. It needs a colour print, and the template says so on its face.

INK_DROPOUT_RGB = (235, 70, 70)


@dataclass(frozen=True)
class TemplateLayout:
    """Page geometry in pixels at ``dpi``. Shared by rendering and extraction."""

    chars: str
    columns: int = 6
    cell: int = 260
    margin: int = 130
    header: int = 170
    marker: int = 56
    baseline: float = 0.72
    x_height: float = 0.42
    label_size: int = 44
    dpi: int = 300

    @property
    def rows(self) -> int:
        return -(-len(self.chars) // self.columns)

    @property
    def grid_origin(self) -> tuple[int, int]:
        return self.margin, self.margin + self.header

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the page."""
        return (
            self.margin * 2 + self.columns * self.cell,
            self.margin * 2 + self.header + self.rows * self.cell,
        )

    def cell_box(self, index: int) -> tuple[int, int, int, int]:
        r, c = divmod(index, self.columns)
        gx, gy = self.grid_origin
        x0, y0 = gx + c * self.cell, gy + r * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    def marker_centers(self) -> np.ndarray:
        """Centres of the TL, TR, BR, BL registration squares, as (x, y)."""
        gx, gy = self.grid_origin
        gw, gh = self.columns * self.cell, self.rows * self.cell
        off = self.marker
        return np.array(
            [
                [gx - off, gy - off],
                [gx + gw + off, gy - off],
                [gx + gw + off, gy + gh + off],
                [gx - off, gy + gh + off],
            ],
            dtype=np.float64,
        )


def _label_font(size: int):
    """A real TrueType face for labels; the PIL built-in bitmap font is ~10px."""
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def build_template(charset: Charset, seed: str = "seed24") -> tuple[np.ndarray, TemplateLayout]:
    """Render the printable page. Returns (RGB image, its layout)."""
    from PIL import Image, ImageDraw

    chars = "".join(g.char for g in seed_charset(charset, seed))
    layout = TemplateLayout(chars=chars)
    width, height = layout.size

    page = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    red = INK_DROPOUT_RGB

    draw.text(
        (layout.margin, layout.margin // 2),
        "Write each letter in its box, sitting on the solid line.",
        fill=(0, 0, 0), font=_label_font(40),
    )
    draw.text(
        (layout.margin, layout.margin // 2 + 56),
        "Dark pen (blue or black). Print in COLOUR. Photograph flat with all four black squares in view.",
        fill=(60, 60, 60), font=_label_font(26),
    )

    half = layout.marker // 2
    for cx, cy in layout.marker_centers():
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(0, 0, 0))

    label_font = _label_font(layout.label_size)
    for i, char in enumerate(chars):
        x0, y0, x1, y1 = layout.cell_box(i)
        draw.rectangle([x0, y0, x1, y1], outline=red, width=2)
        base_y = y0 + int(layout.cell * layout.baseline)
        xh_y = y0 + int(layout.cell * layout.x_height)
        draw.line([x0 + 10, base_y, x1 - 10, base_y], fill=red, width=3)
        for dash in range(x0 + 10, x1 - 10, 18):
            draw.line([dash, xh_y, min(dash + 9, x1 - 10), xh_y], fill=red, width=2)
        draw.text((x0 + 10, y0 + 4), char, fill=red, font=label_font)

    return np.asarray(page), layout


def save_template(path: str | Path, charset: Charset, seed: str = "seed24") -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image, layout = build_template(charset, seed)
    Image.fromarray(image).save(path, dpi=(layout.dpi, layout.dpi))
    return path


class TemplateError(RuntimeError):
    pass


def find_markers(photo_rgb: np.ndarray) -> np.ndarray:
    """Locate the four registration squares. Returns TL, TR, BR, BL as (x, y).

    A marker is a large, solid, roughly square dark blob. Handwriting fails the
    solidity test (strokes fill little of their bounding box) and text fails
    the size test. Of the survivors, the one nearest each image corner wins,
    which tolerates the page being slightly rotated or shot off-centre.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    gray = photo_rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    dark = gray < threshold_otsu(gray)

    h, w = gray.shape
    min_area = (min(h, w) * 0.012) ** 2
    max_area = (min(h, w) * 0.12) ** 2
    candidates = []
    for region in regionprops(label(dark)):
        if not (min_area <= region.area <= max_area):
            continue
        minr, minc, maxr, maxc = region.bbox
        bh, bw = maxr - minr, maxc - minc
        if region.solidity < 0.9 or not (0.6 < bw / max(bh, 1) < 1.66):
            continue
        cy, cx = region.centroid
        candidates.append((cx, cy))
    if len(candidates) < 4:
        raise TemplateError(
            f"found {len(candidates)} registration squares, need 4. "
            "Is the whole page in frame and evenly lit?"
        )

    pts = np.asarray(candidates)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    chosen = np.asarray(
        [pts[np.argmin(np.linalg.norm(pts - corner, axis=1))] for corner in corners]
    )
    if len({tuple(p) for p in chosen}) < 4:
        raise TemplateError("registration squares are ambiguous; retake the photo straighter")
    return chosen


def extract_cells(
    photo_rgb: np.ndarray, layout: TemplateLayout, inset: int = 12
) -> dict[str, np.ndarray]:
    """Undo perspective and cut one greyscale crop per character.

    Crops come from the red channel, so the red guides and labels drop out and
    only the writing remains. The label corner is blanked as well, because a
    faded or monochrome print may not drop out completely.
    """
    from skimage.transform import ProjectiveTransform, warp

    found = find_markers(photo_rgb)
    transform = ProjectiveTransform()
    # warp() wants a map from output (template) coords to input (photo) coords.
    if not transform.estimate(layout.marker_centers(), found):
        raise TemplateError("could not fit a perspective transform to the markers")

    red = photo_rgb[..., 0].astype(np.float64)
    if red.max() > 1.5:
        red /= 255.0
    width, height = layout.size
    flat = warp(red, transform, output_shape=(height, width), order=1, cval=1.0)

    label_w = int(layout.label_size * 1.4)
    label_h = int(layout.label_size * 1.3)
    cells: dict[str, np.ndarray] = {}
    for i, char in enumerate(layout.chars):
        x0, y0, x1, y1 = layout.cell_box(i)
        crop = flat[y0 + inset : y1 - inset, x0 + inset : x1 - inset].copy()
        paper = float(np.percentile(crop, 90)) if crop.size else 1.0
        crop[:label_h, :label_w] = paper
        cells[char] = (np.clip(crop, 0.0, 1.0) * 255).astype(np.uint8)
    return cells


# --------------------------------------------------------------------------- #
# freehand intake: letters on blank paper, no printed template
# --------------------------------------------------------------------------- #


@dataclass
class FreehandConfig:
    """Reading letters written straight onto blank paper.

    The printed template exists to make letters findable and to put them in one
    coordinate frame. Without a printer, both have to be recovered from the
    writing itself: rows are found by clustering, and the frame comes from each
    row's own writing line.
    """

    #: How many characters per written row, in seed order.
    rows: tuple[int, ...] = (10, 10, 10)
    #: Ink threshold for segmentation only; the crops stay continuous.
    ink_level: float = 0.30
    #: Weaker ink is still ink where it continues from a stroke above
    #: ``ink_level``. Below this, it is paper.
    ink_faint: float = 0.20
    #: Components smaller than this fraction of the median letter are specks.
    #: Well below an `i` dot, which is ~5-15% of its stem.
    speck_ratio: float = 0.012
    #: Merge two components into one letter when the horizontal gap between
    #: them is under this fraction of the median letter width.
    merge_gap: float = 0.28
    #: A component this tall, relative to the median, is a letter body rather
    #: than a dot or an accent.
    body_height: float = 0.45
    #: Above this share of the page reading as ink, the paper is ruled or
    #: squared: handwriting alone covers 2-4%, a grid pushes it past 6%.
    ruled_ink_share: float = 0.06
    #: Cell height and baseline position, in x-heights. 4.2 leaves room for
    #: ascenders above and descenders below every hand.
    cell_heights: float = 4.2
    cell_baseline: float = 0.70


def _letter_groups(ink: np.ndarray, cfg: FreehandConfig, n_rows: int):
    """Segment ink into rows of letters, each letter a mask plus its box.

    Two components join into one letter when they overlap horizontally (the dot
    of an `i` over its stem) or sit within a fraction of a letter width of each
    other. Rows are clustered on vertical position, which survives the drift
    and slant of writing without ruled lines.
    """
    from skimage.measure import label, regionprops

    # Hysteresis, not a single level: a pale pen in poor light crosses the
    # threshold only along parts of each stroke, and the letter arrives in
    # pieces — one hand gave a `y` as three bodies and three loose scraps. Weak
    # ink that continues from strong ink is the rest of that stroke; weak ink
    # on its own is paper texture.
    from skimage.filters import apply_hysteresis_threshold

    labels = label(apply_hysteresis_threshold(ink, cfg.ink_faint, cfg.ink_level))
    props = [p for p in regionprops(labels) if p.area >= 6]
    if not props:
        raise TemplateError("no ink found in the photo")

    median_area = float(np.median([p.area for p in props]))
    props = [p for p in props if p.area >= median_area * cfg.speck_ratio]

    # Rows are clustered on letter bodies only. A dot sits high above its stem
    # and, with rows written close together, lands nearer the centre of the row
    # above — which is how the dot of `i` ends up over the `t` on the line
    # above, leaving a dotless `i` behind. Marks are attached to the body they
    # belong to afterwards, by position relative to that body.
    median_height = float(np.median([p.bbox[2] - p.bbox[0] for p in props]))
    bodies = [p for p in props if (p.bbox[2] - p.bbox[0]) >= cfg.body_height * median_height]
    marks = [p for p in props if p not in bodies]
    if len(bodies) < n_rows:
        raise TemplateError("not enough letters found to make out the rows")

    centres = np.array([p.centroid[0] for p in bodies], dtype=np.float64)
    rows = _cluster_1d(centres, n_rows)

    grouped: list[list[dict]] = []
    for row in rows:
        members = [bodies[i] for i in row]
        members.sort(key=lambda p: p.bbox[1])
        widths = [p.bbox[3] - p.bbox[1] for p in members]
        gap = float(np.median(widths)) * cfg.merge_gap

        letters: list[list] = []
        for p in members:
            y0, x0, y1, x1 = p.bbox
            if letters:
                px0, px1 = letters[-1][1], letters[-1][2]
                overlap = min(px1, x1) - max(px0, x0)
                if overlap > 0 or x0 - px1 < gap:
                    letters[-1][1] = min(px0, x0)
                    letters[-1][2] = max(px1, x1)
                    letters[-1][0].append(p)
                    continue
            letters.append([[p], x0, x1])

        grouped.append([{"parts": parts} for parts, _, _ in letters])

    return labels, grouped, marks, median_height


def _promote_punctuation(
    grouped: list[list[dict]], small: list, expected: tuple[int, ...], median_height: float
) -> list:
    """Let a row short of characters claim punctuation from the small parts.

    Height cannot tell a comma from the dot of an `i`: across four hands the
    two overlap completely, at 0.08-0.26 of a letter's height. Position
    distinguishes them — a comma rests on the writing line, a dot floats above
    it — but that alone promotes every scrap lying near the line. So a row only
    takes what it is missing, largest first, and only from components that are
    not part of a letter already found.
    """
    if not grouped:
        return small
    lines = [
        float(np.median([max(p.bbox[2] for p in letter["parts"]) for letter in row]))
        for row in grouped
    ]
    taken: set[int] = set()

    for r, row in enumerate(grouped):
        short = (expected[r] if r < len(expected) else 0) - len(row)
        if short <= 0:
            continue
        spans = [(min(p.bbox[1] for p in letter["parts"]),
                  max(p.bbox[3] for p in letter["parts"])) for letter in row]
        candidates = []
        for i, part in enumerate(small):
            _, x0, y1, x1 = part.bbox
            if i in taken or not (lines[r] - 0.3 * median_height <= y1 <= lines[r] + median_height):
                continue
            width = max(x1 - x0, 1)
            if any(min(b, x1) - max(a, x0) > 0.5 * width for a, b in spans):
                continue  # sits within a letter: a piece of it, not punctuation
            candidates.append((part.area, i, part))
        for _, i, part in sorted(candidates, reverse=True)[:short]:
            taken.add(i)
            grouped[r].append({"parts": [part]})
        grouped[r].sort(key=lambda letter: min(p.bbox[1] for p in letter["parts"]))

    return [part for i, part in enumerate(small) if i not in taken]


def _finish_letters(grouped: list[list[dict]]) -> None:
    """Cache each letter's component labels and bounding box."""
    for row in grouped:
        for letter in row:
            parts = letter["parts"]
            letter["labels"] = [p.label for p in parts]
            letter["box"] = (
                min(p.bbox[1] for p in parts), min(p.bbox[0] for p in parts),
                max(p.bbox[3] for p in parts), max(p.bbox[2] for p in parts),
            )


#: Letters that are written with a dot above them. Anything else floating over
#: a letter is not part of it — on ruled paper it is usually what is left of a
#: printed line where the writing crossed it.
DOTTED = "ij"


def _attach_marks(
    grouped: list[list[dict]], marks: list, median_height: float,
    chars: list[list[str]] | None = None,
) -> None:
    """Give each dot or accent to the letter body it sits over.

    Two things can be a small component. A dot joins the nearest body starting
    below it, allowing the sideways offset handwritten dots always have. A
    stroke fragment — the bar of an `E`, the tick that starts an `n` — joins the
    body directly above it. Whatever is left over and too small to be a
    character is pen noise and is dropped; a comma or full stop is large enough
    to survive as a letter of its own.

    When ``chars`` is known, only `i` and `j` may take a dot. The letters are
    matched to the sheet by position, so this is available — and on ruled paper
    it is the difference between a clean sheet and an alphabet sprinkled with
    the remains of printed lines, which the model then imitates.

    That rule alone throws away a detached stroke that is genuinely part of a
    letter: the top bar of an `E` written without lifting into the stem, the
    crossbar of a `T`. Such a stroke lies *within* its letter's horizontal span
    and no other letter is near it, so it is attached on those two conditions —
    which the leftovers of a printed ruling fail, because they straddle the gap
    between letters and several letters can claim them.
    """
    areas = [p.area for row in grouped for letter in row for p in letter["parts"]]
    noise_area = 0.15 * float(np.median(areas))

    for mark in marks:
        my0, mx0, my1, mx1 = mark.bbox
        best = None
        spanned: list[tuple[int, int]] = []
        touching: list[tuple[int, int]] = []
        for r, row in enumerate(grouped):
            for i, letter in enumerate(row):
                y0 = min(p.bbox[0] for p in letter["parts"])
                y1 = max(p.bbox[2] for p in letter["parts"])
                x0 = min(p.bbox[1] for p in letter["parts"])
                x1 = max(p.bbox[3] for p in letter["parts"])
                sideways = max(0.0, max(x0 - mx1, mx0 - x1))
                below = y0 - my1          # body starts below the mark: a dot
                above = my0 - y1          # body ends above the mark: a fragment
                # Inside the letter's own box: a piece of that letter, cut off
                # from it — removing printed rulings thins the junctions, and
                # the middle bar of an `E` can come away from its stem.
                inside = ((min(y1, my1) - max(y0, my0)) * (min(x1, mx1) - max(x0, mx0))
                          if y1 > my0 and my1 > y0 and x1 > mx0 and mx1 > x0 else 0)
                dots = chars is None or chars[r][i] in DOTTED
                # Overlapping this letter's column, not contained by it: the top
                # bar of an `E` is wider than the stem it belongs to.
                column = min(x1, mx1) - max(x0, mx0) > 0.25 * (mx1 - mx0)
                if column and (-0.2 * median_height <= below <= 1.2 * median_height
                               or -0.2 * median_height <= above <= 0.5 * median_height):
                    spanned.append((r, i))
                # A descender that all but touches the bottom of a letter's body
                # is part of it, even when it curls away from that letter's
                # column: the hook of a `j` sits just below the stem and bends
                # left, so it fails every rule above (centre outside the span,
                # too little column overlap, below rather than above).
                #
                # Below the body only. Allowing a touch from any side also
                # captured the crossing of two grid lines beside writer 2's `A`
                # -- a `+` that touches exactly one letter as surely as a `j`
                # hook does, and would have been imitated into every `A`.
                gap = max(0.0, x0 - mx1, mx0 - x1, y0 - my1, my0 - y1)
                hangs_below = above >= -TOUCH_GAP * median_height
                if gap <= TOUCH_GAP * median_height and hangs_below:
                    touching.append((r, i))
                if inside >= 0.5 * (my1 - my0) * (mx1 - mx0):
                    score = -1e9 + (y0 - my0)  # a containing letter wins outright
                elif (dots and sideways <= 0.5 * median_height
                        and -0.2 * median_height <= below <= 1.2 * median_height):
                    score = below + sideways
                elif (x0 <= 0.5 * (mx0 + mx1) <= x1
                      and -0.2 * median_height <= above <= 0.5 * median_height):
                    score = above
                else:
                    continue
                if best is None or score < best[0]:
                    best = (score, r, i)
        if best is None and len(spanned) == 1:
            r, i = spanned[0]
            log.debug("kept a %d px detached stroke inside the span of one letter", int(mark.area))
            best = (0.0, r, i)
        # Same unique-claimant guard as above. A ruling remnant straddles the
        # gap between two letters, so it touches none of them or several; only
        # a stroke that belongs to exactly one letter touches exactly one.
        if best is None and len(touching) == 1:
            r, i = touching[0]
            log.debug("kept a %d px stroke touching one letter", int(mark.area))
            best = (0.0, r, i)
        if best is not None:
            grouped[best[1]][best[2]]["parts"].append(mark)
        elif mark.area < noise_area:
            log.debug("dropped a %d px speck at %s", int(mark.area), mark.bbox)
        else:
            log.info("dropped a %d px mark at %s that belongs to no letter",
                     int(mark.area), mark.bbox)


def _cluster_1d(values: np.ndarray, k: int, iterations: int = 40) -> list[list[int]]:
    """k-means on one axis, seeded on quantiles (no scikit-learn dependency)."""
    centres = np.quantile(values, np.linspace(0.5 / k, 1 - 0.5 / k, k))
    assign = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        new = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            if (assign == j).any():
                centres[j] = values[assign == j].mean()
    order = np.argsort(centres)
    return [list(np.nonzero(assign == j)[0]) for j in order]


def freehand_cells(
    photo_rgb: np.ndarray, charset: Charset, seed: str = "seed30",
    cfg: FreehandConfig | None = None, debug_path: str | Path | None = None,
) -> dict[GlyphSpec, np.ndarray]:
    """Cut one crop per written character, all sharing a coordinate frame.

    Every crop is the same height and is positioned against its own row's
    fitted writing line, so a bottom edge means the same thing in all of them —
    the property :func:`normalize_samples` depends on and the only thing the
    printed grid was providing. Crops carry only their own letter's ink; rows of
    handwriting sit close enough that a neighbour's descender would otherwise
    land in the frame and drag the baseline estimate with it.
    """
    cfg = cfg or FreehandConfig()
    # Writing order, not charset order: letters are matched to the sheet by
    # position, so the sequence has to be the one the writer was asked for.
    from .charset import LATIN_SEED_SETS

    seed_charset(charset, seed)  # validates that every seed char exists
    by_char = {g.char: g for g in charset}
    specs = [by_char[c] for c in LATIN_SEED_SETS.get(seed, seed)]
    if sum(cfg.rows) != len(specs):
        raise TemplateError(f"rows {cfg.rows} do not add up to {len(specs)} characters")

    # Darkest channel per pixel, not luminance or any single channel: ink is
    # whatever is furthest below the paper in *some* channel, whatever colour
    # the pen was. An orange pen is nearly invisible in red and weak in green —
    # 0.8% of the page reads as ink there, none of it strongly, and letters
    # arrive as disconnected fragments — while in blue it is plain.
    gray = photo_rgb.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    gray = gray[..., :3].min(axis=2) if gray.ndim == 3 else gray
    # Find the sheet first. Anything outside it — a desk, a shadow, the edge of
    # the table — is darker than paper and would otherwise read as one enormous
    # letter, taking the row clustering and every size estimate with it.
    intake = IntakeConfig()
    # Order matters: the desk goes first, so its bulk cannot be mistaken for a
    # thick pen when the rulings are measured.
    ink = _remove_rulings(to_ink_field(_paper_only(gray), intake), cfg)
    labels, rows, marks, median_height = _letter_groups(ink, cfg, len(cfg.rows))

    # Count letter bodies before anything is attached to them, so that a
    # leftover scrap cannot pass as a character and shift every letter after
    # it, and so that dots can be given out knowing which letter is which.
    marks = _promote_punctuation(rows, marks, tuple(cfg.rows), median_height)
    found = tuple(len(r) for r in rows)
    if found != tuple(cfg.rows):
        if debug_path:
            _finish_letters(rows)
            _draw_debug(photo_rgb, rows, specs, cfg.rows, debug_path)
        raise TemplateError(
            f"expected {tuple(cfg.rows)} characters per row but found {found}; "
            f"see {debug_path}" if debug_path else f"expected {tuple(cfg.rows)}, found {found}"
        )

    by_row, at = [], 0
    for count in cfg.rows:
        by_row.append([s.char for s in specs[at : at + count]])
        at += count
    _attach_marks(rows, marks, median_height, by_row)
    _finish_letters(rows)

    # Ink is measured once, on the whole page, and the crops are cut out of
    # that field. Measuring it per crop instead lets the local-contrast window
    # sit inside a thick stroke, which reads the stroke's own interior as paper
    # and leaves hairline outlines — and a hairline then breaks into fragments
    # the speck filter deletes. A pen laid down on paper is much wider relative
    # to a letter than a printed stem is.
    from skimage.morphology import dilation

    heights = [b["box"][3] - b["box"][1] for row in rows for b in row]
    x_height = float(np.median(heights)) * 0.62   # median over mixed case ~ 1.6 x-heights
    pad = max(2, int(round(0.18 * x_height)))

    # The cell is sized to the writing, not to a fixed number of x-heights. A
    # constant tall enough for most hands still cuts the descender off a long
    # `j`, and a letter arriving clipped is worse than one arriving small.
    baselines = [_row_baseline(row, specs[sum(cfg.rows[:r]) : sum(cfg.rows[:r + 1])])
                 for r, row in enumerate(rows)]
    above = below = 0.0
    for row, baseline in zip(rows, baselines):
        for letter in row:
            x0, y0, x1, y1 = letter["box"]
            line = baseline(0.5 * (x0 + x1))
            above = max(above, line - y0)
            below = max(below, y1 - line)
    cell_h = int(round((above + below) * 1.12))
    cell_baseline = (above * 1.06) / max(cell_h, 1)

    cells: dict[GlyphSpec, np.ndarray] = {}
    index = 0
    for row_letters, count, baseline in zip(rows, cfg.rows, baselines):
        row_specs = specs[index : index + count]
        index += count
        for letter, spec in zip(row_letters, row_specs):
            x0, y0, x1, y1 = letter["box"]
            # Dilate before masking so the soft stroke edges, which fall below
            # the segmentation level, stay with their letter.
            keep = dilation(np.isin(labels, letter["labels"]), np.ones((5, 5), bool))
            only = np.where(keep, ink, 0.0)
            base_y = baseline(0.5 * (x0 + x1))
            top = int(round(base_y - cell_baseline * cell_h))
            crop = _crop(only, x0 - pad, top, x1 + pad, top + cell_h, 0.0)
            cells[spec] = 1.0 - crop  # back to a paper-and-ink image

    if debug_path:
        _draw_debug(photo_rgb, rows, specs, cfg.rows, debug_path)
    return cells


def _remove_rulings(ink: np.ndarray, cfg: FreehandConfig) -> np.ndarray:
    """Erase the printed lines of ruled or squared paper, keeping the writing.

    People write on the paper they have, and a grid is ink as far as any
    threshold is concerned — on squared paper it covers 10% of the page against
    3% for the writing, and the letters are lost among it.

    Rulings are separated from writing by thickness, not by direction, because
    a photographed page is never square to the camera and its lines are never
    quite straight. A morphological opening keeps only what survives eroding by
    ``r``, so with the radius set between the two thicknesses the lines vanish
    and the strokes come back at full width.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import disk

    mask = ink > cfg.ink_level
    share = float(mask.mean())
    if share < cfg.ruled_ink_share:
        return ink

    # Half-widths: the rulings dominate by pixel count, so the median measures
    # them, while the pen shows up in the far tail. Both tests are needed. A
    # page can be mostly ink without being ruled — a heavy pen, or a dark
    # surround — and eroding that by a ruling's width would eat the writing.
    spread = distance_transform_edt(mask)[mask]
    line, stroke = float(np.median(spread)), float(np.percentile(spread, 99.5))
    if line > 0.45 * stroke:  # one population of thicknesses: no rulings
        return ink
    radius = int(np.clip(round(0.5 * (line + stroke)), 1, 8))

    # Erode to cores, then regrow those cores inside the original ink for a
    # bounded number of steps. A plain opening deletes whatever is thinner than
    # the radius, and a pen varies along its stroke — that leaves letters
    # pitted and broken. Regrowing restores the thin parts of any stroke that
    # has a thick core somewhere, while a ruling, having no core at all, can
    # only creep back a few pixels from where a letter crosses it.
    from scipy.ndimage import binary_dilation, binary_erosion

    core = binary_erosion(mask, disk(radius))
    grown = core
    for _ in range(3 * radius):
        grown = binary_dilation(grown, disk(1)) & mask
    opened = np.where(grown, ink, 0.0).astype(np.float32)
    left = float((opened > cfg.ink_level).mean())
    if left > 0.6 * share:
        # Barely anything went: the thin structures were part of the writing —
        # a light stroke, the thin side of a nib — not rulings, which dominate
        # the page they are printed on.
        return ink
    if left < 0.004:  # the pen was as thin as the rulings; better to keep both
        log.warning("ruled paper: removing the lines would take the writing too; kept")
        return ink
    log.info("ruled paper: removed lines thinner than %dpx; ink %.1f%% -> %.1f%% of the page",
             2 * radius, 100 * share, 100 * left)
    return opened


def _paper_only(gray: np.ndarray, inset: float = 0.01) -> np.ndarray:
    """Blank everything outside the sheet, keeping the image's coordinates."""
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk

    # Is there anything but paper in the frame? A desk survives being eroded by
    # a few pixels because it is a wide region; rulings and pen strokes do not.
    # Deciding on brightness alone cannot tell a desk from a shadow, and an
    # unevenly lit sheet that fills the frame would be carved up — which is
    # what happens to a photo of squared paper with a shadow across one corner.
    dark = gray < threshold_otsu(gray[::4, ::4])
    if binary_erosion(dark, disk(5)).mean() < 0.02:
        return gray
    # Wide and dark is not enough: a shadow lying across the page is both. A
    # table is *much* darker than paper, while a shadow keeps most of the
    # brightness — and blanking a shadowed corner throws away the letters
    # written in it.
    if float(np.median(gray[dark])) > 0.6 * float(np.median(gray[~dark])):
        return gray

    labels = label(~dark)
    props = regionprops(labels)
    if not props:
        return gray

    # Fill first: the writing is a hole in the sheet, and eroding an unfilled
    # mask would grow those holes and erase the very ink we are after.
    sheet = binary_fill_holes(labels == max(props, key=lambda p: p.area).label)
    if sheet.mean() > 0.95:  # no desk in shot; the sheet is the whole frame
        return gray
    # Erode rather than crop to a box: a sheet photographed at an angle is a
    # quadrilateral, and its slanted edges survive any rectangular crop — they
    # then read as two long dark letters down the side of the page.
    sheet = binary_erosion(sheet, disk(max(2, int(min(gray.shape) * inset))))
    return np.where(sheet, gray, float(np.percentile(gray, 95))).astype(np.float32)


def _row_baseline(letters: list[dict], specs: list[GlyphSpec]):
    """Fit the row's writing line through the feet of its non-descending letters."""
    from .data.frame import BASELINE_CHARS

    sitting = [
        (0.5 * (b["box"][0] + b["box"][2]), b["box"][3])
        for b, s in zip(letters, specs)
        if s.char in BASELINE_CHARS or s.char.isupper() or s.char.isdigit()
    ]
    if len(sitting) < 2:
        flat = float(np.median([b["box"][3] for b in letters]))
        return lambda x: flat
    xs = np.array([p[0] for p in sitting], dtype=np.float64)
    ys = np.array([p[1] for p in sitting], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    keep = np.abs(ys - (slope * xs + intercept)) < 0.5 * np.std(ys - (slope * xs + intercept)) + 3
    if keep.sum() >= 2:
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
    return lambda x: slope * x + intercept


def _crop(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill: float) -> np.ndarray:
    out = np.full((y1 - y0, x1 - x0), fill, dtype=np.float32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def _draw_debug(photo_rgb, rows, specs, counts, path) -> None:
    """Overlay what was found, so a miscount can be seen rather than guessed at."""
    from PIL import Image, ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(photo_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    index = 0
    for row, expected in zip(rows, list(counts) + [0] * len(rows)):
        for letter in row:
            x0, y0, x1, y1 = letter["box"]
            label = specs[index].char if index < len(specs) else "?"
            colour = (0, 160, 0) if len(row) == expected else (220, 0, 0)
            draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
            draw.text((x0 + 2, max(0, y0 - 14)), label, fill=colour)
            index += 1
    img.save(path)


def load_freehand_photo(
    path: str | Path, charset: Charset, seed: str = "seed30", size: int = 128,
    cfg: FreehandConfig | None = None, debug_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Photographed blank-paper writing -> normalized model inputs."""
    from PIL import Image

    photo = np.asarray(Image.open(path).convert("RGB"))
    cells = freehand_cells(photo, charset, seed, cfg, debug_path)
    # The crops already carry page-wide ink measurements, so the per-cell
    # threshold is left near-global; it only has to not undo that work.
    return normalize_samples(cells, IntakeConfig(size=size, threshold_window=0.9)).images


def load_template_photo(
    path: str | Path, charset: Charset, seed: str = "seed24", size: int = 128
) -> dict[str, np.ndarray]:
    """Photographed template -> normalized model inputs, keyed by glyph key."""
    from PIL import Image

    photo = np.asarray(Image.open(path).convert("RGB"))
    _, layout = build_template(charset, seed)
    cells = extract_cells(photo, layout)
    by_char = {g.char: g for g in charset}
    raw = {by_char[c]: img for c, img in cells.items() if c in by_char}
    return normalize_samples(raw, IntakeConfig(size=size)).images


