"""Assemble traced outlines into an installable OpenType font.

Output is CFF-flavoured OpenType (``.otf``). The tracer produces cubic Beziers
and CFF stores cubics natively, so nothing is resampled on the way out; a
TrueType build would require converting every curve to quadratics first and
would introduce error at the last possible moment, after all the care taken
upstream to keep the outlines clean.

This is the step the academic work skips, and it is where "a picture of a font"
becomes "a font". Everything here — winding direction, the ``cmap``, vertical
metrics, ``post``, ``OS/2`` — is required for the file to install and type
correctly rather than merely to exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.roundTools import otRound
from fontTools.pens.t2CharStringPen import T2CharStringPen

from ..charset import Charset, GlyphSpec
from ..vector.trace import BezierPath, TraceConfig, trace_image

log = logging.getLogger(__name__)


@dataclass
class FontMetadata:
    family: str = "Handwriting"
    style: str = "Regular"
    version: str = "1.000"
    designer: str = ""
    units_per_em: int = 1000
    ascender: int = 800
    descender: int = -200
    line_gap: int = 0
    copyright: str = ""

    @property
    def postscript_name(self) -> str:
        fam = "".join(ch for ch in self.family if ch.isalnum())
        sty = "".join(ch for ch in self.style if ch.isalnum())
        return f"{fam}-{sty}"[:63]


@dataclass
class BuildConfig:
    #: Canvas geometry the rasters were produced in. Must match RenderConfig.
    image_size: int = 128
    baseline: float = 0.75
    margin: float = 0.06
    ascender_em: float = 0.8
    #: Side bearings come from the predicted advance. A glyph whose ink is
    #: wider than its whole advance is widened to leave at least this many em
    #: units of clearance, so neighbouring letters cannot collide.
    min_side_bearing: int = 8
    #: The median clearance a font should leave between advance and ink, in em.
    #: Measured across Segoe Print, Ink Free, Times and Calibri: 0.05-0.12.
    min_tracking: float = 0.045
    #: Run generated rasters through clean_raster() before tracing.
    clean: bool = True
    trace: TraceConfig = field(default_factory=TraceConfig)

    @property
    def pixels_per_em(self) -> float:
        return (self.image_size * self.baseline - self.image_size * self.margin) / self.ascender_em


@dataclass
class GlyphOutline:
    """One finished glyph in font units."""

    spec: GlyphSpec
    paths: list[BezierPath]
    advance: int

    @property
    def name(self) -> str:
        return self.spec.name


def _orient(paths: list[BezierPath]) -> list[BezierPath]:
    """Force PostScript winding: outer contours counter-clockwise, holes clockwise.

    Fonts are filled with the non-zero rule, so a counter inside a bowl only
    appears as a hole if it winds opposite to the shape containing it. The
    tracer's output direction depends on which way marching squares happened to
    walk, so it cannot be relied on.

    Containment is decided by area magnitude: a contour strictly inside another
    is necessarily smaller, so sorting by descending area and alternating
    orientation by nesting depth gets the common cases (a bowl, a counter, and
    the occasional island inside a counter) right.
    """
    if not paths:
        return paths

    areas = [p.signed_area() for p in paths]
    order = sorted(range(len(paths)), key=lambda i: -abs(areas[i]))

    out: list[BezierPath] = []
    for i in order:
        path, area = paths[i], areas[i]
        # In image space y runs down, so the sign convention is inverted
        # relative to the usual maths one; the caller flips y before this runs,
        # so positive area here means counter-clockwise in font space.
        depth = _nesting_depth(paths, areas, i)
        want_positive = depth % 2 == 0
        if (area > 0) != want_positive:
            path = path.reverse()
        out.append(path)
    return out


def _nesting_depth(paths: list[BezierPath], areas: list[float], index: int) -> int:
    """How many other contours contain this one."""
    pt = np.asarray(paths[index].start, dtype=np.float64)
    depth = 0
    for j, other in enumerate(paths):
        if j == index or abs(areas[j]) <= abs(areas[index]):
            continue
        if _point_in_polygon(pt, np.asarray(other.points, dtype=np.float64)):
            depth += 1
    return depth


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    xs, ys = polygon[:, 0], polygon[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = ((ys > y) != (ys2 > y)) & (
        x < (xs2 - xs) * (y - ys) / np.where(ys2 - ys == 0, 1e-12, ys2 - ys) + xs
    )
    return bool(crosses.sum() % 2 == 1)


def clean_raster(
    image: np.ndarray,
    low: float = 0.22,
    high: float = 0.78,
    min_component_ratio: float = 0.015,
) -> np.ndarray:
    """Condition a generated raster before tracing.

    A model's output is not a clean binary image. It carries faint grey haze in
    the background and isolated specks, and every speck becomes a closed contour
    and then a stray path in the font. Two steps deal with it:

    *Contrast stretch* between ``low`` and ``high`` pushes the haze to zero and
    the strokes to one, while deliberately leaving a gradient band across the
    actual edges — a hard threshold would remove the sub-pixel edge information
    the tracer depends on and stair-step every outline.

    *Component pruning* drops blobs far smaller than the glyph's main body.

    Ground-truth rasters pass through this essentially unchanged, so the same
    code runs whether the input is generated or real.
    """
    from skimage.measure import label

    ink = np.clip((np.clip(image, 0.0, 1.0) - low) / max(high - low, 1e-6), 0.0, 1.0)

    mask = ink > 0.25
    if not mask.any():
        return ink.astype(np.float32)

    labels = label(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if counts.max() > 0:
        keep = np.nonzero(counts >= counts.max() * min_component_ratio)[0]
        ink = np.where(np.isin(labels, keep), ink, 0.0)
    return ink.astype(np.float32)


class GlyphVectorizer:
    """Turns generated rasters into font-unit outlines."""

    def __init__(self, config: BuildConfig | None = None, units_per_em: int = 1000) -> None:
        self.cfg = config or BuildConfig()
        self.units_per_em = units_per_em

    @property
    def units_per_pixel(self) -> float:
        return self.units_per_em / self.cfg.pixels_per_em

    def vectorize(
        self, spec: GlyphSpec, image: np.ndarray, advance_norm: float
    ) -> GlyphOutline:
        """Trace one glyph raster into font coordinates.

        ``advance_norm`` is the advance width as a fraction of the canvas, which
        is how both the renderer and the model's width head express it.
        """
        cfg = self.cfg
        size = cfg.image_size
        scale = self.units_per_pixel
        baseline_px = size * cfg.baseline

        prepared = clean_raster(image) if cfg.clean else image

        advance_px = advance_norm * size
        # Widen an advance the ink does not fit inside, *before* placing the
        # glyph, so the ink stays centred in whatever advance it ends up with.
        # Corrected afterwards it would shift the letter off its own centre.
        # Overhang is normal typography and is left alone — Times' `f` reaches
        # well past its advance — but ink wider than the whole advance sets
        # text that collides, which the predicted widths do on ~13% of glyphs
        # for a hand whose proportions the corpus does not cover.
        columns = np.nonzero((prepared > cfg.trace.level).any(axis=0))[0]
        if columns.size:
            clearance = 2 * cfg.min_side_bearing / self.units_per_pixel
            advance_px = max(advance_px, columns[-1] - columns[0] + 1 + clearance)

        # The renderer centres each glyph's advance box on the canvas, so the
        # left edge of the advance is here:
        origin_x = (size - advance_px) / 2.0

        def to_font(pt: tuple[float, float]) -> tuple[float, float]:
            x, y = pt
            return ((x - origin_x) * scale, (baseline_px - y) * scale)

        paths = [p.transform(to_font) for p in trace_image(prepared, cfg.trace)]
        paths = _orient(paths)

        advance_units = int(round(advance_px * scale))
        return GlyphOutline(spec=spec, paths=paths, advance=max(advance_units, 0))


def _draw(outline: GlyphOutline, pen: T2CharStringPen) -> None:
    for path in outline.paths:
        pen.moveTo(path.start)
        for c1, c2, end in path.segments:
            pen.curveTo(c1, c2, end)
        pen.closePath()


def build_font(
    outlines: list[GlyphOutline],
    metadata: FontMetadata | None = None,
    config: BuildConfig | None = None,
) -> "FontBuilder":
    """Assemble outlines into a complete CFF OpenType font."""
    meta = metadata or FontMetadata()
    cfg = config or BuildConfig()

    by_name: dict[str, GlyphOutline] = {}
    for outline in outlines:
        by_name[outline.name] = outline

    has_space = "space" in by_name
    glyph_order = [".notdef"] + sorted(by_name) + ([] if has_space else ["space"])

    fb = FontBuilder(meta.units_per_em, isTTF=False)
    fb.setupGlyphOrder(glyph_order)

    # cmap maps codepoints to glyph names. Only isolated forms are mapped
    # directly; contextual forms (Phase 2) are reached through GSUB instead, so
    # they deliberately have no cmap entry.
    cmap = {
        outline.spec.codepoint: name
        for name, outline in by_name.items()
        if outline.spec.form.value == "isol"
    }

    advances: dict[str, int] = {".notdef": meta.units_per_em // 2}
    charstrings: dict[str, object] = {}

    notdef_pen = T2CharStringPen(advances[".notdef"], None)
    charstrings[".notdef"] = notdef_pen.getCharString()

    for name in sorted(by_name):
        outline = by_name[name]
        advance = _clamped_advance(outline, cfg, meta)
        advances[name] = advance
        pen = T2CharStringPen(advance, None)
        _draw(outline, pen)
        charstrings[name] = pen.getCharString()

    # Space is synthesized, never generated.
    #
    # It has no outline, so there is nothing for the model to draw and nothing
    # for the tracer to trace — it is purely an advance width. Leaving it to the
    # charset means the exported font has no U+0020 at all, which no viewer
    # complains about and which makes the font useless for actual text: every
    # word runs into the next. Its width is derived from the glyphs that were
    # generated, so it stays in proportion with the hand rather than being a
    # fixed fraction of the em.
    if not has_space:
        advances["space"] = _space_advance(by_name, meta)
        charstrings["space"] = T2CharStringPen(advances["space"], None).getCharString()
        cmap[0x20] = "space"

    fb.setupCharacterMap(cmap)

    fb.setupCFF(
        meta.postscript_name,
        {
            "FullName": f"{meta.family} {meta.style}",
            "FamilyName": meta.family,
            "Weight": meta.style,
            "version": meta.version,
        },
        charstrings,
        {},
    )

    metrics = {name: (advances[name], _left_bearing(by_name.get(name))) for name in glyph_order}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(
        ascent=meta.ascender, descent=meta.descender, lineGap=meta.line_gap
    )

    fb.setupNameTable(
        {
            "familyName": meta.family,
            "styleName": meta.style,
            "uniqueFontIdentifier": f"{meta.postscript_name};{meta.version}",
            "fullName": f"{meta.family} {meta.style}",
            "psName": meta.postscript_name,
            "version": f"Version {meta.version}",
            "copyright": meta.copyright or "Generated from a handwriting sample.",
            "designer": meta.designer or "",
        }
    )
    fb.setupOS2(
        sTypoAscender=meta.ascender,
        sTypoDescender=meta.descender,
        sTypoLineGap=meta.line_gap,
        usWinAscent=meta.ascender,
        usWinDescent=abs(meta.descender),
        sxHeight=_estimate_x_height(by_name, meta),
        sCapHeight=_estimate_cap_height(by_name, meta),
        achVendID="HFNT",
        fsType=0,
    )
    fb.setupPost(isFixedPitch=0)
    return fb


def _bbox(outline: GlyphOutline | None) -> tuple[float, float, float, float] | None:
    if outline is None or not outline.paths:
        return None
    pts = np.concatenate([np.asarray(p.points, dtype=np.float64) for p in outline.paths])
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def _left_bearing(outline: GlyphOutline | None) -> int:
    box = _bbox(outline)
    return otRound(box[0]) if box else 0


def _clamped_advance(outline: GlyphOutline, cfg: BuildConfig, meta: FontMetadata) -> int:
    """Sanity-bound the advance without disturbing the glyph's position.

    Ink overhanging the advance is normal typography, not an error — Times' own
    ``f`` reaches about 30% past its advance, and italics do it constantly. An
    earlier version widened any advance the ink exceeded, which collided with
    how glyphs are framed: the renderer centres the *advance box* on the canvas,
    so changing the advance moves the glyph, and the exported font no longer
    matched the raster it was traced from.

    So overhang is left alone. The only case still corrected is an advance that
    is degenerate or absurd relative to the ink, which would make text unsettable
    rather than merely tight.
    """
    box = _bbox(outline)
    advance = max(outline.advance, 0)
    if box is None:
        return advance
    ink_width = box[2] - box[0]
    if advance <= 0 and ink_width > 0:
        return otRound(ink_width) + 2 * cfg.min_side_bearing
    return advance


def _space_advance(by_name: dict[str, GlyphOutline], meta: FontMetadata) -> int:
    """Pick a word space in proportion with the generated letterforms.

    Roughly a quarter of an em is the usual starting point for a text face, but
    keying off the font's own lowercase widths tracks a condensed or a wide hand
    instead of setting every style the same. ``n`` and ``o`` are the
    conventional reference widths.
    """
    references = [by_name[n].advance for n in ("n", "o", "e", "a") if n in by_name]
    if references:
        return max(int(round(sum(references) / len(references) * 0.55)), 1)
    lowercase = [
        outline.advance for name, outline in by_name.items()
        if len(name) == 1 and name.islower() and outline.advance > 0
    ]
    if lowercase:
        return max(int(round(sorted(lowercase)[len(lowercase) // 2] * 0.55)), 1)
    return int(meta.units_per_em * 0.26)


def _estimate_x_height(by_name: dict[str, GlyphOutline], meta: FontMetadata) -> int:
    box = _bbox(by_name.get("x")) or _bbox(by_name.get("o"))
    return otRound(box[3]) if box else meta.units_per_em // 2


def _estimate_cap_height(by_name: dict[str, GlyphOutline], meta: FontMetadata) -> int:
    box = _bbox(by_name.get("H")) or _bbox(by_name.get("E"))
    return otRound(box[3]) if box else int(meta.units_per_em * 0.7)


def save_font(fb: "FontBuilder", path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(path))
    return path


def rasters_to_font(
    rasters: dict[str, np.ndarray],
    advances: dict[str, float],
    charset: Charset,
    metadata: FontMetadata | None = None,
    config: BuildConfig | None = None,
) -> "FontBuilder":
    """Convenience path: generated rasters keyed by glyph key -> font."""
    cfg = config or BuildConfig()
    meta = metadata or FontMetadata()
    vectorizer = GlyphVectorizer(cfg, meta.units_per_em)
    advances = _tracked(rasters, advances, cfg)

    outlines: list[GlyphOutline] = []
    for spec in charset:
        image = rasters.get(spec.key)
        if image is None:
            continue
        outlines.append(vectorizer.vectorize(spec, image, advances.get(spec.key, 0.5)))
    return build_font(outlines, meta, cfg)


def _tracked(
    rasters: dict[str, np.ndarray], advances: dict[str, float], cfg: BuildConfig
) -> dict[str, float]:
    """Open the whole font up if its predicted advances set text too tightly.

    Real faces leave a median of 0.05-0.12 em between a glyph's advance and its
    ink; the width head, on a hand whose proportions are unlike any font in the
    corpus, has predicted as little as 0.015, which sets text with letters
    touching. The correction is one constant added to every advance — tracking,
    in the typographic sense — rather than a floor per glyph, so the model's
    relative widths, which are what it is actually good at, survive intact.

    Only ever loosens: a font already set comfortably is left alone.
    """
    canvas_per_em = cfg.pixels_per_em / cfg.image_size
    gaps = []
    for key, advance in advances.items():
        image = rasters.get(key)
        if image is None:
            continue
        columns = np.nonzero((image > cfg.trace.level).any(axis=0))[0]
        if columns.size:
            gaps.append(advance - (columns[-1] - columns[0] + 1) / cfg.image_size)
    if not gaps:
        return advances

    short = cfg.min_tracking / canvas_per_em - float(np.median(gaps))
    if short <= 0:
        return advances
    log.info("advances set %.3f em too tight; tracking the whole font open",
             short * canvas_per_em)
    return {key: advance + short for key, advance in advances.items()}
