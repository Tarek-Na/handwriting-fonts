"""Does the exported font actually work as a font?

Pixel metrics say whether the letterforms are right. They say nothing about
whether the file installs, whether a text engine can find the glyphs, or
whether the words come out correctly spaced — and it is entirely possible to
score well on the first and fail all of the second.

So this module shapes real text with HarfBuzz, which is the engine Chrome,
Firefox, Android and LibreOffice all use. If a string shapes here without
falling back to ``.notdef``, it will type correctly in those.

Phase 2 leans on this much harder than Phase 1. Latin needs almost no shaping,
which is precisely why Arabic's GSUB requirements have to be validated
separately and early rather than discovered at export time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: Deliberately mundane text. Pangrams exercise the alphabet; the mixed line
#: exercises the punctuation and digits that get forgotten until someone types
#: a price into a document.
DEFAULT_SAMPLES = (
    "Hamburgefonstiv",
    "The quick brown fox jumps over the lazy dog",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
    "Waltz, bad nymph, for quick jigs vex! (12 items; 34/50)",
)


@dataclass
class ShapingReport:
    font: str
    ok: bool
    tables: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    n_glyphs: int = 0
    cmap_size: int = 0
    notdef_chars: list[str] = field(default_factory=list)
    zero_advance_chars: list[str] = field(default_factory=list)
    #: Adjacent glyph pairs in the sample text whose ink actually intersects.
    #: Reported, never fatal: overhang is normal typography and only the build
    #: step's own clearance rule decides what counts as too much.
    overlapping_pairs: int = 0
    shaped_pairs: int = 0
    worst_overlap_em: float = 0.0
    overlap_examples: list[tuple[str, str]] = field(default_factory=list)
    out_of_charset: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[{status}] {self.font}",
            f"  glyphs={self.n_glyphs} cmap={self.cmap_size} tables={len(self.tables)}",
        ]
        if self.missing_tables:
            lines.append(f"  missing tables: {', '.join(self.missing_tables)}")
        fatal = [c for c in self.notdef_chars if c not in self.out_of_charset]
        if fatal:
            lines.append(f"  unmapped characters ({len(fatal)}): {''.join(fatal[:20])!r}")
        if self.out_of_charset:
            lines.append(
                f"  outside charset, not fatal ({len(self.out_of_charset)}): "
                f"{''.join(self.out_of_charset[:20])!r}"
            )
        if self.zero_advance_chars:
            shown = "".join(self.zero_advance_chars[:20])
            lines.append(f"  zero-advance glyphs ({len(self.zero_advance_chars)}): {shown!r}")
        if self.overlapping_pairs:
            shown = ", ".join(f"{a}{b}" for a, b in self.overlap_examples[:8])
            lines.append(
                f"  colliding pairs ({self.overlapping_pairs} of {self.shaped_pairs}, "
                f"worst {self.worst_overlap_em:.3f} em): {shown}"
            )
        for err in self.errors:
            lines.append(f"  error: {err}")
        return "\n".join(lines)


#: Tables a text engine needs before it will treat the file as a usable font.
REQUIRED_TABLES = ("head", "hhea", "maxp", "name", "cmap", "hmtx", "OS/2", "post")


def shape_text(font_path: str | Path, text: str) -> list[tuple[int, int, int]]:
    """Shape ``text`` and return (glyph id, cluster, advance) per glyph."""
    import uharfbuzz as hb

    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    font = hb.Font(face)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {})

    return [
        (info.codepoint, info.cluster, pos.x_advance)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions)
    ]


def ink_collisions(font_path: str | Path, samples: tuple[str, ...] = DEFAULT_SAMPLES):
    """Count adjacent glyph pairs in ``samples`` whose ink actually intersects.

    Per-glyph advances can each be defensible and still set text that collides:
    an advance says how far the pen moves, not where the ink is, so a wide
    letter followed by a wide letter overlaps whenever the sum of their
    overhangs exceeds the gap. The build step guards against the extreme case
    by widening an advance narrower than its own ink
    (:func:`hfont.fontbuild.build.GlyphVectorizer.vectorize`), but that rule is
    per-glyph and cannot see the pair.

    So this shapes the real strings with the real advances and compares ink
    boxes: for neighbours *i* and *i+1*, the pair collides when the right edge
    of *i*'s ink lands past the left edge of *i+1*'s. Returns the count, the
    number of pairs examined, the worst overlap in em, and the pairs themselves.

    Overhang is not an error in itself — Times' ``f`` reaches well past its
    advance — so the result is reported rather than treated as a failure.
    """
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    tt = TTFont(str(font_path), lazy=True)
    glyph_set = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    upem = int(tt["head"].unitsPerEm) or 1000

    bounds: dict[str, tuple[float, float] | None] = {}

    def ink_x(name: str) -> tuple[float, float] | None:
        if name not in bounds:
            pen = BoundsPen(glyph_set)
            try:
                glyph_set[name].draw(pen)
            except Exception:
                pen.bounds = None
            bounds[name] = (pen.bounds[0], pen.bounds[2]) if pen.bounds else None
        return bounds[name]

    collisions: list[tuple[str, str]] = []
    worst = 0.0
    pairs = 0
    for text in samples:
        shaped = shape_text(font_path, text)
        pen_x = 0.0
        placed = []
        for gid, cluster, advance in shaped:
            name = order[gid] if gid < len(order) else ".notdef"
            char = text[cluster] if cluster < len(text) else "?"
            placed.append((char, name, pen_x))
            pen_x += advance
        for (c1, n1, x1), (c2, n2, x2) in zip(placed, placed[1:]):
            b1, b2 = ink_x(n1), ink_x(n2)
            if b1 is None or b2 is None:
                continue  # a space has no ink and cannot collide
            pairs += 1
            gap = (x2 + b2[0]) - (x1 + b1[1])
            if gap < 0:
                collisions.append((c1, c2))
                worst = max(worst, -gap / upem)

    tt.close()
    return collisions, pairs, worst


def check_font(
    font_path: str | Path,
    samples: tuple[str, ...] = DEFAULT_SAMPLES,
    charset=None,
) -> ShapingReport:
    """Structural and shaping validation of an exported font.

    ``charset`` scopes what counts as a failure. A character the font was never
    meant to contain shaping to ``.notdef`` is a gap in the inventory, not a
    broken font, and conflating the two makes the check useless as a pass/fail
    gate — you stop trusting it the third time it fails for a missing currency
    symbol. Out-of-charset misses are still reported, just not fatal.
    """
    from fontTools.ttLib import TTFont

    path = Path(font_path)
    report = ShapingReport(font=path.name, ok=True)
    expected = set(charset.chars) | {" "} if charset is not None else None

    try:
        tt = TTFont(str(path), lazy=True)
    except Exception as exc:
        report.ok = False
        report.errors.append(f"will not open: {exc}")
        return report

    report.tables = sorted(tt.keys())
    report.missing_tables = [t for t in REQUIRED_TABLES if t not in tt]
    if report.missing_tables:
        report.ok = False

    try:
        report.n_glyphs = int(tt["maxp"].numGlyphs)
        cmap = tt.getBestCmap()
        report.cmap_size = len(cmap)
    except Exception as exc:
        report.ok = False
        report.errors.append(f"cmap/maxp unreadable: {exc}")
        cmap = {}

    # Shaping: every character of every sample must resolve to a real glyph.
    # Glyph id 0 is .notdef — the tofu box — and means the text engine could not
    # find the character, which is the single most common way a generated font
    # looks fine in a viewer and fails in a word processor.
    seen_notdef: list[str] = []
    zero_advance: list[str] = []
    try:
        for text in samples:
            shaped = shape_text(path, text)
            for gid, cluster, advance in shaped:
                char = text[cluster] if cluster < len(text) else "?"
                if gid == 0 and char not in seen_notdef:
                    seen_notdef.append(char)
                if advance == 0 and not char.isspace() and char not in zero_advance:
                    zero_advance.append(char)
    except ImportError:
        report.errors.append("uharfbuzz not installed; shaping not verified")
    except Exception as exc:
        report.ok = False
        report.errors.append(f"shaping failed: {exc}")

    report.notdef_chars = seen_notdef
    report.zero_advance_chars = zero_advance

    try:
        collisions, pairs, worst = ink_collisions(path, samples)
    except Exception as exc:  # a missing shaper must not fail the whole check
        report.errors.append(f"collision check skipped: {exc}")
    else:
        report.overlapping_pairs = len(collisions)
        report.shaped_pairs = pairs
        report.worst_overlap_em = worst
        seen: list[tuple[str, str]] = []
        for pair in collisions:
            if pair not in seen:
                seen.append(pair)
        report.overlap_examples = seen

    if expected is None:
        fatal_notdef = seen_notdef
        fatal_zero = zero_advance
    else:
        fatal_notdef = [c for c in seen_notdef if c in expected]
        fatal_zero = [c for c in zero_advance if c in expected]
        report.out_of_charset = [c for c in seen_notdef if c not in expected]
    if fatal_notdef or fatal_zero:
        report.ok = False

    tt.close()
    return report


def render_string(font_path: str | Path, text: str, size: int = 64, padding: int = 12):
    """Rasterize a shaped string, for eyeballing spacing and joins.

    Per-glyph metrics can all be correct while the words still look wrong, so
    this renders through the real shaper and real advances rather than tiling
    glyph images. In Phase 2 this is what a joining-quality metric will measure.
    """
    import numpy as np
    from fontTools.ttLib import TTFont

    from ..data.raster import FlatteningPen, fill_contours

    tt = TTFont(str(font_path), lazy=True)
    glyph_set = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    upem = tt["head"].unitsPerEm
    scale = size / upem

    shaped = shape_text(font_path, text)
    total_advance = sum(adv for _, _, adv in shaped) * scale
    width = int(total_advance) + padding * 2
    height = int(size * 1.6) + padding * 2
    baseline = int(size * 1.15) + padding

    contours = []
    pen_x = float(padding)
    for gid, _, advance in shaped:
        name = order[gid] if gid < len(order) else ".notdef"
        pen = FlatteningPen(glyph_set, scale=scale, offset_x=pen_x, offset_y=baseline)
        try:
            glyph_set[name].draw(pen)
        except Exception:
            pass
        contours.extend(pen.contours)
        pen_x += advance * scale

    tt.close()
    if width <= 0 or height <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    return fill_contours(contours, width, height, supersample=4)
