"""A joining Arabic font from one AHAWP writer's letters (step A1).

Takes one copy of each of the writer's 65 letter forms, reduces each to its
pen path, cuts the glyph at the path's ends on its joining sides, redraws it
at the writer's pen weight and writes the ends as cursive-attachment anchors
(``hfont.fontbuild.joined``). No model is involved.

    python scripts/ahawp_font.py --writer user001 --out output/ahawp/user001.otf \\
        --preview output/ahawp/user001.png

The preview shapes a few words with HarfBuzz and draws them from the font's
own outlines, so what it shows is what a text engine would set. It is an
image of real handwriting: keep it out of git (``output/`` is ignored).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import GlyphSpec  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.fontbuild.build import FontMetadata  # noqa: E402
from hfont.fontbuild import rasm  # noqa: E402
from hfont.fontbuild.joined import build_joined_font  # noqa: E402

WORDS = ["سلام عليكم", "كلب جميل", "قلم", "لا", "عسل", "ملك"]
#: Words that need letters AHAWP does not hold, composed from shared bodies.
COMPOSED_WORDS = ["تفاحة", "شمس", "خبز", "ذهب", "ظل", "غرفة", "ثلاثة", "حتى", "أرض"]
ALEF_HEIGHT = 650          # font units an isolated alef stands tall
STUB = 0.06                # em: length of each connecting stub (two meet at a join)
DOT_SIZE = 0.4             # alef heights: the largest a dot can be


def writer_font(archive, names, family: str, copy: int | None = None, compose: bool = True):
    """(ttfont, glyphs, letters) for one writer; ``copy`` picks a repeat, None the typical one.

    ``compose`` adds the letters AHAWP does not hold (ت ث ح خ ...) from the
    writer's own bodies and dots (``hfont.fontbuild.rasm``)."""
    return letters_font(ahawp.writer_letters(archive, names, copy), family, compose)


def final_lam_alef(ink: np.ndarray) -> np.ndarray:
    """A written isolated lam-alef made ready to join on its right: a short connecting stroke,
    at the writer's pen width, from the right end of its base out past its right edge, along
    the row the base sits on.

    Joined forms take their connection at their rightmost point. A lam-alef's alef may lean
    out to the right (a real sheet: a long curve rising to the right), so that point was half
    way up the alef, the join went there and the alef was cut as if it were a connecting tail.
    A written final lam-alef joins at its base instead.
    """
    from scipy.ndimage import label

    m = np.asarray(ink) > 0.5
    pieces, n = label(m, np.ones((3, 3), int))
    if n == 0:
        return np.asarray(ink, np.float32)
    body = pieces == 1 + int(np.argmax(np.bincount(pieces.ravel())[1:]))
    rows, cols = np.nonzero(body)
    height, width = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    bottom = rows.max()
    near = rows >= bottom - max(3, int(0.06 * height))
    x0 = int(cols[near].max())                               # the right end of the base
    y = int(round(rows[near][cols[near] >= x0 - 2].mean()))
    radius = max(1, int(round(ahawp.pen_radius({ahawp.LAM_ALEF: m.astype(np.float32)}))))
    reach = int(cols.max()) + max(8, int(0.15 * width))     # out past the ligature's right edge
    out = np.zeros((m.shape[0], max(m.shape[1], reach + radius + 2)), np.float32)
    out[:, :m.shape[1]] = m
    out[max(y - radius, 0):y + radius + 1, x0:reach + 1] = 1.0
    return out


def with_final_lam_alef(letters: dict) -> dict:
    """``letters`` with lam-alef's final form (after a joining letter, as in سلام) made from its
    isolated form when only that was written.

    Arabic always writes lam followed by alef as one ligature; the font's rule for the final
    position needs a final ligature glyph, and without one سلام was set as a separate medial
    lam and final alef. The written ligature is used, given a connection at its base
    (``final_lam_alef``).
    """
    from hfont.charset import Form

    final = GlyphSpec(ahawp.LAM_ALEF.char, Form.FINA)
    if ahawp.LAM_ALEF in letters and final not in letters:
        letters = {**letters, final: final_lam_alef(letters[ahawp.LAM_ALEF])}
    return letters


def letters_font(letters: dict, family: str, compose: bool = True):
    """(ttfont, glyphs, letters) from one writer's letter images, keyed by GlyphSpec,
    framed as AHAWP's scans (from the archive, or a photographed sheet)."""
    letters = with_final_lam_alef(letters)
    lines = ahawp.writing_lines(letters)
    alef = letters.get(GlyphSpec("ا"))
    rows = np.nonzero(alef.any(axis=1))[0] if alef is not None else np.array([])
    if rows.size < 2:
        rows = np.array([0, int(np.median([np.ptp(np.nonzero(im.any(axis=1))[0]) for im in letters.values()]))])
    alef_px = max(float(rows[-1] - rows[0] + 1), 8.0)
    scale = ALEF_HEIGHT / alef_px
    if compose:
        letters, lines = rasm.compose(letters, lines, dot_size=DOT_SIZE * alef_px,
                                      tolerance=0.12 * alef_px)
    lam = GlyphSpec("ل").name
    alefs = {GlyphSpec("ا").name: ahawp.LAM_ALEF.name}
    font, glyphs = build_joined_font(
        letters, radius=ahawp.pen_radius(letters), metadata=FontMetadata(family=family),
        lam=lam, alefs=alefs, baselines=lines, scale=scale, stub=STUB)
    return font, glyphs, letters


def versions_font(versions: list[dict], family: str, compose: bool = True, map_fn=None):
    """(ttfont, glyphs) from several versions of one writer's letters (step A6).

    ``versions[0]`` is the main set; each later dict holds another version of
    some forms (images keyed by GlyphSpec, framed as ``letters_font`` wants).
    Each version gets its own writing lines and composed letters; a form that
    differs from version 0 becomes an alternate (``GlyphSpec.variant``), which
    the font cycles through (``calt``). Scale and pen come from version 0.
    """
    versions = [with_final_lam_alef(versions[0])] + list(versions[1:])
    base = versions[0]
    alef = base.get(GlyphSpec("ا"))
    rows = np.nonzero(alef.any(axis=1))[0] if alef is not None else np.array([])
    if rows.size < 2:
        rows = np.array([0, int(np.median([np.ptp(np.nonzero(im.any(axis=1))[0]) for im in base.values()]))])
    alef_px = max(float(rows[-1] - rows[0] + 1), 8.0)
    images, baselines, first = {}, {}, {}
    for v, letters in enumerate(versions):
        letters = {**base, **letters}
        lines = ahawp.writing_lines(letters)
        if compose:
            letters, lines = rasm.compose(letters, lines, dot_size=DOT_SIZE * alef_px, tolerance=0.12 * alef_px)
        for spec, im in letters.items():
            if v and spec in first and first[spec].shape == im.shape and np.array_equal(first[spec], im):
                continue                                   # the same as version 0: no alternate
            if v == 0:
                first[spec] = im
            key = GlyphSpec(spec.char, spec.form, v)
            images[key], baselines[key] = im, lines[spec]
    font, glyphs = build_joined_font(
        images, radius=ahawp.pen_radius(base), metadata=FontMetadata(family=family),
        lam=GlyphSpec("ل").name, alefs={GlyphSpec("ا").name: ahawp.LAM_ALEF.name},
        baselines=baselines, scale=ALEF_HEIGHT / alef_px, stub=STUB, map_fn=map_fn)
    return font, glyphs


def shape(font_bytes: bytes, text: str, features: dict | None = None):
    """[(glyph name, x, y)] in visual order, and the run's total advance."""
    import uharfbuzz as hb
    from fontTools.ttLib import TTFont

    order = TTFont(io.BytesIO(font_bytes)).getGlyphOrder()
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(font_bytes)), buf, features or {})
    placed, cursor = [], 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        placed.append((order[info.codepoint], cursor + pos.x_offset, pos.y_offset))
        cursor += pos.x_advance
    return placed, cursor


def render(font_bytes: bytes, lines: list[str], px_per_em: int = 90,
           features: dict | None = None) -> np.ndarray:
    """Draw shaped lines from the font's outlines, as a grey image: each glyph even-odd filled,
    glyphs overlaid (where two joined letters overlap, ink, not a hole)."""
    from fontTools.pens.recordingPen import DecomposingRecordingPen
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageChops, ImageDraw

    font = TTFont(io.BytesIO(font_bytes))
    glyph_set = font.getGlyphSet()
    upm = font["head"].unitsPerEm
    k = px_per_em / upm

    def contours(name: str):
        pen = DecomposingRecordingPen(glyph_set)
        glyph_set[name].draw(pen)
        out, cur = [], []
        for op, args in pen.value:
            if op == "moveTo":
                cur = [args[0]]
            elif op == "lineTo":
                cur.append(args[0])
            elif op == "curveTo":
                p0, (p1, p2, p3) = cur[-1], args
                for t in np.linspace(0, 1, 9)[1:]:
                    cur.append(tuple((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
                                     + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)))
            elif op in ("closePath", "endPath") and len(cur) > 2:
                out.append(cur)
                cur = []
        return out

    runs = [shape(font_bytes, text, features) for text in lines]
    width = int(max(total for _, total in runs) * k) + 60
    line_h = int(1.9 * px_per_em)
    canvas = Image.new("1", (width, line_h * len(lines) + 20), 0)
    for row, (placed, total) in enumerate(runs):
        x0 = width - 30 - total * k                    # right-aligned, as Arabic is set
        y0 = 10 + row * line_h + 1.15 * px_per_em      # baseline
        for name, gx, gy in placed:
            glyph = Image.new("1", canvas.size, 0)
            for contour in contours(name):                 # even-odd within a glyph: its holes stay open
                layer = Image.new("1", canvas.size, 0)
                ImageDraw.Draw(layer).polygon(
                    [(x0 + (gx + x) * k, y0 - (gy + y) * k) for x, y in contour], fill=1)
                glyph = ImageChops.logical_xor(glyph, layer)
            canvas = ImageChops.logical_or(canvas, glyph)  # glyphs laid over each other, as text engines do
    return 255 - np.asarray(canvas, np.uint8) * 255


def check_joins(font_bytes: bytes, glyphs, words: list[str]) -> dict:
    """Shape ``words`` and measure every join: does exit meet entry, and how far
    do letters leave the writing line (font units)?"""
    by_name = {g.outline.name: g for g in glyphs}
    joins = misses = missing = 0
    drift = 0
    for word in words:
        for part in word.split():
            placed, _ = shape(font_bytes, part)
            missing += sum(1 for name, _, _ in placed if name not in by_name)
            placed = [p for p in placed if p[0] in by_name]
            if not placed:
                continue
            drift = max(drift, max(abs(y) for _, _, y in placed))
            for (left, lx, ly), (right, rx, ry) in zip(placed, placed[1:]):
                arrive, leave = by_name[left].entry, by_name[right].exit
                if arrive is None or leave is None:
                    continue                      # the earlier letter does not join forwards
                joins += 1
                if (lx + arrive[0], ly + arrive[1]) != (rx + leave[0], ry + leave[1]):
                    misses += 1
    return {"joins": joins, "misses": misses, "drift": drift, "missing": missing}


def overhangs(font_bytes: bytes, glyphs, limit: int = 150) -> int:
    """Joining glyphs whose ink reaches more than ``limit`` units past a join, over the neighbour."""
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(font_bytes))
    glyph_set, hmtx = font.getGlyphSet(), font["hmtx"]
    count = 0
    for g in glyphs:
        if not (g.entry or g.exit):
            continue
        pen = BoundsPen(glyph_set)
        glyph_set[g.outline.name].draw(pen)
        if pen.bounds is None:
            continue
        left, right = pen.bounds[0], pen.bounds[2]
        if (g.exit and -left > limit) or (g.entry and right - hmtx[g.outline.name][0] > limit):
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", default=str(ROOT / "data/ahawp/isolated_alphabets_per_user.zip"))
    ap.add_argument("--writer", default="user001")
    ap.add_argument("--all", action="store_true", help="build every writer's font and check its joins")
    ap.add_argument("--copy", type=int, default=None, help="which repeat to use (default: typical)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", default=None, help="PNG of shaped test words (private)")
    ap.add_argument("--no-compose", action="store_true",
                    help="only the letters AHAWP holds: no ت ث ح خ ... from shared bodies")
    args = ap.parse_args()
    compose = not args.no_compose
    words = WORDS + COMPOSED_WORDS
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    archive = ahawp.open_archive(args.archive)
    if args.all:
        writers = ahawp.index(archive)
        total = {"joins": 0, "misses": 0, "missing": 0}
        counts, drift, over = [], [], 0
        for writer, names in writers.items():
            font, glyphs, _ = writer_font(archive, names, f"AHAWP {writer}", args.copy, compose)
            buffer = io.BytesIO()
            font.save(buffer)
            result = check_joins(buffer.getvalue(), glyphs, words)
            total["joins"] += result["joins"]
            total["misses"] += result["misses"]
            total["missing"] += result["missing"]
            drift.append(result["drift"])
            counts.append(len(glyphs))
            over += overhangs(buffer.getvalue(), glyphs)
        print(f"{len(writers)} writers: {total['joins']} joins in {len(words)} test phrases, "
              f"{total['misses']} do not meet, {total['missing']} letters had no glyph; letters leave the line by at most "
              f"{max(drift)} units (median writer {int(np.median(drift))}); "
              f"glyphs per writer {min(counts)}-{max(counts)} (median {int(np.median(counts))}); "
              f"{over} joining glyphs reach over a neighbour (> 0.15 em past the join)")
        return 0
    names = ahawp.index(archive)[args.writer]
    font, glyphs, _ = writer_font(archive, names, f"AHAWP {args.writer}", args.copy, compose)
    buffer = io.BytesIO()
    font.save(buffer)
    data = buffer.getvalue()
    joined = sum(1 for g in glyphs if g.entry or g.exit)
    print(f"{args.writer}: {len(glyphs)} glyphs, {joined} with join points, {len(data) // 1024} KB")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(data)
    if args.preview:
        from PIL import Image

        with_curs = render(data, words)
        without = render(data, words, features={"curs": False})
        gap = np.full((with_curs.shape[0], 30), 255, np.uint8)
        width = max(with_curs.shape[1], without.shape[1])
        pad = lambda im: np.pad(im, ((0, 0), (width - im.shape[1], 0)), constant_values=255)  # noqa: E731
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate([pad(without), gap, pad(with_curs)], axis=1)).save(args.preview)
        print(f"preview (left: joins off, right: joins on) -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
