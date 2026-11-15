"""A handwriting font whose Arabic letters join, built from pen centrelines.

Arabic letters connect, and a font makes them connect in one of two ways: the
designer draws every connecting stroke to end at one fixed height, or the font
carries, for each glyph, the point where the pen arrives (entry) and the point
where it leaves (exit), and the text engine moves each letter so its exit
meets the next letter's entry (OpenType ``curs``). Handwriting does not keep
one height, so this module does the second.

With traced outlines those two points have to be guessed from where a blob of
ink crosses the glyph's edge. Here each letter is first reduced to its pen
path (``hfont.generate.centreline``), and the points are simply the ends of
that path on the joining sides: Arabic runs right to left, so the pen arrives
at a letter's right and leaves at its left. The glyph's box is cut exactly at
those two points, the path is drawn again at the writer's pen weight, and the
points are written as ``curs`` anchors. A round pen cap overlaps the neighbour
by the pen's half-width, which hides the seam.

Which forms join where (the shaper picks the form; see ``arabic.py``):

    isol  neither side      init  leaves on the left
    medi  both sides        fina  arrives on the right
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..charset import Form, GlyphSpec
from ..vector.trace import trace_image
from .build import BuildConfig, FontMetadata, GlyphOutline, _orient, build_font

log = logging.getLogger(__name__)

JOINS_PREVIOUS = {Form.MEDI, Form.FINA}     # the pen arrives: entry, on the right
JOINS_NEXT = {Form.INIT, Form.MEDI}         # the pen leaves: exit, on the left


@dataclass
class JoinedGlyph:
    """One glyph in font units, with where the pen arrives and leaves."""

    outline: GlyphOutline
    entry: tuple[int, int] | None = None
    exit: tuple[int, int] | None = None


def join_points(line: np.ndarray, form: Form) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """(entry, exit) of a pen path as (x, y) pixels: its rightmost and leftmost points.

    Only the sides this form joins on get a point. Among path pixels in the
    extreme column the lowest is taken: a connecting stroke runs along the
    writing line, under whatever else reaches that far.
    """
    # Dots and other detached marks are not where the pen joins: only the
    # letter's body, the largest connected piece of the path, is considered.
    line = _body(line)
    ys, xs = np.nonzero(line)
    if xs.size == 0:
        return None, None

    def extreme(column: int) -> tuple[int, int]:
        return int(column), int(ys[xs == column].max())

    entry = extreme(xs.max()) if form in JOINS_PREVIOUS else None
    leave = extreme(xs.min()) if form in JOINS_NEXT else None
    return entry, leave


def _body(line: np.ndarray) -> np.ndarray:
    """The letter's body: the largest connected piece of its pen path."""
    from scipy.ndimage import label

    pieces, n = label(line, np.ones((3, 3), int))
    if n <= 1:
        return line.copy()
    return pieces == 1 + int(np.argmax(np.bincount(pieces.ravel())[1:]))


def retarget(line: np.ndarray, form: Form, baseline_row: float, stub: float,
             tolerance: float):
    """Replace each connecting tail by a short stub that ends on the writing line.

    A letter form written on its own gets a long connecting tail, ending at
    whatever height the pen stopped: set as text, words come out stretched,
    and since each join passes its height error on to the next, a long word
    climbs off the line. On each joining side the tail is the stretch of the
    body, from its end inwards, that is a single stroke staying within
    ``tolerance`` pixels of the row where it ends. It is cut where the letter
    proper begins, and a straight stub ``stub`` pixels long is drawn from
    there to the writing line. Every join then sits at one height.

    Returns ``(path, (entry, exit))``: the stub tips are the join points.
    They are not searched for again afterwards, because cutting a long tail
    can leave its stroke smaller than another (a kaf's separate upper stroke,
    yeh's dots written as one swoosh), and the largest piece is then the
    wrong one.
    """
    from skimage.draw import line as draw_line

    out = line.copy()
    tips: dict[int, tuple[int, int]] = {}
    for side, joins in ((+1, form in JOINS_PREVIOUS), (-1, form in JOINS_NEXT)):
        if not joins:
            continue
        body = _body(out)
        ys, xs = np.nonzero(body)
        if xs.size == 0:
            continue
        end_x = xs.max() if side > 0 else xs.min()
        end_y = int(ys[xs == end_x].max())
        x, y = int(end_x), end_y
        while True:                                     # walk inwards along the tail
            nxt = x - side
            column = ys[xs == nxt]
            if column.size == 0 or column.max() - column.min() > 2:
                break
            ny = int(column.max())
            if abs(ny - y) > 2 or abs(ny - end_y) > tolerance:
                break
            x, y = nxt, ny
        if abs(end_x - x) > stub:                       # a tail longer than the stub: cut it
            beyond = (xs > x) if side > 0 else (xs < x)
            out[ys[beyond], xs[beyond]] = False
        else:                                           # too short to cut: extend from its end
            x, y = int(end_x), end_y
        tip_x = int(np.clip(x + side * int(round(stub)), 0, out.shape[1] - 1))
        tip_y = int(np.clip(round(baseline_row), 0, out.shape[0] - 1))
        # The stub's last step is flat, so its tip is alone in the end column
        # and is the point ``join_points`` takes, however steep the stub.
        rr, cc = draw_line(y, x, tip_y, int(np.clip(tip_x - side, 0, out.shape[1] - 1)))
        out[rr, cc] = True
        out[tip_y, tip_x] = True
        tips[side] = (tip_x, tip_y)
    return _pull_in(out, tips.get(+1), tips.get(-1)), (tips.get(+1), tips.get(-1))


def _pull_in(path: np.ndarray, entry, leave) -> np.ndarray:
    """Slide detached pieces (dots, a lifted stroke) back inside the joins.

    A writer may put a dot under the middle of a long connecting stroke (a
    beh written on its own); once the tail is cut, the dot is left hanging
    out past the join, over the neighbouring letter. The pieces stranded on
    each side are moved sideways together, as one group, just far enough to
    sit within the joins: two or three dots keep their spacing instead of
    landing on one spot (a medial yeh's two dots, written under its tail,
    became one sliver). A group wider than the room between the joins is
    centred between them.
    """
    from scipy.ndimage import label

    pieces, n = label(path, np.ones((3, 3), int))
    if n < 2:
        return path
    keep = {pieces[t[1], t[0]] for t in (entry, leave) if t is not None}
    left = leave[0] if leave is not None else None
    right = entry[0] if entry is not None else None
    stranded: dict[int, list[int]] = {-1: [], +1: []}
    for i in range(1, n + 1):
        if i in keep:
            continue
        xs = np.nonzero(pieces == i)[1]
        if left is not None and xs.min() < left:
            stranded[-1].append(i)
        elif right is not None and xs.max() > right:
            stranded[+1].append(i)
    out = path.copy()
    for side, group in stranded.items():
        if not group:
            continue
        ys, xs = np.nonzero(np.isin(pieces, group))
        lo, hi = int(xs.min()), int(xs.max())
        if left is not None and right is not None and hi - lo > right - left - 2:
            shift = (left + right) // 2 - (lo + hi) // 2
        elif side < 0:
            shift = left + 1 - lo
            if right is not None and hi + shift > right:
                shift = right - 1 - hi
        else:
            shift = right - 1 - hi
            if left is not None and lo + shift < left:
                shift = left + 1 - lo
        shift = _clear_shift(out, ys, xs, shift, left, right)
        out[ys, xs] = False
        out[ys, np.clip(xs + shift, 0, path.shape[1] - 1)] = True
    return out


def _clear_shift(path: np.ndarray, ys: np.ndarray, xs: np.ndarray, shift: int,
                 left: int | None, right: int | None, gap: float = 2.0) -> int:
    """The shift nearest ``shift`` that keeps the pieces at ``ys, xs`` inside the
    joins and ``gap`` px clear of the rest of the path; ``shift`` itself when none is.

    Sliding stranded dots in can land them on the letter's own ink (teh's two
    dots on the tooth of its initial form) and they merge with it.
    """
    from scipy.ndimage import distance_transform_edt

    rest = path.copy()
    rest[ys, xs] = False
    if not rest.any():
        return shift
    clearance = distance_transform_edt(~rest)
    lo = (left + 1 - xs.min()) if left is not None else -path.shape[1]
    hi = (right - 1 - xs.max()) if right is not None else path.shape[1]
    for s in sorted(range(int(lo), int(hi) + 1), key=lambda s: (abs(s - shift), s)):
        cols = xs + s
        if cols.min() < 0 or cols.max() >= path.shape[1]:
            continue
        if clearance[ys, cols].min() >= gap:
            return s
    return shift


def joined_glyph(spec: GlyphSpec, image: np.ndarray, radius: float,
                 config: BuildConfig | None = None, units_per_em: int = 1000,
                 bearing: float = 0.04, baseline_px: float | None = None,
                 scale: float | None = None, stub: float | None = None) -> JoinedGlyph | None:
    """Trace one written letter into a glyph that can join its neighbours.

    ``image`` is the letter as written, ink 1, any size. ``radius`` is the
    pen's half-width in pixels. ``baseline_px`` is the row of the writing line
    and ``scale`` the font units per pixel; both default to the usual model
    canvas (``config``). A side that joins is cut at the pen path's end; a
    free side gets ``bearing`` em. ``stub``, in em, turns each connecting tail
    into a stub of that length ending on the writing line (``retarget``);
    None leaves the tails as written.
    """
    from scipy.ndimage import distance_transform_edt

    from ..generate import _draw as draw_pen
    from ..generate import centreline

    cfg = config or BuildConfig()
    scale = scale if scale is not None else units_per_em / cfg.pixels_per_em
    if baseline_px is None:
        baseline_px = cfg.image_size * cfg.baseline
    line = centreline(image)
    if line is None:
        return None
    if stub is not None:
        line, (entry, leave) = retarget(line, spec.form, baseline_px, stub * units_per_em / scale,
                                        tolerance=0.08 * units_per_em / scale)
    else:
        entry, leave = join_points(line, spec.form)
    redrawn = draw_pen(distance_transform_edt(~line), radius)

    columns = np.nonzero((redrawn > cfg.trace.level).any(axis=0))[0]
    free = bearing * units_per_em / scale
    left = float(leave[0]) if leave else columns[0] - free
    right = float(entry[0]) if entry else columns[-1] + 1 + free

    def to_font(pt: tuple[float, float]) -> tuple[float, float]:
        return ((pt[0] - left) * scale, (baseline_px - pt[1]) * scale)

    paths = _orient([p.transform(to_font) for p in trace_image(redrawn, cfg.trace)])
    advance = int(round((right - left) * scale))

    def anchor(point) -> tuple[int, int] | None:
        if point is None:
            return None
        x, y = to_font((point[0] + 0.5, point[1] + 0.5))
        return int(round(x)), int(round(y))

    glyph = JoinedGlyph(GlyphOutline(spec=spec, paths=paths, advance=advance),
                        entry=anchor(entry), exit=anchor(leave))
    # The join sits exactly on the glyph's edge, whatever rounding did above.
    if glyph.entry:
        glyph.entry = (advance, glyph.entry[1])
    if glyph.exit:
        glyph.exit = (0, glyph.exit[1])
    return glyph


def cursive_feature_code(glyphs: list[JoinedGlyph]) -> str:
    """OpenType ``curs``: each letter's exit is moved onto the next letter's entry.

    ``RightToLeft`` makes the *last* letter of a joined run sit on the
    baseline, with earlier ones stepping up or down from it, as a hand does.
    """
    rules = []
    for g in sorted(glyphs, key=lambda g: g.outline.name):
        if g.entry is None and g.exit is None:
            continue
        entry = f"<anchor {g.entry[0]} {g.entry[1]}>" if g.entry else "<anchor NULL>"
        leave = f"<anchor {g.exit[0]} {g.exit[1]}>" if g.exit else "<anchor NULL>"
        rules.append(f"    pos cursive {g.outline.name} {entry} {leave};")
    if not rules:
        return ""
    return "\n".join(["feature curs {", "    lookupflag RightToLeft IgnoreMarks;", *rules,
                      "} curs;"]) + "\n"


def build_joined_font(images: dict[GlyphSpec, np.ndarray], radius: float,
                      metadata: FontMetadata | None = None,
                      config: BuildConfig | None = None, lam: str | None = None,
                      alefs: dict[str, str] | None = None,
                      baselines: dict[GlyphSpec, float] | None = None,
                      scale: float | None = None, stub: float | None = None):
    """Written letters, keyed by letter and positional form, to a joining font.

    ``baselines`` gives each image's writing-line row and ``scale`` the font
    units per pixel, for letters not on the model canvas (see ``joined_glyph``).

    Returns ``(ttfont, glyphs)``: a compiled font with positional forms
    (``init``/``medi``/``fina``), the lam-alef ligatures given by ``lam`` and
    ``alefs`` (see ``arabic.arabic_feature_code``), and cursive attachment.
    """
    import io

    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
    from fontTools.ttLib import TTFont

    from .arabic import arabic_feature_code

    meta = metadata or FontMetadata()
    cfg = config or BuildConfig()
    glyphs = [g for spec, image in sorted(images.items())
              if (g := joined_glyph(spec, image, radius, cfg, meta.units_per_em,
                                    baseline_px=(baselines or {}).get(spec), scale=scale,
                                    stub=stub)) is not None]
    buffer = io.BytesIO()
    build_font([g.outline for g in glyphs], meta, cfg).save(buffer)
    buffer.seek(0)
    font = TTFont(buffer)
    code = arabic_feature_code(font.getGlyphOrder(), lam or "", alefs or {})
    code += "\n" + cursive_feature_code(glyphs)
    addOpenTypeFeaturesFromString(font, code)
    log.info("joined font: %d glyphs, %d with join points", len(glyphs),
             sum(1 for g in glyphs if g.entry or g.exit))
    return font, glyphs
