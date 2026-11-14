"""Arabic letters that differ only in their dots, composed from the writer's own.

Arabic's 28 letters are drawn on 18 bodies (the *rasm*): ب ت ث share one,
ح خ ج another, د ذ, ر ز, س ش, ص ض, ط ظ, ع غ, and ة is ه with two dots. A
sheet that holds one letter per body is therefore enough for the whole
alphabet, provided the dots are added the way this writer adds them.

Everything here works on the written letters themselves, before they are
reduced to pen paths, so a composed letter is a written body plus written
dots: nothing is drawn that the writer did not draw.

* ``split_dots`` separates a letter's dots from its body: the small pieces of
  ink that are not the letter's largest piece.
* ``dot_marks`` lifts the writer's single dot and pair of dots from letters
  that carry them (noon, feh, beh; qaf, yeh), and builds three dots as the
  pair with a single above, as a hand writes ث and ش.
* ``compose`` puts them on the shared bodies (``RASM``).

Placement: where the source letter has dots of its own (ب for ت and ث, ج for
خ) the new dots go at the same place along the letter, above it. Otherwise
they go over the middle of the part of the letter that rises above the
writing line (not the connecting tail), and for ظ beside its upright. The gap
under the dots is the writer's own, measured on letters written with dots
above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..charset import Form, GlyphSpec

ALL_FORMS = (Form.ISOL, Form.INIT, Form.MEDI, Form.FINA)


@dataclass(frozen=True)
class Rasm:
    """A letter drawn as ``source``'s body with ``dots`` dots above."""

    source: str
    dots: int
    forms: tuple[Form, ...] = ALL_FORMS
    beside_upright: bool = False


RASM: dict[str, Rasm] = {
    "ت": Rasm("ب", 2),
    "ث": Rasm("ب", 3),
    "ح": Rasm("ج", 0),
    "خ": Rasm("ج", 1),
    "ذ": Rasm("د", 1, (Form.ISOL, Form.FINA)),
    "ز": Rasm("ر", 1, (Form.ISOL, Form.FINA)),
    "ش": Rasm("س", 3),
    "ض": Rasm("ص", 1),
    "ظ": Rasm("ط", 1, beside_upright=True),
    "غ": Rasm("ع", 1),
    "ة": Rasm("ه", 2, (Form.ISOL, Form.FINA)),
    "ى": Rasm("ي", 0, (Form.ISOL, Form.FINA)),
}

#: Letters written with dots above, in the order their marks are preferred.
ONE_ABOVE = ("ن", "ف")
TWO_ABOVE = ("ق",)
#: Fallbacks: the same marks written below.
ONE_BELOW = ("ب",)
TWO_BELOW = ("ي",)
_FORM_PREFERENCE = (Form.INIT, Form.MEDI, Form.FINA, Form.ISOL)


def _pieces(image: np.ndarray):
    from scipy.ndimage import find_objects, label

    pieces, n = label(image > 0.5, np.ones((3, 3), int))
    return pieces, find_objects(pieces), np.bincount(pieces.ravel())[1:]


def split_dots(image: np.ndarray, dot_size: float) -> tuple[np.ndarray, np.ndarray]:
    """(body, dots) of a written letter, ink 1.

    The dots are the pieces of ink, other than the largest, that fit in a
    ``dot_size`` square. Other separate pieces (kaf's stroke, a lifted upright)
    stay with the body.
    """
    pieces, boxes, sizes = _pieces(image)
    dots = np.zeros(image.shape, bool)
    if len(boxes) < 2:
        return image.copy(), dots.astype(image.dtype)
    largest = int(np.argmax(sizes)) + 1
    for i, (rows, cols) in enumerate(boxes, start=1):
        if i != largest and max(rows.stop - rows.start, cols.stop - cols.start) <= dot_size:
            dots |= pieces == i
    body = image.copy()
    body[dots] = 0
    return body, dots.astype(image.dtype)


def _crop(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _first_mark(letters, chars, n: int, dot_size: float):
    """The dots of the first of ``chars`` (any form) written with ``n`` dot pieces or one joined mark."""
    for char in chars:
        for form in _FORM_PREFERENCE:
            image = letters.get(GlyphSpec(char, form))
            if image is None:
                continue
            _, dots = split_dots(image, dot_size)
            _, boxes, _ = _pieces(dots)
            if 1 <= len(boxes) <= n:
                return _crop(dots > 0.5)
    return None


def dot_marks(letters: dict[GlyphSpec, np.ndarray], dot_size: float) -> dict[int, np.ndarray]:
    """The writer's dot marks, 1, 2 and 3 dots, as cropped binary images."""
    one = _first_mark(letters, ONE_ABOVE + ONE_BELOW, 1, dot_size)
    two = _first_mark(letters, TWO_ABOVE + TWO_BELOW, 2, dot_size)
    if one is None:
        return {}
    if two is None:                                   # two singles side by side
        gap = max(1, one.shape[1] // 3)
        two = np.zeros((one.shape[0], 2 * one.shape[1] + gap), bool)
        two[:, :one.shape[1]] = one
        two[:, one.shape[1] + gap:] = one
    gap = max(1, one.shape[0] // 3)
    width = max(two.shape[1], one.shape[1])
    three = np.zeros((one.shape[0] + gap + two.shape[0], width), bool)
    left = (width - one.shape[1]) // 2
    three[:one.shape[0], left:left + one.shape[1]] = one
    left = (width - two.shape[1]) // 2
    three[one.shape[0] + gap:, left:left + two.shape[1]] = two
    return {1: one, 2: two, 3: three}


def _top_under(body: np.ndarray, x0: int, x1: int) -> int | None:
    rows = np.nonzero(body[:, max(x0, 0):max(x1, 1)].any(axis=1))[0]
    return int(rows[0]) if rows.size else None


def dot_gap(letters: dict[GlyphSpec, np.ndarray], dot_size: float) -> float | None:
    """Rows between the writer's dots and the body under them, on letters with dots above."""
    gaps = []
    for char in ONE_ABOVE + TWO_ABOVE:
        for form in ALL_FORMS:
            image = letters.get(GlyphSpec(char, form))
            if image is None:
                continue
            body, dots = split_dots(image, dot_size)
            rows, cols = np.nonzero(dots > 0.5)
            if rows.size == 0:
                continue
            top = _top_under(body > 0.5, cols.min(), cols.max() + 1)
            if top is not None and top > rows.max():
                gaps.append(top - rows.max() - 1)
    return float(np.median(gaps)) if gaps else None


def _place_x(body: np.ndarray, line: float, tolerance: float, source_dots: np.ndarray,
             beside_upright: bool) -> int:
    """Column the new dots are centred on."""
    ys, xs = np.nonzero(source_dots > 0.5)
    if xs.size:
        return int(round(xs.mean()))
    ys, xs = np.nonzero(body > 0.5)
    proper = ys < line - tolerance                    # the letter, not its tail on the line
    if proper.any():
        ys, xs = ys[proper], xs[proper]
    if beside_upright:
        upright = int(xs[np.argmin(ys)])              # the column reaching highest
        right = xs > upright + max(2, int(tolerance))
        if right.any():
            xs = xs[right]
    return int(np.median(xs))


def compose_one(body: np.ndarray, mark: np.ndarray | None, x: int, gap: float):
    """``body`` with ``mark`` centred on column ``x``, ``gap`` rows above the ink under it.

    Returns (image, rows added at the top), padding upwards when the dots
    would leave the image.
    """
    if mark is None:
        return body.copy(), 0
    h, w = mark.shape
    x0 = int(np.clip(x - w // 2, 0, max(body.shape[1] - w, 0)))
    top = _top_under(body > 0.5, x0, x0 + w)
    if top is None:
        top = int(np.nonzero((body > 0.5).any(axis=1))[0][0])
    y0 = int(round(top - gap - h))
    pad = max(0, -y0)
    out = np.pad(body, ((pad, 0), (0, max(0, x0 + w - body.shape[1]))))
    region = out[y0 + pad:y0 + pad + h, x0:x0 + w]
    region[mark[:region.shape[0], :region.shape[1]]] = 1
    return out, pad


def compose(letters: dict[GlyphSpec, np.ndarray], lines: dict[GlyphSpec, float],
            dot_size: float, tolerance: float, gap: float | None = None):
    """Add every ``RASM`` letter the writer did not write; returns (letters, lines) extended.

    ``dot_size`` bounds a dot (pixels), ``tolerance`` is how far above the
    writing line a stroke must rise to count as the letter rather than its
    connecting tail. Letters already present are left as written.
    """
    marks = dot_marks(letters, dot_size)
    if gap is None:
        gap = dot_gap(letters, dot_size)
    if gap is None:
        gap = dot_size / 3
    out, out_lines = dict(letters), dict(lines)
    for char, rule in RASM.items():
        if rule.dots and rule.dots not in marks:
            continue
        for form in rule.forms:
            spec, source = GlyphSpec(char, form), GlyphSpec(rule.source, form)
            if spec in letters or source not in letters or source not in lines:
                continue
            body, own = split_dots(letters[source], dot_size)
            mark = marks.get(rule.dots) if rule.dots else None
            x = _place_x(body, lines[source], tolerance, own, rule.beside_upright)
            image, pad = compose_one(body, mark, x, gap)
            out[spec] = image
            out_lines[spec] = lines[source] + pad
    return out, out_lines
