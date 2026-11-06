"""Letters the writer never wrote, built from strokes they did write.

The model draws its own average letter for anything not on the sheet. Some
missing letters are, in most people's printing, another written letter turned
over or mirrored: ``u`` is ``n`` turned over, ``w`` is ``m`` turned over, ``d``
is ``b`` mirrored, ``p`` is ``b`` flipped top to bottom and ``q`` is ``b``
turned over, ``9`` is ``6`` turned over and ``W`` is ``M`` turned over. Built
that way the font has the writer's real strokes in seven more letters,
whatever the model can do.

**Measured, and not used in fonts.** scripts/compose_eval.py builds these
letters for 80 held-out SD19 writers from their seed letters and scores them
against what each writer really wrote (clDice, 501 letters, November 2026).
Composed 0.505, model 0.580, nearest font 0.578. Only ``9`` from ``6``
roughly ties (0.477 vs 0.468); the composed letter is closer than the model's
for 13-51% of writers depending on the letter. People do not write these
letters as turned-over copies: many write ``u`` as a rounded cup with no stem
and ``w`` as two cups, and mirroring or flipping reverses the slant. The code
is kept so the result can be re-checked, not wired into font assembly.

All images are ink fields on the model's canvas (1 = ink), with the baseline at
row ``size * baseline``. Lowercase turns are about the middle of the writer's
x-height band, so the bowl stays between baseline and x-height and an ascender
becomes a descender; capitals and digits turn about their own ink box.
"""

from __future__ import annotations

import numpy as np

#: target <- (donor, operation). Operations: "turn" (rotate 180 degrees),
#: "mirror" (left-right), "flip" (top-bottom).
DONORS: dict[str, tuple[str, str]] = {
    "u": ("n", "turn"),
    "w": ("m", "turn"),
    "d": ("b", "mirror"),
    "p": ("b", "flip"),
    "q": ("b", "turn"),
    "9": ("6", "turn"),
    "W": ("M", "turn"),
}

#: Lowercase letters whose height is the x-height.
X_HEIGHT = "acemnorsuvwxz"

#: Pixels kept clear at the canvas edge when a tail is fitted.
EDGE = 2.0


def _rows(image: np.ndarray) -> np.ndarray:
    return np.nonzero((image > 0.5).any(axis=1))[0]


def x_band(letters: dict[str, np.ndarray], line: float) -> tuple[float, float] | None:
    """(top, bottom) rows of the writer's x-height band, from their own letters."""
    tops = [rows[0] for ch, im in letters.items() if ch in X_HEIGHT and (rows := _rows(im)).size]
    if not tops:
        return None
    return float(np.median(tops)), line


def _shift(image: np.ndarray, rows: float, cols: float) -> np.ndarray:
    from scipy.ndimage import shift

    if abs(rows) < 1e-6 and abs(cols) < 1e-6:
        return image
    return np.clip(shift(image, (rows, cols), order=1, mode="constant", cval=0.0), 0, 1)


def _fit_below(image: np.ndarray, line: float) -> np.ndarray:
    """Squeeze whatever runs past the bottom edge back onto the canvas, toward the line.

    ``image`` is taller than the canvas (padded below); the part under the
    baseline is compressed just enough to fit, as intake does for long tails.
    """
    from skimage.transform import resize

    size = image.shape[1]
    rows = _rows(image)
    if not rows.size or rows[-1] + 1 <= size - EDGE:
        return image[:size]
    split = int(np.floor(line))
    below = image[split : rows[-1] + 1]
    room = int(size - EDGE - split)
    squeezed = resize(below, (room, size), order=1, anti_aliasing=True)
    out = image[:size].copy()
    out[split:] = 0.0
    out[split : split + room] = squeezed
    return np.clip(out, 0, 1).astype(np.float32)


def _ink_centre_col(image: np.ndarray) -> float:
    cols = np.nonzero((image > 0.5).any(axis=0))[0]
    return (cols[0] + cols[-1] + 1) / 2.0 if cols.size else image.shape[1] / 2.0


def transform(image: np.ndarray, op: str, axis_row: float | None) -> np.ndarray:
    """Turn, mirror or flip a glyph in place on its canvas.

    ``axis_row`` is the row the glyph turns or flips about (an edge position,
    so the middle of the x-height band); None turns it about its own ink box.
    The ink stays centred where the donor's was horizontally.
    """
    size = image.shape[0]
    centre = _ink_centre_col(image)
    rows = _rows(image)
    if axis_row is None and rows.size:
        axis_row = (rows[0] + rows[-1] + 1) / 2.0
    pad = size  # room for a tail to land before it is fitted
    tall = np.concatenate([image, np.zeros((pad, size), image.dtype)])
    if op == "mirror":
        out = tall[:, ::-1]
        return _shift(out, 0, 2 * centre - size)[:size]
    if op == "flip":
        out = tall[::-1]
    elif op == "turn":
        out = tall[::-1, ::-1]
    else:
        raise ValueError(f"unknown operation {op!r}")
    # Flipping the tall canvas maps edge position y to (size + pad) - y; move
    # the axis back to where it was.
    moved = _shift(out, 2 * axis_row - (size + pad), 0)
    if op == "turn":
        moved = _shift(moved, 0, 2 * centre - size)
    return moved


def compose(letters: dict[str, np.ndarray], line: float,
            wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    """Build every missing letter in ``DONORS`` whose donor the writer wrote.

    ``letters`` maps characters to the writer's glyphs; letters already
    written are never replaced.
    """
    band = x_band(letters, line)
    out: dict[str, np.ndarray] = {}
    for target, (donor, op) in DONORS.items():
        if target in letters or donor not in letters or (wanted is not None and target not in wanted):
            continue
        lowercase = target.islower()
        if lowercase and op != "mirror" and band is None:
            continue
        axis = (band[0] + band[1]) / 2.0 if lowercase and op != "mirror" else None
        image = transform(letters[donor].astype(np.float32), op, axis)
        if image.shape[0] > image.shape[1]:
            image = _fit_below(image, line)
        out[target] = image.astype(np.float32)
    return out
