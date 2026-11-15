"""AHAWP: 82 writers, each writing 65 Arabic letter forms ten times.

Khan et al., "Arabic handwritten alphabets, words and paragraphs per user
(AHAWP) dataset", Data in Brief 2022, CC BY 4.0, Mendeley Data
``2h76672znt``. The letters archive (``isolated_alphabets_per_user.zip``,
204 MB) holds 53,199 grey scans, 128 px high and as wide as the letter, named
``user007_beh_middle_013.png``: writer, letter, positional form, copy. It is
the one public set in which the same hand writes every positional form, and
writes it more than once, which gives Arabic both a few-shot split (hold out
forms) and an own-second-copy ceiling from the start.

The 18 letters are the dotless-skeleton groups plus two extras, so the
alphabet's other letters (teh, theh, khah, ...) are not in it: they share a
body with one that is.

The scans carry no writing line. ``writing_lines`` recovers one per letter
from the letter's own connecting stroke (see there).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np

from ..charset import Form, GlyphSpec

LETTERS = {
    "ain": "ع", "alif": "ا", "beh": "ب", "dal": "د", "feh": "ف", "heh": "ه", "jeem": "ج",
    "kaf": "ك", "lam": "ل", "meem": "م", "noon": "ن", "qaf": "ق", "raa": "ر", "sad": "ص",
    "seen": "س", "tah": "ط", "waw": "و", "yaa": "ي",
}
FORMS = {"regular": Form.ISOL, "begin": Form.INIT, "middle": Form.MEDI, "end": Form.FINA}
#: Two entries that are not letter x form: alef with hamza above, and the
#: lam-alef ligature (its isolated presentation form, U+FEFB).
EXTRA = {"alif_hamza": GlyphSpec("أ"), "lam_alif": GlyphSpec("ﻻ")}
LAM_ALEF = EXTRA["lam_alif"]

_NAME = re.compile(r"(user\d+)_([a-z]+)_([a-z]+)_(\d+)\.png$")


def spec_for(letter: str, form: str) -> GlyphSpec | None:
    if f"{letter}_{form}" in EXTRA:
        return EXTRA[f"{letter}_{form}"]
    if letter in LETTERS and form in FORMS:
        return GlyphSpec(LETTERS[letter], FORMS[form])
    return None


def index(archive: zipfile.ZipFile) -> dict[str, dict[GlyphSpec, list[str]]]:
    """writer -> glyph -> archive names of its copies, in copy order."""
    writers: dict[str, dict[GlyphSpec, list[str]]] = {}
    for name in sorted(archive.namelist()):
        m = _NAME.search(name)
        if not m:
            continue
        writer, letter, form, _ = m.groups()
        spec = spec_for(letter, form)
        if spec is not None:
            writers.setdefault(writer, {}).setdefault(spec, []).append(name)
    return writers


def ink(png: bytes, min_piece: int = 6) -> np.ndarray:
    """A scan as binary ink (1 = pen), cut by Otsu's threshold, specks dropped."""
    from PIL import Image
    from scipy.ndimage import label
    from skimage.filters import threshold_otsu

    grey = np.asarray(Image.open(io.BytesIO(png)).convert("L"), np.float32)
    if grey.max() - grey.min() < 30:                  # a blank cell
        return np.zeros(grey.shape, np.float32)
    mask = grey < threshold_otsu(grey)
    pieces, n = label(mask, np.ones((3, 3), int))
    sizes = np.bincount(pieces.ravel())
    for i in range(1, n + 1):
        if sizes[i] < min_piece:
            mask[pieces == i] = False
    return _drop_border(mask).astype(np.float32)


def _drop_border(mask: np.ndarray, edge: int = 20, thin: int = 12) -> np.ndarray:
    """Remove the form's cell border caught in the crop.

    About 4% of scans (25 writers) carry a line 1-3 px thick running the
    scan's whole width along its top or bottom edge. It is not handwriting,
    and a connecting-stroke reader would take it for the writing line. A piece
    is dropped when it spans 80% of the width, is at most ``thin`` px tall,
    lies within ``edge`` px of the top or bottom, and other ink remains: a
    real stroke that long is joined to its letter, so it is never a piece of
    its own beside one.
    """
    from scipy.ndimage import find_objects, label

    pieces, n = label(mask, np.ones((3, 3), int))
    if n < 2:
        return mask
    height, width = mask.shape
    out = mask.copy()
    for i, box in enumerate(find_objects(pieces), start=1):
        rows, cols = box
        if (cols.stop - cols.start >= 0.8 * width and rows.stop - rows.start <= thin
                and (rows.start >= height - edge or rows.stop <= edge)):
            out[pieces == i] = False
    return out if out.any() else mask


def typical(copies: list[np.ndarray]) -> int:
    """Index of the copy nearest the median in size and ink: skips the scribbled ones."""
    def measure(im: np.ndarray) -> tuple[float, float, float]:
        ys, xs = np.nonzero(im)
        if xs.size == 0:
            return (0.0, 0.0, 0.0)
        return (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float(xs.size))

    m = np.array([measure(c) for c in copies])
    spread = np.maximum(np.median(m, axis=0), 1.0)
    return int(np.argmin((np.abs(m - np.median(m, axis=0)) / spread).sum(axis=1)))


def writer_letters(archive: zipfile.ZipFile, names: dict[GlyphSpec, list[str]],
                   copy: int | None = None) -> dict[GlyphSpec, np.ndarray]:
    """One ink image per glyph: the ``copy``-th, or the typical one when None."""
    out = {}
    for spec, files in names.items():
        if copy is not None:
            if copy < len(files):
                out[spec] = ink(archive.read(files[copy]))
            continue
        copies = [ink(archive.read(f)) for f in files]
        out[spec] = copies[typical(copies)]
    return {spec: im for spec, im in out.items() if im.sum() >= 20}


def writing_lines(letters: dict[GlyphSpec, np.ndarray]) -> dict[GlyphSpec, float]:
    """The row of the writing line in each letter's image.

    A joining form shows it: its connecting stroke runs along the line, so the
    line is at the form's join point (the mean of the two, for a medial form).
    An isolated form has no such stroke, but its final form is the same body
    with one, so the isolated letter is given the same drop from line to lowest
    ink as its final form, scaled by their heights. A letter with neither
    (hamza, the ligature) rests its lowest ink on the line.
    """
    from ..fontbuild.joined import join_points
    from ..generate import centreline

    def extent(im: np.ndarray) -> tuple[int, int]:
        rows = np.nonzero(im.any(axis=1))[0]
        return int(rows[0]), int(rows[-1])

    lines: dict[GlyphSpec, float] = {}
    for spec, im in letters.items():
        if spec.form is Form.ISOL:
            continue
        path = centreline(im)
        if path is None:
            continue
        points = [p for p in join_points(path, spec.form) if p is not None]
        if points:
            lines[spec] = float(np.mean([p[1] for p in points]))
    for spec, im in letters.items():
        if spec in lines:
            continue
        top, bottom = extent(im)
        final = GlyphSpec(spec.char, Form.FINA)
        if spec.form is Form.ISOL and final in lines:
            ftop, fbottom = extent(letters[final])
            drop = (fbottom - lines[final]) * (bottom - top + 1) / max(fbottom - ftop + 1, 1)
            lines[spec] = bottom - max(drop, 0.0)
        else:
            lines[spec] = float(bottom)
    return lines


def pen_radius(letters: dict[GlyphSpec, np.ndarray]) -> float:
    """The writer's pen half-width in pixels: ink depth along the pen paths."""
    from scipy.ndimage import distance_transform_edt

    from ..generate import centreline

    depths = []
    for im in letters.values():
        path = centreline(im)
        if path is not None:
            depths.append(distance_transform_edt(im > 0.5)[path])
    return float(np.median(np.concatenate(depths))) if depths else 2.0


def split_of(writer: str) -> str:
    """``train``, ``val`` or ``test`` for a writer, fixed by number.

    Writer numbers ending in 0 or 5 are the test (16 writers), those ending
    in 1 or 6 validation (17), the other 49 training. A writer is on one side
    only, so a model is always scored on hands it never saw.
    """
    n = int(writer.removeprefix("user"))
    return "test" if n % 5 == 0 else "val" if n % 5 == 1 else "train"


def open_archive(path: str | Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)
