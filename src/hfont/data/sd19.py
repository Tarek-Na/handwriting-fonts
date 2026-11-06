"""NIST Special Database 19 as training "fonts": one per real writer.

SD19 holds 3,669 writers who each printed the alphabet once in boxes on a form
(lowercase field, uppercase field, digit fields). The 2nd edition's per-writer
images (``by_write.zip``) carry no labels, but the 1995 1st edition has a
``.cls`` label file for every field, and the 2nd edition names each PNG after
its field plus the image's index in that field (``l1454_12_00007.png``). So the
k-th label in ``l1454_12.cls`` belongs to ``l1454_12_0000k.png``. Checked on
writer f1454_12 by eye (all 47 letters) before this was written, and against
by_field by pixel hash (scripts/sd19_labels.py).

``.cls`` format: the first line is the number of images, then one line per
image with its character as two hex digits of ASCII (``61`` = ``a``).

**Where a letter sits is lost.** Segmentation cut each letter out of its row
and centred it in a 128x128 box, so a ``p`` sits as high as an ``o``: the
baseline, and whether a letter descends, are gone. Sizes are kept (a ``b`` is
taller than an ``o`` from the same writer), and that is what puts the line
back: letters sit on it, except that ``g p q y`` hang from the writer's own
x-height (the median height of their ``a c e m n o r s u v w x z``), so their
tails are as long as that writer made them. ``j`` and ``Q`` use where the
baseline falls in typefaces, as a fraction of ink height (``PLACEMENT``,
median over the 361 local corpus fonts), since a dot or a tail makes their
top a poor anchor. Then the letters go through the same framing as the
photographed sheets (``intake.normalize_samples``), so an SD19 writer lands
on the canvas the way a photographed writer does.

Advances (letter widths with spacing) are unknown and are written as NaN; the
loss skips them.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..charset import Charset, GlyphSpec

log = logging.getLogger(__name__)

#: Fields used: lowercase, uppercase, digits. ``c`` (the Constitution preamble)
#: has letters too, but many copies of a few and in running text.
FIELDS = "lud"

#: Letters whose top is at the x-height and whose tail hangs below the line.
HANGING = "gpqy"

#: Where the baseline falls, as a fraction of ink height from the top, for
#: letters whose top is not a good anchor: median over the 361 fonts in
#: data/cache_local. Also the fallback for HANGING letters when a writer has no
#: x-height letters. Everything else sits on the line (1.0).
PLACEMENT = {"g": 0.69, "p": 0.69, "q": 0.69, "y": 0.69, "j": 0.76, "Q": 0.83}

#: Letters whose height is the writer's x-height (no ascender, no descender).
X_HEIGHT = "acemnorsuvwxz"

_PNG = re.compile(r"by_write/(hsf_\d+)/f(\d{4}_\d{2})/([lud])\d{4}_\d{2}/[lud]\d{4}_\d{2}_(\d{5})\.png$")


def parse_cls(text: str) -> list[str]:
    """Characters of one ``.cls`` file, in image order."""
    lines = text.split()
    if not lines:
        return []
    count = int(lines[0])
    chars = [chr(int(code, 16)) for code in lines[1:]]
    if len(chars) != count:
        raise ValueError(f".cls declares {count} images but lists {len(chars)}")
    return chars


@dataclass
class Writer:
    """One SD19 form: who wrote it, and which PNG holds which character."""

    writer_id: str                     # e.g. "hsf_2/f1454_12"
    members: dict[str, list[str]] = field(default_factory=dict)  # char -> PNG names

    def chars(self) -> set[str]:
        return set(self.members)


def index_writers(by_write: zipfile.ZipFile, labels: dict[str, str]) -> list[Writer]:
    """Pair every by_write PNG with its label.

    ``labels`` maps a ``.cls`` path relative to by_write (``hsf_2/f1454_12/l1454_12.cls``)
    to its text. A field whose PNG count differs from its label count is
    dropped whole: an off-by-one there would mislabel every letter after it.
    """
    pngs: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for name in by_write.namelist():
        m = _PNG.search(name)
        if m:
            hsf, form, kind, index = m.groups()
            pngs.setdefault((hsf, form, kind), []).append((int(index), name))

    writers: dict[str, Writer] = {}
    dropped = 0
    for (hsf, form, kind), files in sorted(pngs.items()):
        if kind not in FIELDS:
            continue
        text = labels.get(f"{hsf}/f{form}/{kind}{form}.cls")
        if text is None:
            dropped += 1
            continue
        chars = parse_cls(text)
        files.sort()
        if len(chars) != len(files) or [i for i, _ in files] != list(range(len(files))):
            dropped += 1
            continue
        writer = writers.setdefault(f"{hsf}/f{form}", Writer(f"{hsf}/f{form}"))
        for ch, (_, name) in zip(chars, files):
            writer.members.setdefault(ch, []).append(name)
    if dropped:
        log.warning("dropped %d fields whose labels did not line up", dropped)
    return [writers[k] for k in sorted(writers)]


def _ink(png: bytes) -> np.ndarray:
    from PIL import Image

    gray = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32) / 255.0
    return 1.0 - gray  # SD19 is black ink on white


def on_a_line(inks: dict[str, np.ndarray], pad: int = 2) -> dict[str, np.ndarray]:
    """Re-seat centred letters on one shared baseline, as cells of one page.

    Returns paper-white images (``1 - ink``), all the same size, in which every
    letter's baseline is the same row: the "cells cut from the same template"
    that ``intake.normalize_samples`` expects.
    """
    boxes = {}
    for ch, ink in inks.items():
        rows = np.nonzero((ink > 0.5).any(axis=1))[0]
        cols = np.nonzero((ink > 0.5).any(axis=0))[0]
        if rows.size and cols.size:
            boxes[ch] = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
    if not boxes:
        return {}
    tallest = max(b - t for t, b, _, _ in boxes.values())
    widest = max(r - l for _, _, l, r in boxes.values())
    height, width = 2 * tallest + 2 * pad, widest + 2 * pad
    line = tallest + pad
    small = [b - t for c, (t, b, _, _) in boxes.items() if c in X_HEIGHT]
    x_height = float(np.median(small)) if small else None
    out = {}
    for ch, (top, bottom, left, right) in boxes.items():
        h = bottom - top
        if ch in HANGING and x_height is not None:
            # Written without a real tail (no taller than an x-height letter):
            # leave it on the line rather than hang a body below it.
            above = int(round(min(x_height, h) if h > 1.15 * x_height else h))
        else:
            above = int(round(PLACEMENT.get(ch, 1.0) * h))
        cell = np.zeros((height, width), np.float32)
        y0, x0 = line - above, pad + (widest - (right - left)) // 2
        cell[y0 : y0 + h, x0 : x0 + right - left] = inks[ch][top:bottom, left:right]
        out[ch] = 1.0 - cell
    return out


def writer_glyphs(by_write: zipfile.ZipFile, writer: Writer, charset: Charset,
                  size: int = 128) -> dict[str, np.ndarray]:
    """One writer's letters on the model's canvas, keyed by glyph key.

    A character written more than once (a few forms repeat one) uses the first.
    """
    from ..intake import IntakeConfig, normalize_samples

    spec_by_char: dict[str, GlyphSpec] = {g.char: g for g in charset}
    inks = {ch: _ink(by_write.read(names[0]))
            for ch, names in writer.members.items() if ch in spec_by_char}
    cells = on_a_line(inks)
    raw = {spec_by_char[ch]: image for ch, image in cells.items()}
    if not raw:
        return {}
    return normalize_samples(
        raw, IntakeConfig(size=size, ink_is_measured=True, normalize_pen=False)
    ).images


def iter_writer_fonts(by_write_zip: str | Path, labels: dict[str, str], charset: Charset,
                      size: int = 128, min_letters: int = 40) -> Iterator[tuple[Writer, dict[str, np.ndarray]]]:
    """Every writer with at least ``min_letters`` usable glyphs, converted."""
    with zipfile.ZipFile(by_write_zip) as z:
        for writer in index_writers(z, labels):
            if len(writer.members) < min_letters:
                continue
            glyphs = writer_glyphs(z, writer, charset, size)
            if len(glyphs) >= min_letters:
                yield writer, glyphs


#: Category of SD19 writers in the cache; ``DatasetConfig.sd19_weight`` keys on it.
CATEGORY = "SD19"


def _split(writer_id: str, held_out: float) -> str:
    """Stable train/val/test by writer, from a hash of the writer id."""
    u = int(hashlib.blake2b(writer_id.encode(), digest_size=4).hexdigest(), 16) / 2**32
    return "val" if u < held_out else "test" if u < 2 * held_out else "train"


def add_to_cache(cache_dir: str | Path, writers: Iterator[tuple[Writer, dict[str, np.ndarray]]],
                 held_out: float = 0.03) -> int:
    """Write SD19 writers into an existing glyph cache as fonts; returns how many.

    Each writer becomes one ``fonts/sd19__<id>.npz`` in the cache's format, and
    a record of category ``SD19`` in ``meta.json``. SD19 records already there
    are replaced, so this can be re-run. Advances are NaN (unknown). Re-pack
    the cache afterwards (``prepare.pack_cache``) if it is packed.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    order = {k: i for i, k in enumerate(meta["glyph_keys"])}
    records = [r for r in meta["fonts"] if r.get("category") != CATEGORY]
    (cache_dir / "fonts").mkdir(parents=True, exist_ok=True)
    added = 0
    for writer, glyphs in writers:
        keys = sorted((k for k in glyphs if k in order), key=order.get)
        images = np.stack([np.round(np.clip(glyphs[k], 0, 1) * 255).astype(np.uint8) for k in keys])
        font_id = f"sd19/{writer.writer_id}"
        file = font_id.replace("/", "__") + ".npz"
        np.savez_compressed(
            cache_dir / "fonts" / file,
            images=images,
            keys=np.asarray(keys),
            advances=np.full(len(keys), np.nan, np.float32),
            blank=np.zeros(len(keys), bool),
            hashes=np.asarray([hashlib.blake2b(im.tobytes(), digest_size=8).hexdigest()
                               for im in images]),
        )
        records.append({
            "font_id": font_id, "file": file, "family": f"SD19 {writer.writer_id}",
            "style": "Regular", "category": CATEGORY, "split": _split(writer.writer_id, held_out),
            "n_glyphs": len(keys), "scale": 1.0, "baseline_y": 0.0, "units_per_em": 0,
        })
        added += 1
    meta["fonts"] = sorted(records, key=lambda r: r["font_id"])
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return added
