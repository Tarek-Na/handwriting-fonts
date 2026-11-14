"""Redraw every glyph of a packed cache with one thin pen.

Fonts are bold or light, SD19 writers used thick markers, and photographed
handwriting is thin ballpoint. The model trained on the first two draws heavy
letters whatever pen the writer used, and reads a photographed sheet as
something it never saw (reports/Colab runs November 2026.md, third session).
Here every training glyph is reduced to its pen path
(``hfont.generate.centreline``) and drawn again at one half-width, so a
letter's weight carries no information and the three sources look alike.
Train on the result with ``--set data.pen_radius=<radius>``; inference then
redraws the writer's letters the same way before the model reads them, and
draws the output in the writer's own pen (``generate_rasters(pen=True)``).

Writes a packed cache only (``meta.json`` with ``"pen_radius"``, the packed
images and index); the per-font files are not copied.

    python scripts/pen_cache.py --src data/cache_sd19 --dst data/cache_pen
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.data.prepare import PACKED_IMAGES, PACKED_INDEX  # noqa: E402
from hfont.generate import restroke  # noqa: E402

log = logging.getLogger("pen_cache")
_SRC = None


def _open(path: str) -> None:
    global _SRC
    _SRC = np.load(path, mmap_mode="r")


def _redraw(job: tuple[int, int, float]) -> tuple[int, np.ndarray]:
    start, stop, radius = job
    out = np.empty((stop - start, *_SRC.shape[1:]), np.uint8)
    for i in range(start, stop):
        image = _SRC[i].astype(np.float32) / 255.0
        out[i - start] = np.round(restroke(image, radius) * 255.0).astype(np.uint8)
    return start, out


def convert(src: str | Path, dst: str | Path, radius: float, workers: int = 1,
            chunk: int = 512) -> int:
    """Redraw the packed cache at ``src`` into ``dst``; returns the glyph count."""
    src, dst = Path(src), Path(dst)
    if not (src / PACKED_IMAGES).is_file():
        raise FileNotFoundError(f"{src} is not packed (hfont.data.prepare.pack_cache)")
    dst.mkdir(parents=True, exist_ok=True)
    images = np.load(src / PACKED_IMAGES, mmap_mode="r")
    out = np.lib.format.open_memmap(dst / PACKED_IMAGES, mode="w+", dtype=np.uint8,
                                    shape=images.shape)
    jobs = [(i, min(i + chunk, len(images)), radius) for i in range(0, len(images), chunk)]
    t0, done = time.time(), 0
    if workers > 1:
        with Pool(workers, initializer=_open, initargs=(str(src / PACKED_IMAGES),)) as pool:
            results = pool.imap_unordered(_redraw, jobs)
            for start, block in results:
                out[start:start + len(block)] = block
                done += len(block)
                if done % (chunk * 40) < chunk:
                    log.info("  %d/%d glyphs (%.0fs)", done, len(images), time.time() - t0)
    else:
        _open(str(src / PACKED_IMAGES))
        for job in jobs:
            start, block = _redraw(job)
            out[start:start + len(block)] = block
    out.flush()
    shutil.copy(src / PACKED_INDEX, dst / PACKED_INDEX)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    meta["pen_radius"] = radius
    (dst / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return len(images)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--radius", type=float, default=2.5, help="pen half-width in pixels")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    n = convert(args.src, args.dst, args.radius, args.workers)
    log.info("%d glyphs redrawn at radius %.2f into %s (%.0fs)", n, args.radius, args.dst,
             time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
