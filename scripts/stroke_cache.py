"""Add pen-stroke sequences to a packed cache (step A3).

Every glyph image is reduced to its pen path and traced into strokes in one
fixed pen order (``hfont.strokes``); the strokes are resampled every
``--step`` pixels into ``(dx, dy, lift)`` rows. The rows of all glyphs are
stored end to end in ``strokes.npz`` next to the packed images:

    points   (total, 3) float16   dx, dy in pixels from the previous point
                                  (the first from the image's top-left
                                  corner), lift 1 on a stroke's last point
    offsets  (glyphs + 1,) int64  glyph i is points[offsets[i]:offsets[i+1]]
    step     the resampling step, pixels

A glyph with no pen path gets no rows. The stroke model trains on these.

    python scripts/stroke_cache.py --cache data/cache_sd19 --workers 12
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.data.prepare import PACKED_IMAGES  # noqa: E402
from hfont.strokes import glyph_sequence  # noqa: E402

STROKES = "strokes.npz"
log = logging.getLogger("stroke_cache")
_SRC = None


def _open(path: str) -> None:
    global _SRC
    _SRC = np.load(path, mmap_mode="r")


def _trace(job: tuple[int, int, float]) -> tuple[int, list[np.ndarray]]:
    start, stop, step = job
    out = []
    for i in range(start, stop):
        seq = glyph_sequence(_SRC[i].astype(np.float32) / 255.0, step)
        out.append(np.zeros((0, 3), np.float32) if seq is None else seq)
    return start, out


def convert(cache: str | Path, step: float = 3.0, workers: int = 1, chunk: int = 256,
            limit: int | None = None) -> Path:
    """Trace every glyph of the packed cache at ``cache``; returns the strokes file."""
    cache = Path(cache)
    images = np.load(cache / PACKED_IMAGES, mmap_mode="r")
    n = len(images) if limit is None else min(limit, len(images))
    jobs = [(i, min(i + chunk, n), step) for i in range(0, n, chunk)]
    seqs: list[np.ndarray | None] = [None] * n
    t0, done = time.time(), 0
    if workers > 1:
        with Pool(workers, initializer=_open, initargs=(str(cache / PACKED_IMAGES),)) as pool:
            for start, block in pool.imap_unordered(_trace, jobs):
                seqs[start:start + len(block)] = block
                done += len(block)
                if done % (chunk * 100) < chunk:
                    log.info("  %d/%d glyphs (%.0fs)", done, n, time.time() - t0)
    else:
        _open(str(cache / PACKED_IMAGES))
        for job in jobs:
            start, block = _trace(job)
            seqs[start:start + len(block)] = block
    lengths = np.array([len(s) for s in seqs], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    points = np.concatenate(seqs).astype(np.float16) if offsets[-1] else np.zeros((0, 3), np.float16)
    path = cache / STROKES
    np.savez(path, points=points, offsets=offsets, step=np.float32(step))
    log.info("%d glyphs, %d with strokes, median %d points, max %d", n, int((lengths > 0).sum()),
             int(np.median(lengths)), int(lengths.max()))
    return path


def load(cache: str | Path):
    """(points, offsets, step) of a cache's strokes file, the points memory-mapped."""
    with np.load(Path(cache) / STROKES) as data:
        return data["points"], data["offsets"], float(data["step"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--step", type=float, default=3.0, help="resampling step, pixels")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N glyphs only (a quick look)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    path = convert(args.cache, args.step, args.workers, limit=args.limit)
    log.info("-> %s (%.0fs)", path, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
