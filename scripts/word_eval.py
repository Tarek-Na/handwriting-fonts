"""Whole words: does a font made from isolated letters set words like the writer's own?

    python scripts/word_eval.py --split val --method model:runs/v1_drop/best.pt --out runs/words_val.json
    HFONT_VARIATION=ends python scripts/word_eval.py --method model:runs/ab_ends_s0/best.pt ...

Letter-level scores (``variation.py score``) judge one form at a time. A
word adds what they cannot see: how wide the letters run and how the joins
carry the line. For each writer and test word:

* the reference is the word set in fonts made from the writer's own full
  sheet, one per even-numbered copy (as ``score`` uses even copies);
* the method's font is built from the writer's isolated letters (copy 0)
  as ``variation.sheet_font`` does, and sets the word;
* **width error**: |log(width / median real width)|, 0 when the word runs
  exactly as long as the writer's;
* **shape**: aligned centreline F1 after stretching the word across to the
  real word's width (best over the real copies), so shape is judged apart
  from length;
* **raw**: the same F1 without stretching.

Methods: ``model:PATH`` (several versions per form, the font cycling them),
``template`` (the template skeletons), ``own`` (fonts from the writer's
odd-numbered copies: the ceiling). Every rendering is of real handwriting
or made from it, and stays in memory; only numbers are written.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

WORDS = ["سلام", "عليكم", "كلب", "جميل", "قلم", "عسل", "ملك", "تفاحة", "شمس", "خبز", "ذهب", "ظل", "غرفة",
         "ثلاثة", "أرض"]
REAL_COPIES = (0, 2, 4, 6, 8)
OWN_COPIES = (1, 3, 5, 7, 9)


def word_ink(font_bytes: bytes, word: str) -> np.ndarray:
    from ahawp_font import render

    im = render(font_bytes, [word])
    ink = (im < 128) if im.mean() > 128 else (im > 128)
    rows, cols = np.nonzero(ink)
    return ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.float32)


def stretch(ink: np.ndarray, width: int) -> np.ndarray:
    """``ink`` resampled across to ``width`` columns, rows kept."""
    from PIL import Image

    im = Image.fromarray((255 * ink).astype(np.uint8)).resize((max(1, width), ink.shape[0]), Image.BILINEAR)
    return (np.asarray(im) > 127).astype(np.float32)


def compare(drawn: list[np.ndarray], real: list[np.ndarray]) -> dict:
    from hfont.evaluate.metrics import aligned_centreline_f1

    real_w = float(np.median([r.shape[1] for r in real]))
    width = float(np.mean([abs(np.log(d.shape[1] / real_w)) for d in drawn]))
    raw = float(np.mean([max(aligned_centreline_f1(d, r) for r in real) for d in drawn]))
    shape = float(np.mean([max(aligned_centreline_f1(stretch(d, r.shape[1]), r) for r in real) for d in drawn]))
    return {"width_err": width, "raw": raw, "shape": shape, "width_ratio": float(np.mean([d.shape[1] for d in drawn])) / real_w}


def _font_bytes(font) -> bytes:
    buffer = io.BytesIO()
    font.save(buffer)
    return buffer.getvalue()


def _writer(args) -> list[dict]:
    writer, method, versions = args
    import variation as v
    from ahawp_font import writer_font
    from hfont.data import ahawp

    archive = ahawp.open_archive(v.ARCHIVE)
    names = ahawp.index(archive)[writer]
    real_fonts = [_font_bytes(writer_font(archive, names, f"r{c}", copy=c)[0]) for c in REAL_COPIES]
    if method == "own":
        fonts = [_font_bytes(writer_font(archive, names, f"o{c}", copy=c)[0]) for c in OWN_COPIES]
    elif method == "template":
        fonts = [_font_bytes(v.sheet_font(writer, None, 1, 0.4)[0])]
    else:
        fonts = [_font_bytes(v.sheet_font(writer, method.split(":", 1)[1], versions, 0.4)[0])]
    rows = []
    for word in WORDS:
        real = [word_ink(f, word) for f in real_fonts]
        drawn = [word_ink(f, word) for f in fonts]
        rows.append({"writer": writer, "word": word, **compare(drawn, real)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--method", required=True, help="model:PATH, template or own")
    ap.add_argument("--versions", type=int, default=3)
    ap.add_argument("--label", default=None, help="key in --out (default: the method and HFONT_VARIATION)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import variation as v
    from hfont.data import ahawp

    writers = sorted(w for w in ahawp.index(ahawp.open_archive(v.ARCHIVE)) if ahawp.split_of(w) == args.split)
    writers = writers[:args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(_writer, [(w, args.method, args.versions) for w in writers]) for r in part]
    label = args.label or f"{args.method}|{v.VARIANT or 'original'}"
    m = {k: np.mean([r[k] for r in rows]) for k in ("width_err", "width_ratio", "shape", "raw")}
    print(f"{label:48} {args.split} {len(writers)} writers {len(rows)} words: width error {m['width_err']:.3f} "
          f"(width x{m['width_ratio']:.2f})  shape {m['shape']:.3f}  raw {m['raw']:.3f}  {time.time() - t0:.0f}s",
          flush=True)
    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    done[label] = rows
    args.out.write_text(json.dumps(done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
