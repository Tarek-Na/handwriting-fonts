"""Which property of a hand actually predicts how well the model copies it?

Round 2 killed the slant hypothesis: measured from the stems rather than from
the ink's principal axis, slant does not correlate with leave-one-out at all.
This asks the same question of every other cheap property of a glyph set —
stroke weight absolute and relative, x-height, how much ink a letter carries,
how many pieces it is written in, how much outline detail it has — so that the
next hypothesis is chosen from evidence rather than from a hunch.

Reads the leave-one-out scores measured by scripts/slant_causal.py --measure,
so run that first.

    python scripts/what_predicts_score.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.intake import load_freehand_photo  # noqa: E402
from hfont.vector.trace import TraceConfig, trace_image  # noqa: E402
from slant import font_slant, load_font  # noqa: E402

X_HEIGHT_CHARS = "aemnors"


def features(images: dict[str, np.ndarray]) -> dict[str, float]:
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label, regionprops

    widths, inks, pieces, segments, heights = [], [], [], [], []
    for key, image in images.items():
        mask = image > 0.5
        if mask.sum() < 20:
            continue
        widths.append(float(np.mean(distance_transform_edt(mask)[mask])))
        inks.append(float(mask.sum()))
        pieces.append(sum(1 for p in regionprops(label(mask)) if p.area >= 6))
        try:
            segments.append(sum(len(p.segments) for p in trace_image(image, TraceConfig())))
        except Exception:
            pass
        rows = np.nonzero(mask.any(axis=1))[0]
        if chr(int(key.split(".")[0], 16)) in X_HEIGHT_CHARS:
            heights.append(rows[-1] - rows[0] + 1)

    x_height = float(np.median(heights)) if heights else float("nan")
    stroke = float(np.mean(widths))
    return {
        "stroke_px": stroke,
        "stroke_per_x_height": stroke / x_height if x_height else float("nan"),
        "x_height": x_height,
        "ink_per_glyph": float(np.mean(inks)),
        "pieces_per_glyph": float(np.mean(pieces)),
        "segments_per_glyph": float(np.mean(segments)) if segments else float("nan"),
        "slant": font_slant(images),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "output/myhand_diag/round2/slant_causal.json"))
    parser.add_argument("--photo", default=str(ROOT / "MyHandwriting.jpeg"))
    args = parser.parse_args()

    measured = json.loads(Path(args.results).read_text(encoding="utf-8"))["measure"]["fonts"]
    rows = []
    for name, loo, _ in measured:
        images = load_font(name)
        if images is None:
            continue
        rows.append((name, loo, features(images)))

    names = list(rows[0][2])
    print(f"{'font':10s} {'LOO':>6s} " + " ".join(f"{n[:9]:>10s}" for n in names))
    for name, loo, f in rows:
        print(f"{name:10s} {loo:6.3f} " + " ".join(f"{f[n]:10.3f}" for n in names))

    mine = features(load_freehand_photo(args.photo, LATIN_CORE, "seed30", 128))
    print(f"{'writer1':10s} {0.403:6.3f} " + " ".join(f"{mine[n]:10.3f}" for n in names))

    loo = np.array([r[1] for r in rows])
    print(f"\ncorrelation with leave-one-out across {len(rows)} held-out fonts:")
    for n in names:
        values = np.array([r[2][n] for r in rows])
        good = ~np.isnan(values)
        if good.sum() > 2:
            r = float(np.corrcoef(loo[good], values[good])[0, 1])
            flag = "  <-- strong" if abs(r) >= 0.6 else ""
            print(f"  {n:22s} r = {r:+.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
