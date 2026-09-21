"""What can each metric actually see at a real hand's stroke width?

Round 2 showed tolerant F1 rises when strokes are dilated. This asks the
sharper question: at the ~1.3px strokes real handwriting arrives with, how much
of each metric's range is left for anything at all? A metric whose tolerance is
wider than a stroke cannot tell a small misplacement from a perfect copy, and a
metric that rewards fat strokes cannot tell "right shape, too heavy" from
"wrong shape".

Each perturbation is applied to the prediction only, with the target fixed, so
the numbers are directly comparable: a shift by one pixel, a thickening, a
thinning, a broken stroke (topology changed, ink almost unchanged), and a
different letter entirely as the floor.

    python scripts/metric_sensitivity.py --out output/review
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
from hfont.evaluate.metrics import cl_dice, iou, ssim, tolerant_f1  # noqa: E402
from hfont.intake import load_freehand_photo  # noqa: E402
from slant import load_font  # noqa: E402


def break_stroke(image: np.ndarray, gap: int = 3) -> np.ndarray:
    """Erase a small band across the glyph: topology changes, ink barely does."""
    out = image.copy()
    rows = np.nonzero((image > 0.5).any(axis=1))[0]
    if rows.size < 4 * gap:
        return out
    middle = rows[len(rows) // 2]
    out[middle : middle + gap, :] = 0.0
    return out


def perturbations(image: np.ndarray, other: np.ndarray):
    from scipy.ndimage import shift
    from skimage.morphology import dilation, disk, erosion

    return {
        "identical": image,
        "shifted 1px": shift(image, (0, 1), order=1, cval=0.0),
        "shifted 2px": shift(image, (0, 2), order=1, cval=0.0),
        "shifted 3px": shift(image, (0, 3), order=1, cval=0.0),
        "dilated 1px": dilation(image, disk(1)),
        "dilated 2px": dilation(image, disk(2)),
        "eroded 1px": erosion(image, disk(1)),
        "broken stroke": break_stroke(image),
        "a different letter": other,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=str(ROOT / "MyHandwriting.jpeg"))
    parser.add_argument("--out", default=str(ROOT / "output/review"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = {"writer 1 (1.3px strokes)": load_freehand_photo(args.photo, LATIN_CORE,
                                                               "seed30", 128)}
    for name, label in (("segoepr", "Segoe Print (2.1px strokes)"),
                        ("times", "Times (2.6px strokes)")):
        images = load_font(name)
        if images is not None:
            samples[label] = images

    results = {}
    for label, images in samples.items():
        keys = list(images)
        print(f"\n== {label}")
        print(f"{'perturbation':22s} {'IoU':>7s} {'tol-F1':>7s} {'clDice':>7s} {'SSIM':>7s} "
              f"{'ink':>6s}")
        table = {}
        for name in ("identical", "shifted 1px", "shifted 2px", "shifted 3px",
                     "dilated 1px", "dilated 2px", "eroded 1px", "broken stroke",
                     "a different letter"):
            scores = {"iou": [], "tol_f1": [], "cl_dice": [], "ssim": [], "ink": []}
            for i, key in enumerate(keys):
                target = images[key]
                other = images[keys[(i + 7) % len(keys)]]
                pred = perturbations(target, other)[name]
                if (target > 0.5).sum() < 20:
                    continue
                scores["iou"].append(iou(pred, target))
                scores["tol_f1"].append(tolerant_f1(pred, target))
                scores["cl_dice"].append(cl_dice(pred, target))
                scores["ssim"].append(ssim(pred, target))
                scores["ink"].append((pred > 0.5).sum() / max((target > 0.5).sum(), 1))
            row = {k: float(np.mean(v)) for k, v in scores.items() if v}
            table[name] = row
            print(f"{name:22s} {row['iou']:7.3f} {row['tol_f1']:7.3f} {row['cl_dice']:7.3f} "
                  f"{row['ssim']:7.3f} {row['ink']:6.2f}")
        results[label] = table

    (out / "metric_sensitivity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n== usable range: identical minus a different letter")
    print(f"{'sample':28s} {'IoU':>7s} {'tol-F1':>7s} {'clDice':>7s}")
    for label, table in results.items():
        floor, top = table["a different letter"], table["identical"]
        print(f"{label:28s} {top['iou'] - floor['iou']:7.3f} "
              f"{top['tol_f1'] - floor['tol_f1']:7.3f} {top['cl_dice'] - floor['cl_dice']:7.3f}")

    print("\n== what a 1px shift costs, as a share of that range")
    for label, table in results.items():
        floor, top, one = (table["a different letter"], table["identical"],
                           table["shifted 1px"])
        share = lambda m: (top[m] - one[m]) / max(top[m] - floor[m], 1e-9)  # noqa: E731
        print(f"{label:28s} IoU {share('iou'):6.1%}  tol-F1 {share('tol_f1'):6.1%}  "
              f"clDice {share('cl_dice'):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
