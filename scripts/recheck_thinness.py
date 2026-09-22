"""Was round 2's thinness conclusion itself a metric artifact?

Round 2 concluded that stroke thinness is what costs writer 1, on the evidence
that dilating references and targets lifted leave-one-out from 0.403 to 0.593.
That experiment was scored with tolerant F1 — which scripts/metric_sensitivity.py
has since shown is blind to a 1px shift and unmoved by dilating a correct glyph.
Dilating the *target* widens the band a wrong stroke can land in, so the lift
may be the metric relaxing rather than the model improving.

This re-scores the same interventions with clDice, which is weight-blind, and
with IoU, which punishes weight. The reading:

* if clDice rises with dilation too, thinness is genuinely causal;
* if clDice is flat while tolerant F1 climbs, round 2 was measuring the ruler.

The second intervention — dilate the references, generate, erode the output
back — is scored against unmodified targets, so it cannot be inflated this way
and acts as the control.

    python scripts/recheck_thinness.py --out output/review
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
from hfont.evaluate.metrics import cl_dice, ink_coverage_ratio, iou, tolerant_f1  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import load_freehand_photo  # noqa: E402
from slant import load_font  # noqa: E402

METRICS = {"tol_f1": tolerant_f1, "iou": iou, "cl_dice": cl_dice,
           "coverage": ink_coverage_ratio}


def score_loo(model, content, refs: dict[str, np.ndarray],
              targets: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """Leave-one-out, scored with every metric at once.

    ``targets`` defaults to ``refs``; passing the untouched originals is what
    makes the second intervention immune to the metric relaxing.
    """
    targets = refs if targets is None else targets
    scores = {name: [] for name in METRICS}
    for key in refs:
        others = {k: v for k, v in refs.items() if k != key}
        made = generate_rasters(model, others, {key: content[key]})[0][key]
        for name, fn in METRICS.items():
            scores[name].append(fn(made, targets[key]))
    return {name: float(np.mean(v)) for name, v in scores.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=str(ROOT / "MyHandwriting.jpeg"))
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--out", default=str(ROOT / "output/review"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from skimage.morphology import dilation, disk, erosion

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)
    subjects = {"writer1": load_freehand_photo(args.photo, LATIN_CORE, "seed30", 128)}
    for name in ("BRADHITC",):
        images = load_font(name)
        if images is not None:
            subjects[name] = images

    results: dict[str, object] = {}

    print("== round 2's experiment, re-scored: dilate references AND targets\n")
    print(f"{'sample':10s} {'dilation':>9s} {'tol-F1':>8s} {'IoU':>8s} {'clDice':>8s} "
          f"{'coverage':>9s}")
    both = {}
    for name, refs in subjects.items():
        rows = []
        for radius in (0, 1, 2):
            fat = ({k: dilation(v, disk(radius)) for k, v in refs.items()} if radius
                   else dict(refs))
            row = score_loo(model, content, fat)
            row["radius"] = radius
            rows.append(row)
            print(f"{name:10s} {radius:9d} {row['tol_f1']:8.3f} {row['iou']:8.3f} "
                  f"{row['cl_dice']:8.3f} {row['coverage']:9.2f}")
        both[name] = rows
    results["dilate_both"] = both

    print("\n== the control: dilate references only, thin the output back, "
          "score against untouched letters\n")
    print(f"{'sample':10s} {'variant':>16s} {'tol-F1':>8s} {'IoU':>8s} {'clDice':>8s} "
          f"{'coverage':>9s}")
    trick = {}
    for name, refs in subjects.items():
        plain = score_loo(model, content, refs)
        print(f"{name:10s} {'as-is':>16s} {plain['tol_f1']:8.3f} {plain['iou']:8.3f} "
              f"{plain['cl_dice']:8.3f} {plain['coverage']:9.2f}")

        scores = {m: [] for m in METRICS}
        for key in refs:
            others = {k: dilation(v, disk(1)) for k, v in refs.items() if k != key}
            made = generate_rasters(model, others, {key: content[key]})[0][key]
            thinned = erosion(made, disk(1))
            for m, fn in METRICS.items():
                scores[m].append(fn(thinned, refs[key]))
        fixed = {m: float(np.mean(v)) for m, v in scores.items()}
        print(f"{name:10s} {'fat refs, thinned':>16s} {fixed['tol_f1']:8.3f} "
              f"{fixed['iou']:8.3f} {fixed['cl_dice']:8.3f} {fixed['coverage']:9.2f}")
        trick[name] = {"as_is": plain, "thickened": fixed}
    results["thickness_trick"] = trick

    (out / "recheck_thinness.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n== verdict")
    for name, rows in both.items():
        d = {m: rows[-1][m] - rows[0][m] for m in METRICS}
        print(f"{name}: dilating both by 2px moves tol-F1 {d['tol_f1']:+.3f}, "
              f"IoU {d['iou']:+.3f}, clDice {d['cl_dice']:+.3f}")
    for name, pair in trick.items():
        d = {m: pair["thickened"][m] - pair["as_is"][m] for m in METRICS}
        print(f"{name}: the control moves tol-F1 {d['tol_f1']:+.3f}, "
              f"IoU {d['iou']:+.3f}, clDice {d['cl_dice']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
