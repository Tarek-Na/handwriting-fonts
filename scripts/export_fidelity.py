"""How much does the export path cost, and does it cost thin hands more?

`hfont.cli roundtrip` answers this for one font at its native weight and
reports ~0.98 IoU, which is where the claim "the pipeline is not the problem"
comes from. But real handwriting arrives with ~1.3px strokes, four times
thinner than the Times stems that number was measured on, and a raster is
turned into Beziers at a fixed 0.5 level before being rasterized again.

This runs the same raster -> outline -> OTF -> raster round trip across a range
of stroke widths, with the model taken out of the loop, so that whatever it
costs is a ceiling on what any model can deliver through this pipeline. A
writer's own photographed letters are included, since they are the real input
distribution rather than an approximation of it.

    python scripts/export_fidelity.py --out output/review
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfont.charset import LATIN_CORE, Charset  # noqa: E402
from hfont.data.render import FontFrame, FontRenderer, RenderConfig  # noqa: E402
from hfont.evaluate.metrics import iou, tolerant_f1  # noqa: E402
from hfont.fontbuild.build import (  # noqa: E402
    BuildConfig,
    FontMetadata,
    rasters_to_font,
    save_font,
)
from hfont.intake import load_freehand_photo  # noqa: E402
from slant import load_font  # noqa: E402


def stroke_width(images: dict[str, np.ndarray]) -> float:
    from scipy.ndimage import distance_transform_edt

    widths = []
    for image in images.values():
        mask = image > 0.5
        if mask.sum() >= 20:
            widths.append(float(np.mean(distance_transform_edt(mask)[mask])))
    return float(np.mean(widths)) if widths else 0.0


def round_trip(images: dict[str, np.ndarray], advances: dict[str, float],
               charset: Charset, tmp: Path, name: str) -> dict[str, float]:
    """Build a font from rasters and render it back in the same frame."""
    rc = RenderConfig()
    cfg = BuildConfig(image_size=rc.size, baseline=rc.baseline, margin=rc.margin,
                      ascender_em=rc.ascender_em)
    builder = rasters_to_font(images, advances, charset, FontMetadata(family=name), cfg)
    path = save_font(builder, tmp / f"{name}.otf")
    frame = FontFrame(scale=rc.pixels_per_em / 1000.0, baseline_y=rc.baseline_px,
                      units_per_em=1000)
    rebuilt, _ = FontRenderer(path, rc).render(charset, frame=frame)

    pairs = [(images[k], rebuilt[k].image) for k in images if k in rebuilt]
    if not pairs:
        return {"n": 0}
    return {
        "n": len(pairs),
        "iou": float(np.mean([iou(b, a) for a, b in pairs])),
        "tol_f1": float(np.mean([tolerant_f1(b, a) for a, b in pairs])),
        "ink_ratio": float(np.mean([(b > 0.5).sum() / max((a > 0.5).sum(), 1)
                                    for a, b in pairs])),
        "stroke_px": stroke_width(images),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=str(ROOT / "MyHandwriting.jpeg"))
    parser.add_argument("--out", default=str(ROOT / "output/review"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    from skimage.morphology import dilation, erosion, disk

    results = []

    print(f"{'sample':28s} {'stroke':>7s} {'IoU':>7s} {'tol-F1':>7s} {'ink':>6s}")

    # A real hand, as the model actually receives it.
    mine = load_freehand_photo(args.photo, LATIN_CORE, "seed30", 128)
    mine_full = {k: mine[k] for k in mine}
    advances = {k: 0.5 for k in mine_full}
    row = round_trip(mine_full, advances, Charset("seed", [g for g in LATIN_CORE
                                                          if g.key in mine_full]), tmp, "writer1")
    row["sample"] = "writer1 (photographed)"
    results.append(row)
    print(f"{row['sample']:28s} {row['stroke_px']:7.2f} {row['iou']:7.3f} "
          f"{row['tol_f1']:7.3f} {row['ink_ratio']:6.2f}")

    # Fonts at native weight, then thinned and thickened around it.
    for name in ("segoepr", "BRADHITC", "times"):
        images = load_font(name)
        if images is None:
            continue
        keys = list(images)
        charset = Charset("probe", [g for g in LATIN_CORE if g.key in images])
        for label, transform in (
            ("eroded 1px", lambda v: erosion(v, disk(1))),
            ("native", lambda v: v),
            ("dilated 1px", lambda v: dilation(v, disk(1))),
            ("dilated 2px", lambda v: dilation(v, disk(2))),
        ):
            shaped = {k: transform(images[k]) for k in keys}
            if stroke_width(shaped) <= 0:
                continue
            row = round_trip(shaped, {k: 0.5 for k in keys}, charset, tmp,
                             f"{name}_{label.replace(' ', '')}")
            row["sample"] = f"{name} {label}"
            results.append(row)
            print(f"{row['sample']:28s} {row['stroke_px']:7.2f} {row['iou']:7.3f} "
                  f"{row['tol_f1']:7.3f} {row['ink_ratio']:6.2f}")

    (out / "export_fidelity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    thin = [r for r in results if r.get("stroke_px", 0) < 1.6 and r.get("n")]
    thick = [r for r in results if r.get("stroke_px", 0) >= 2.2 and r.get("n")]
    if thin and thick:
        print(f"\nstrokes under 1.6px: IoU {np.mean([r['iou'] for r in thin]):.3f}, "
              f"tol-F1 {np.mean([r['tol_f1'] for r in thin]):.3f}  (n={len(thin)})")
        print(f"strokes over 2.2px:  IoU {np.mean([r['iou'] for r in thick]):.3f}, "
              f"tol-F1 {np.mean([r['tol_f1'] for r in thick]):.3f}  (n={len(thick)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
