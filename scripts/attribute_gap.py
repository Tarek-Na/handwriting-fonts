"""Three of the four candidate causes behind one writer's low score.

Companion to scripts/scale_cost.py, which measures the fourth (scale). For a
real photographed sample this answers:

* **intake quality** - did the letters arrive intact? Counts the connected
  components of each reference against what the character should have, reports
  strokes the intake dropped, and gives the residuals of each row's fitted
  writing line. Also reports the sample's resolution in photo pixels, which is
  what the writer can actually change.
* **style hedging** - are the generated letters heavier, smoother and more
  upright than the hand that wrote them (README known issues 1 and 3)? Slant
  comes from the second moments of the ink; regularity from the spread of
  stroke widths across a glyph.
* **seed coverage** - does leave-one-out improve when the model is given the
  30 letters of `seed30` rather than the 24 of `seed24`? Scored on the same 24
  glyphs either way, so the numbers are comparable.

    python scripts/attribute_gap.py --photo MyHandwriting.jpeg --out output/myhand_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.evaluate.report import leave_one_out  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import FreehandConfig, load_freehand_photo  # noqa: E402

#: How many separate marks each character is normally written with.
EXPECTED_PARTS = {"i": 2, "j": 2}


def parts_of(image: np.ndarray, level: float = 0.5, min_area: int = 6) -> int:
    from skimage.measure import label, regionprops

    return sum(1 for p in regionprops(label(image > level)) if p.area >= min_area)


def slant_degrees(image: np.ndarray) -> float:
    """Tilt of the ink's principal axis from vertical, in degrees."""
    ys, xs = np.nonzero(image > 0.5)
    if ys.size < 20:
        return 0.0
    ys = ys - ys.mean()
    xs = xs - xs.mean()
    mu20, mu02, mu11 = (xs * xs).mean(), (ys * ys).mean(), (xs * ys).mean()
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    return float(np.degrees(angle))


def stroke_spread(image: np.ndarray) -> float:
    """Coefficient of variation of stroke half-width: how even the pen is."""
    from scipy.ndimage import distance_transform_edt

    mask = image > 0.5
    if not mask.any():
        return 0.0
    widths = distance_transform_edt(mask)[mask]
    return float(np.std(widths) / max(np.mean(widths), 1e-6))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--rows", default="10,10,10")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FreehandConfig(rows=tuple(int(n) for n in args.rows.split(",")))

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)
    refs30 = load_freehand_photo(args.photo, LATIN_CORE, "seed30", 128, cfg)
    by_char = {chr(int(k.split(".")[0], 16)): k for k in refs30}
    findings: dict[str, object] = {}

    # ---- intake quality ---------------------------------------------------
    print("== intake quality")
    broken = []
    for char, key in by_char.items():
        parts = parts_of(refs30[key])
        if parts != EXPECTED_PARTS.get(char, 1):
            broken.append((char, parts, EXPECTED_PARTS.get(char, 1)))
    print(f"  glyphs whose piece count is not what the character needs: {len(broken)}/{len(refs30)}")
    for char, got, want in sorted(broken):
        print(f"    {char}: {got} piece(s), expected {want}")

    from PIL import Image
    from skimage.measure import label, regionprops

    from hfont.intake import IntakeConfig, _paper_only, _remove_rulings, to_ink_field

    photo = np.asarray(Image.open(args.photo).convert("RGB")).astype(np.float32) / 255.0
    ink = _remove_rulings(to_ink_field(_paper_only(photo[..., :3].min(axis=2)), IntakeConfig()), cfg)
    from scipy.ndimage import distance_transform_edt

    mask = ink > cfg.ink_level
    widths = distance_transform_edt(mask)[mask]
    letters = [p for p in regionprops(label(mask)) if p.area >= 30]
    heights = [p.bbox[2] - p.bbox[0] for p in letters]
    print(f"  photo: {photo.shape[1]}x{photo.shape[0]}px, pen half-width "
          f"{np.median(widths):.1f}px, median letter height {np.median(heights):.0f}px")
    print(f"  strokes per letter height: {np.median(widths) * 2 / np.median(heights):.3f} "
          f"(a letter is ~{np.median(heights) / max(np.median(widths) * 2, 1e-6):.0f} strokes tall)")
    findings["intake"] = {
        "broken_glyphs": broken,
        "photo_px": [int(photo.shape[1]), int(photo.shape[0])],
        "pen_half_width_px": float(np.median(widths)),
        "letter_height_px": float(np.median(heights)),
    }

    # ---- style hedging ----------------------------------------------------
    print("\n== style hedging (real vs generated, on the writer's own letters)")
    real_slant, gen_slant, real_even, gen_even, ratios = [], [], [], [], []
    for key, real in refs30.items():
        others = {k: v for k, v in refs30.items() if k != key}
        gen = generate_rasters(model, others, {key: content[key]})[0][key]
        real_slant.append(slant_degrees(real))
        gen_slant.append(slant_degrees(gen))
        real_even.append(stroke_spread(real))
        gen_even.append(stroke_spread(gen))
        ratios.append((gen > 0.5).sum() / max((real > 0.5).sum(), 1))
    print(f"  ink ratio            real 1.00  generated {np.median(ratios):.2f}")
    print(f"  slant, degrees       real {np.mean(np.abs(real_slant)):5.1f}  "
          f"generated {np.mean(np.abs(gen_slant)):5.1f}")
    print(f"  stroke-width spread  real {np.mean(real_even):.3f}  generated {np.mean(gen_even):.3f}"
          "   (lower = more even pen)")
    findings["hedging"] = {
        "ink_ratio_median": float(np.median(ratios)),
        "slant_real": float(np.mean(np.abs(real_slant))),
        "slant_generated": float(np.mean(np.abs(gen_slant))),
        "stroke_spread_real": float(np.mean(real_even)),
        "stroke_spread_generated": float(np.mean(gen_even)),
    }

    # ---- seed coverage ----------------------------------------------------
    print("\n== seed coverage")
    seed24 = LATIN_SEED_SETS["seed24"]
    refs24 = {by_char[c]: refs30[by_char[c]] for c in seed24 if c in by_char}
    scored = [by_char[c] for c in seed24 if c in by_char]

    def loo_on(reference_set: dict[str, np.ndarray], keys: list[str]) -> float:
        scores = []
        from hfont.evaluate.metrics import tolerant_f1

        for key in keys:
            others = {k: v for k, v in reference_set.items() if k != key}
            gen = generate_rasters(model, others, {key: content[key]})[0][key]
            scores.append(tolerant_f1(gen, refs30[key]))
        return float(np.mean(scores))

    with24 = loo_on(refs24, scored)
    with30 = loo_on(refs30, scored)
    print(f"  the same 24 glyphs, scored with 23 references: {with24:.3f}")
    print(f"  the same 24 glyphs, scored with 29 references: {with30:.3f}  "
          f"({with30 - with24:+.3f})")
    print(f"  all 30 glyphs, 29 references: {leave_one_out(model, refs30, content)['tol_f1']:.3f}")
    findings["seed_coverage"] = {"loo_24_refs": with24, "loo_29_refs": with30}

    (out / "attribution.json").write_text(json.dumps(findings, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
