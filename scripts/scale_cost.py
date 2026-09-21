"""What does arriving small on the canvas cost a writer?

Real hands land at 15-25px x-height because framing fits their whole
ascender-to-descender range onto the canvas, where typeset fonts land at 39-55
(README, "Real hands are outside the corpus's proportions"). This measures the
price of that directly, without retraining: take held-out handwriting-style
fonts, rescale each font's glyphs about the baseline so its x-height lands at a
given size, and score leave-one-out at each size. The drop from native size to
~25px is roughly what scale is costing writer 1.

Rescaling is done on the normalized canvas, about the fixed baseline row, which
is exactly the transform intake applies when a hand's proportions force a
smaller scale: the letters shrink, their strokes with them, and the baseline
stays put.

    python scripts/scale_cost.py --sizes 45,35,30,25,20
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.evaluate.report import leave_one_out  # noqa: E402
from hfont.generate import content_images_from_font, load_checkpoint  # noqa: E402

#: Handwriting-style faces that ship with Windows and are not in Google Fonts,
#: so the model has not seen them. The three at the front are the stand-ins
#: scripts/handwriting_e2e.py already uses.
CANDIDATES = [
    "segoepr", "segoesc", "Inkfree", "BRADHITC", "Gabriola", "MISTRAL",
    "PRISTINA", "FREESCPT", "JUICE___", "RAGE",
]
X_HEIGHT_CHARS = "aemnors"


def x_height(images: dict[str, np.ndarray]) -> float:
    heights = []
    for key, image in images.items():
        if chr(int(key.split(".")[0], 16)) not in X_HEIGHT_CHARS:
            continue
        rows = np.nonzero((image > 0.5).any(axis=1))[0]
        if rows.size:
            heights.append(rows[-1] - rows[0] + 1)
    return float(np.median(heights)) if heights else 0.0


def rescale(image: np.ndarray, factor: float, baseline: float) -> np.ndarray:
    """Shrink a glyph about the baseline row and the canvas centre."""
    from scipy.ndimage import affine_transform

    size = image.shape[0]
    centre = size / 2.0
    matrix = np.array([[1 / factor, 0.0], [0.0, 1 / factor]])
    offset = (baseline - baseline / factor, centre - centre / factor)
    return affine_transform(image, matrix, offset=offset, order=1, mode="constant", cval=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--seed", default="seed30")
    parser.add_argument("--sizes", default="45,35,30,25,20")
    parser.add_argument("--fonts-dir", default="C:/Windows/Fonts")
    args = parser.parse_args()

    sizes = [float(s) for s in args.sizes.split(",")]
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)
    seed_keys = [g.key for g in LATIN_CORE if g.char in LATIN_SEED_SETS.get(args.seed, args.seed)]
    baseline = RenderConfig().size * RenderConfig().baseline

    table: dict[str, dict[str, float]] = {}
    for name in CANDIDATES:
        matches = glob.glob(f"{args.fonts_dir}/{name}.ttf")
        if not matches:
            continue
        try:
            rendered, _ = FontRenderer(matches[0], RenderConfig()).render(LATIN_CORE)
        except Exception as exc:
            print(f"  {name}: could not render ({exc})")
            continue
        if not all(k in rendered for k in seed_keys):
            continue
        refs = {k: rendered[k].image for k in seed_keys}
        native = x_height(refs)
        if native <= 0:
            continue

        scores = {"native": leave_one_out(model, refs, content)["tol_f1"]}
        for target in sizes:
            factor = target / native
            if factor >= 1.0:
                continue
            small = {k: rescale(v, factor, baseline) for k, v in refs.items()}
            scores[f"{target:.0f}px"] = leave_one_out(model, small, content)["tol_f1"]
        table[name] = {"x_height": native, **scores}
        print(f"  {name:10s} native x-height {native:4.1f}px  " +
              "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))

    if not table:
        print("no usable fonts found")
        return 1

    print(f"\n{'size':10s} {'mean tol-F1':>12s} {'drop from native':>18s}")
    native_mean = float(np.mean([t["native"] for t in table.values()]))
    print(f"{'native':10s} {native_mean:12.3f} {'-':>18s}")
    for target in sizes:
        key = f"{target:.0f}px"
        values = [t[key] for t in table.values() if key in t]
        if values:
            mean = float(np.mean(values))
            print(f"{key:10s} {mean:12.3f} {mean - native_mean:+18.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
