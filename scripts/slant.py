"""Slant measured from the stems themselves, and the shear that undoes it.

The first slant metric took the principal axis of a glyph's ink, which is
really a proportion measure: a tall narrow letter reads as upright however it
leans, and a wide one reads as slanted. This measures the thing directly —
the dominant direction of near-vertical stroke edges, from a magnitude-weighted
orientation histogram, folded to an undirected line and averaged circularly.

`--selfcheck` shears a font by known angles and reports what the metric gets
back, which is what makes the number trustworthy before it is used to judge
anything.

    python scripts/slant.py --selfcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402

#: Stroke directions this far from vertical are stems; the rest are the
#: horizontals and curves, which carry no slant information.
STEM_WINDOW_DEG = 40.0


def stem_slant(image: np.ndarray, window: float = STEM_WINDOW_DEG) -> float:
    """Degrees the near-vertical strokes lean, positive = leaning right."""
    from scipy.ndimage import gaussian_filter, sobel

    smooth = gaussian_filter(image.astype(np.float32), 1.0)
    gy, gx = sobel(smooth, axis=0), sobel(smooth, axis=1)
    magnitude = np.hypot(gx, gy)
    if not np.any(magnitude > 0):
        return 0.0
    strong = magnitude > max(np.percentile(magnitude[magnitude > 0], 75), 1e-6)
    if not strong.any():
        return 0.0

    # The stroke runs along the edge, at right angles to the gradient. Fold to
    # [0, 180): a stroke has a direction but no arrow.
    along = np.degrees(np.arctan2(gy[strong], gx[strong])) + 90.0
    along = np.mod(along, 180.0)
    weight = magnitude[strong]

    near_vertical = np.abs(along - 90.0) <= window
    if near_vertical.sum() < 10:
        return 0.0
    angles, weights = along[near_vertical], weight[near_vertical]

    # The peak of the histogram, not its mean. A shear turns a vertical line by
    # exactly its own angle but a line already off vertical by less, so
    # averaging across the band reads a 10 deg shear as 7. The peak is the
    # stems themselves, and it tracks the shear one for one.
    bins = np.arange(90.0 - window, 90.0 + window + 1.0, 1.0)
    counts, _ = np.histogram(angles, bins=bins, weights=weights)
    if counts.sum() <= 0:
        return 0.0
    counts = np.convolve(counts, np.ones(5) / 5.0, mode="same")
    peak = int(np.argmax(counts))
    centre = 0.5 * (bins[peak] + bins[peak + 1])
    # Parabolic refinement between neighbouring bins.
    if 0 < peak < len(counts) - 1:
        a, b, c = counts[peak - 1], counts[peak], counts[peak + 1]
        denominator = a - 2 * b + c
        if abs(denominator) > 1e-9:
            centre += 0.5 * (a - c) / denominator
    return float(90.0 - centre)


def shear(image: np.ndarray, degrees: float, baseline: float | None = None) -> np.ndarray:
    """Shear horizontally about the baseline row. Positive leans right."""
    from scipy.ndimage import affine_transform

    if abs(degrees) < 1e-6:
        return image
    size = image.shape[0]
    baseline = size * RenderConfig().baseline if baseline is None else baseline
    k = np.tan(np.radians(degrees))
    # Inverse map: output (y, x) <- input (y, x - k * (baseline - y)), so that
    # a positive angle carries the top of the glyph to the right.
    matrix = np.array([[1.0, 0.0], [-k, 1.0]])
    offset = (0.0, k * baseline)
    return affine_transform(image, matrix, offset=offset, order=1, mode="constant", cval=0.0)


def font_slant(images: dict[str, np.ndarray]) -> float:
    """Ink-weighted mean stem slant over a set of glyphs."""
    angles, weights = [], []
    for image in images.values():
        ink = float((image > 0.5).sum())
        if ink < 20:
            continue
        angles.append(stem_slant(image))
        weights.append(ink)
    if not angles:
        return 0.0
    return float(np.average(angles, weights=weights))


def load_font(name: str, fonts_dir: str = "C:/Windows/Fonts") -> dict[str, np.ndarray] | None:
    keys = [g.key for g in LATIN_CORE if g.char in LATIN_SEED_SETS["seed30"]]
    for ext in (".ttf", ".TTF"):
        try:
            rendered, _ = FontRenderer(f"{fonts_dir}/{name}{ext}", RenderConfig()).render(LATIN_CORE)
        except Exception:
            continue
        if all(k in rendered for k in keys):
            return {k: rendered[k].image for k in keys}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--font", default="segoepr")
    args = parser.parse_args()

    if args.selfcheck:
        print("shearing a font by a known angle and measuring it back:\n")
        print(f"{'font':10s} {'applied':>8s} {'measured':>9s} {'error':>7s}")
        for name in ("segoepr", "Gabriola", "JUICE___", "times"):
            images = load_font(name)
            if images is None:
                continue
            base = font_slant(images)
            for applied in (0.0, 10.0, 20.0, -10.0):
                sheared = {k: shear(v, applied) for k, v in images.items()}
                got = font_slant(sheared) - base
                print(f"{name:10s} {applied:8.1f} {got:9.1f} {got - applied:+7.1f}")
            print(f"{'':10s} (unsheared reading for {name}: {base:+.1f} deg)")
        return 0

    images = load_font(args.font)
    if images is None:
        print(f"could not render {args.font}")
        return 1
    print(f"{args.font}: stem slant {font_slant(images):+.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
