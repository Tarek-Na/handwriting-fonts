"""Stress-test the freehand intake path with a synthetic photographed sheet.

The template version of this test (``simulate_template_photo.py``) has printed
markers and a grid to lean on. Writing on blank paper has neither: rows, the
writing line and the letter boundaries all have to be recovered from the
writing. This renders a font as if someone had written the seed letters out in
rows — with a sloping line, uneven spacing and a pen much thicker than any
printed stem — photographs it badly, and measures what comes back.

The number that matters is framing agreement: intake-normalized glyphs against
the same glyphs framed the way the training corpus was. The model only ever saw
the second framing, so any gap here is a domain gap the pipeline itself
introduces, before handwriting style enters into it.

Usage::

    python scripts/simulate_freehand_photo.py C:/Windows/Fonts/Inkfree.ttf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS, GlyphSpec  # noqa: E402
from hfont.data.raster import FlatteningPen, fill_contours  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.evaluate.metrics import tolerant_f1  # noqa: E402
from hfont.intake import FreehandConfig, freehand_cells, IntakeConfig, normalize_samples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_template_photo import degrade, iou  # noqa: E402

INK_RGB = np.array([170, 40, 55], dtype=np.float64)  # red gel pen, as written


def write_sheet(
    font_path: str, seed: str = "seed30", rows: tuple[int, ...] = (10, 10, 10),
    cap_px: int = 70, pen: float = 1.0, slope: float = 0.02, seed_rng: int = 0,
) -> np.ndarray:
    """Render the seed letters onto a blank sheet, as if written by hand."""
    from scipy.ndimage import grey_dilation

    rng = np.random.default_rng(seed_rng)
    renderer = FontRenderer(font_path)
    glyph_set = renderer._glyph_set
    scale = cap_px / (renderer.units_per_em * 0.7)

    chars = LATIN_SEED_SETS.get(seed, seed)
    step = int(cap_px * 2.2)
    page = np.full((int(cap_px * 3.0 * len(rows)) + cap_px, step * max(rows) + 2 * cap_px, 3), 238.0)

    index = 0
    for r, count in enumerate(rows):
        baseline = cap_px * (2.0 + 3.0 * r)
        x = cap_px
        for _ in range(count):
            char = chars[index]
            index += 1
            name = renderer.glyph_name(GlyphSpec(char))
            tile = int(cap_px * 3)
            advance = float(getattr(glyph_set[name], "width", 0) or 0) * scale
            flat = FlatteningPen(glyph_set, scale=scale, offset_x=tile * 0.25, offset_y=tile * 0.5)
            glyph_set[name].draw(flat)
            cover = fill_contours(flat.contours, tile, tile, 4)
            if pen > 1.0:  # a pen lays down a far wider stroke than a printed stem
                cover = grey_dilation(cover, size=(int(pen), int(pen)))

            # A hand does not hold the line: the row slopes and letters wobble.
            top = int(baseline + slope * x + rng.normal(0, cap_px * 0.03) - tile * 0.5)
            left = int(x + rng.normal(0, cap_px * 0.05))
            y0, x0 = max(top, 0), max(left, 0)
            y1, x1 = min(top + tile, page.shape[0]), min(left + tile, page.shape[1])
            patch = cover[y0 - top : y1 - top, x0 - left : x1 - left][..., None]
            region = page[y0:y1, x0:x1]
            page[y0:y1, x0:x1] = region * (1 - patch) + INK_RGB * patch
            x += int(advance + step * 0.55)

    return np.clip(page, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("font")
    parser.add_argument("--pen", type=float, default=7.0, help="pen width in pixels")
    parser.add_argument("--seed", default="seed30")
    parser.add_argument("--save", default=None)
    parser.add_argument("--debug", default=None)
    args = parser.parse_args()

    sheet = write_sheet(args.font, args.seed, pen=args.pen)
    photo = degrade(sheet)
    if args.save:
        from PIL import Image

        Image.fromarray(photo).save(args.save)

    def intake(image):
        cells = freehand_cells(image, LATIN_CORE, args.seed, FreehandConfig(), args.debug)
        return normalize_samples(cells, IntakeConfig(threshold_window=0.9)).images

    clean = intake(sheet)
    degraded = intake(photo)

    corpus, _ = FontRenderer(args.font, RenderConfig()).render(LATIN_CORE)
    keys = [k for k in clean if k in degraded and k in corpus]

    robust = [tolerant_f1(degraded[k], clean[k]) for k in keys]
    framing = [tolerant_f1(clean[k], corpus[k].image) for k in keys]
    framing_iou = [iou(clean[k], corpus[k].image) for k in keys]

    print(f"font: {Path(args.font).name} | {len(keys)} of {len(clean)} glyphs recovered")
    print(f"  photo robustness  (degraded vs clean sheet): tolF1 {np.mean(robust):.3f} "
          f"(min {np.min(robust):.3f})")
    print(f"  framing agreement (intake vs corpus frame) : tolF1 {np.mean(framing):.3f} "
          f"(min {np.min(framing):.3f})  IoU {np.mean(framing_iou):.3f}")

    ink = lambda d: np.mean([(v > 0.5).sum() for v in d.values()])  # noqa: E731
    print(f"  ink px per glyph: intake {ink(clean):.0f} vs corpus "
          f"{ink({k: v.image for k, v in corpus.items() if k in keys}):.0f}")
    worst = sorted(zip(keys, framing), key=lambda kv: kv[1])[:6]
    print("  worst framing: " + " ".join(
        f"{chr(int(k.split('.')[0], 16))}:{v:.2f}" for k, v in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
