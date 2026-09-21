"""Stress-test handwriting intake with a synthetic photographed template.

Real photographed handwriting is the input the brief expects to fail quietly,
and there is none to test with yet. This gets as close as possible without it:

1. render a handwriting-style font into the printed template, as blue ink;
2. degrade the page the way a phone photo does - perspective, uneven light,
   blur, sensor noise, JPEG;
3. run the real intake path (marker detection, perspective removal, red-channel
   dropout, normalization) on both the clean page and the degraded photo.

Two numbers come out:

* photo robustness  - IoU between glyphs recovered from the degraded photo and
  from the clean page. Measures what the photograph itself costs.
* framing agreement - IoU between intake-normalized glyphs and the same glyphs
  normalized the way the training corpus was. The model only ever saw the
  second framing; any systematic offset here is a domain gap introduced by the
  pipeline itself, before handwriting style even enters into it.

Usage::

    python scripts/simulate_template_photo.py C:/Windows/Fonts/Inkfree.ttf
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.charset import LATIN_CORE, GlyphSpec, seed_charset  # noqa: E402
from hfont.data.raster import FlatteningPen, fill_contours  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.intake import (  # noqa: E402
    IntakeConfig,
    build_template,
    extract_cells,
    normalize_samples,
)

INK_RGB = np.array([30, 45, 120], dtype=np.float64)  # blue ballpoint


def write_into_template(page: np.ndarray, layout, font_path: str) -> np.ndarray:
    """Composite each seed glyph into its cell, on the baseline, as ink."""
    renderer = FontRenderer(font_path)
    glyph_set = renderer._glyph_set
    page = page.astype(np.float64).copy()
    scale = layout.cell * 0.5 / renderer.units_per_em

    for i, char in enumerate(layout.chars):
        name = renderer.glyph_name(GlyphSpec(char))
        if name is None:
            continue
        x0, y0, x1, y1 = layout.cell_box(i)
        advance = float(getattr(glyph_set[name], "width", 0) or 0) * scale
        pen = FlatteningPen(
            glyph_set,
            scale=scale,
            offset_x=(layout.cell - advance) / 2.0,
            offset_y=layout.cell * layout.baseline,
        )
        glyph_set[name].draw(pen)
        cover = fill_contours(pen.contours, layout.cell, layout.cell, 4)[..., None]
        region = page[y0:y1, x0:x1]
        page[y0:y1, x0:x1] = region * (1 - cover) + INK_RGB * cover
    return np.clip(page, 0, 255).astype(np.uint8)


def degrade(page: np.ndarray, seed: int = 0) -> np.ndarray:
    """Phone-photo simulation: perspective, lighting, blur, noise, JPEG."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage.transform import ProjectiveTransform, warp

    rng = np.random.default_rng(seed)
    h, w = page.shape[:2]

    # Shoot it on a table: pad with a darker background, then skew the corners.
    pad = int(0.08 * max(h, w))
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 90.0)
    canvas[pad:pad + h, pad:pad + w] = page
    H, W = canvas.shape[:2]
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    dst = src + rng.uniform(-0.04, 0.04, size=(4, 2)) * [W, H]
    t = ProjectiveTransform()
    t.estimate(src, dst)
    photo = np.stack(
        [warp(canvas[..., c], t, output_shape=(H, W), cval=90, preserve_range=True)
         for c in range(3)], axis=-1)

    # Uneven light: a smooth gradient from one corner, plus a warm white balance.
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.72 + 0.28 * (0.6 * xx / W + 0.4 * (1 - yy / H))
    photo = photo * light[..., None] * np.array([1.0, 0.97, 0.9])

    photo = np.stack([gaussian_filter(photo[..., c], 1.2) for c in range(3)], axis=-1)
    photo = photo + rng.normal(0, 4.0, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="JPEG", quality=80)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("font")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="write the simulated photo here")
    args = parser.parse_args()

    page, layout = build_template(LATIN_CORE)
    written = write_into_template(page, layout, args.font)
    photo = degrade(written, args.seed)
    if args.save:
        from PIL import Image
        Image.fromarray(photo).save(args.save)

    by_char = {g.char: g for g in LATIN_CORE}

    def intake(image):
        cells = extract_cells(image, layout)
        raw = {by_char[c]: img for c, img in cells.items()}
        return normalize_samples(raw, IntakeConfig()).images

    clean = intake(written)
    degraded = intake(photo)

    # How the training corpus frames the same glyphs.
    corpus, _ = FontRenderer(args.font, RenderConfig()).render(LATIN_CORE)
    seed_keys = [g.key for g in seed_charset(LATIN_CORE, "seed24")]

    from hfont.evaluate.metrics import tolerant_f1

    both = [k for k in seed_keys if k in clean and k in degraded and k in corpus]
    robust = [iou(degraded[k], clean[k]) for k in both]
    robust_f = [tolerant_f1(degraded[k], clean[k]) for k in both]
    framing = [iou(clean[k], corpus[k].image) for k in both]
    framing_f = [tolerant_f1(clean[k], corpus[k].image) for k in both]

    print(f"font: {Path(args.font).name} | {len(both)} glyphs recovered of {len(seed_keys)}")
    print(f"  photo robustness  (degraded vs clean page): IoU {np.mean(robust):.3f}  "
          f"tolF1 {np.mean(robust_f):.3f} (min {np.min(robust_f):.3f})")
    print(f"  framing agreement (intake vs corpus frame): IoU {np.mean(framing):.3f}  "
          f"tolF1 {np.mean(framing_f):.3f} (min {np.min(framing_f):.3f})")

    def extent(images):
        rows = [np.nonzero((images[k] > 0.5).any(axis=1))[0] for k in seed_keys if k in images]
        return np.mean([r[-1] - r[0] for r in rows if r.size])

    print(f"  mean glyph height px: intake {extent(clean):.1f} vs corpus "
          f"{extent({k: v.image for k, v in corpus.items()}):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
