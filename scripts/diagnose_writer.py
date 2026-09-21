"""Where does one writer's leave-one-out score go?

Answers, for a real photographed sample: which letters are dragging the score
down, whether the model is beating the content-copy baseline on each of them,
and whether the generated letters are heavier or more regular than the hand
that wrote them (README known issues 1 and 3).

Per seed glyph it reports leave-one-out tol-F1 and IoU, the ink ratio of the
generated letter against the real one, the Noto Sans content-copy baseline for
the same glyph, and two shape measures: mean stroke half-width and the number
of Bézier segments the tracer needs, generated against real. A hand drawn more
regularly than it was written shows up as fewer segments and a flatter spread
of stroke widths.

It also writes a side-by-side strip, worst glyph first.

    python scripts/diagnose_writer.py --photo MyHandwriting.jpeg \
        --checkpoint runs/phase1c/hfont_step030000.pt --out output/myhand_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.evaluate.metrics import iou, tolerant_f1  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import FreehandConfig, load_freehand_photo  # noqa: E402
from hfont.vector.trace import TraceConfig, trace_image  # noqa: E402


def shape_stats(image: np.ndarray) -> tuple[float, float, int]:
    """Mean stroke half-width, its spread, and how many curve segments it takes."""
    from scipy.ndimage import distance_transform_edt

    mask = image > 0.5
    if not mask.any():
        return 0.0, 0.0, 0
    spread = distance_transform_edt(mask)[mask]
    try:
        segments = sum(len(path.segments) for path in trace_image(image, TraceConfig()))
    except Exception:
        segments = 0
    return float(np.mean(spread)), float(np.std(spread)), int(segments)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--seed", default="seed30")
    parser.add_argument("--rows", default="10,10,10")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = FreehandConfig(rows=tuple(int(n) for n in args.rows.split(",")))
    refs = load_freehand_photo(args.photo, LATIN_CORE, args.seed, 128, cfg)
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)

    rows = []
    for key, real in refs.items():
        char = chr(int(key.split(".")[0], 16))
        others = {k: v for k, v in refs.items() if k != key}
        made, _ = generate_rasters(model, others, {key: content[key]})
        gen = made[key]

        real_w, real_sd, real_seg = shape_stats(real)
        gen_w, gen_sd, gen_seg = shape_stats(gen)
        rows.append({
            "char": char,
            "key": key,
            "tol_f1": tolerant_f1(gen, real),
            "iou": iou(gen, real),
            "baseline_tol_f1": tolerant_f1(content[key], real),
            "ink_ratio": float((gen > 0.5).sum() / max((real > 0.5).sum(), 1)),
            "real_stroke": real_w,
            "gen_stroke": gen_w,
            "real_stroke_sd": real_sd,
            "gen_stroke_sd": gen_sd,
            "real_segments": real_seg,
            "gen_segments": gen_seg,
            "real_ink_px": int((real > 0.5).sum()),
        })

    rows.sort(key=lambda r: r["tol_f1"])
    (out / "per_glyph.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    from PIL import Image

    strip = np.concatenate(
        [np.concatenate([refs[r["key"]],
                         generate_rasters(model, {k: v for k, v in refs.items() if k != r["key"]},
                                          {r["key"]: content[r["key"]]})[0][r["key"]]], axis=0)
         for r in rows], axis=1)
    lines = [strip[:, i * 1280 : (i + 1) * 1280] for i in range(int(np.ceil(strip.shape[1] / 1280)))]
    width = max(line.shape[1] for line in lines)
    canvas = np.zeros((sum(line.shape[0] + 8 for line in lines), width), dtype=np.float32)
    y = 0
    for line in lines:
        canvas[y : y + line.shape[0], : line.shape[1]] = line
        y += line.shape[0] + 8
    Image.fromarray(((1 - canvas) * 255).astype(np.uint8)).save(out / "per_glyph_worst_first.png")

    print(f"{'char':4s} {'tolF1':>6s} {'IoU':>6s} {'base':>6s} {'ink':>6s} "
          f"{'stroke r/g':>12s} {'segs r/g':>10s}")
    for r in rows:
        print(f"{r['char']:4s} {r['tol_f1']:6.3f} {r['iou']:6.3f} {r['baseline_tol_f1']:6.3f} "
              f"{r['ink_ratio']:6.2f} {r['real_stroke']:5.1f}/{r['gen_stroke']:<5.1f} "
              f"{r['real_segments']:4d}/{r['gen_segments']:<4d}")

    arr = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    print(f"\nmean tol-F1 {arr('tol_f1').mean():.3f} | baseline {arr('baseline_tol_f1').mean():.3f} "
          f"| beats baseline on {int((arr('tol_f1') > arr('baseline_tol_f1')).sum())}/{len(rows)}")
    print(f"ink ratio median {np.median(arr('ink_ratio')):.2f} | "
          f"stroke half-width real {arr('real_stroke').mean():.2f} vs generated {arr('gen_stroke').mean():.2f} | "
          f"segments real {arr('real_segments').mean():.1f} vs generated {arr('gen_segments').mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
