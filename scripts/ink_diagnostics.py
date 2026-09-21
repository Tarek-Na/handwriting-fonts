"""Where does the model's excess ink come from? (README, known issue 1)

Runs the four checks behind that issue on local system fonts, generating each
font's 51 non-seed glyphs from its 24 seed glyphs:

1. content weight  - same references, Noto Sans instanced at wght 300 vs 700
2. shrinkage       - ink ratio for the lightest / middle / heaviest fonts
3. soft edges      - ink ratio on soft values vs. thresholded at 0.5
4. trace level     - ink ratio, IoU and tol-F1 when cutting at 0.5 ... 0.8

    python scripts/ink_diagnostics.py --checkpoint runs/phase1c/hfont_step030000.pt
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, seed_charset  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.evaluate.metrics import iou, tolerant_f1  # noqa: E402
from hfont.generate import content_images_from_font, generate_rasters, load_checkpoint  # noqa: E402

LEVELS = (0.5, 0.6, 0.7, 0.8)


def noto_instance(var_font: Path, weight: int, out_dir: Path) -> Path:
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    path = out_dir / f"noto_{weight}.ttf"
    instancer.instantiateVariableFont(TTFont(var_font), {"wght": weight, "wdth": 100}).save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--fonts", default="C:/Windows/Fonts/*.ttf")
    parser.add_argument("--n-fonts", type=int, default=40)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    seed_keys = {g.key for g in seed_charset(LATIN_CORE, "seed24")}
    keys = [g.key for g in LATIN_CORE if g.key not in seed_keys]

    tmp = Path(tempfile.mkdtemp())
    content = {w: content_images_from_font(noto_instance(Path(args.content_font), w, tmp), LATIN_CORE)
               for w in (300, 400, 700)}

    paths = sorted(glob.glob(args.fonts))
    random.Random(0).shuffle(paths)
    rows = []
    for path in paths:
        if len(rows) >= args.n_fonts:
            break
        try:
            rendered, _ = FontRenderer(path, RenderConfig()).render(LATIN_CORE)
        except Exception:
            continue
        if not all(k in rendered for k in list(seed_keys) + keys):
            continue
        truth = {k: np.round(rendered[k].image * 255) / 255 for k in rendered}
        if any(truth[k].sum() < 20 for k in keys):  # symbol or broken fonts
            continue
        refs = {k: truth[k] for k in seed_keys}
        out = {w: generate_rasters(model, refs, {k: content[w][k] for k in keys})[0]
               for w in (300, 400, 700)}

        o = out[400]
        true_ink = sum(truth[k].sum() for k in keys)
        true_bin = [(truth[k] > 0.5).astype(np.float32) for k in keys]
        row = {
            "name": Path(path).stem,
            "ink_per_glyph": true_ink / len(keys),
            "soft": sum(o[k].sum() for k in keys) / true_ink,
            "w300": sum(out[300][k].sum() for k in keys) / true_ink,
            "w700": sum(out[700][k].sum() for k in keys) / true_ink,
        }
        for lv in LEVELS:
            pred = [(o[k] > lv).astype(np.float32) for k in keys]
            row[lv] = (
                sum(p.sum() for p in pred) / sum(t.sum() for t in true_bin),
                float(np.mean([iou(p, t) for p, t in zip(pred, true_bin)])),
                float(np.mean([tolerant_f1(p, t) for p, t in zip(pred, true_bin)])),
            )
        rows.append(row)

    print(f"{len(rows)} fonts, {len(keys)} generated glyphs each\n")
    w3 = np.array([r["w300"] for r in rows])
    w7 = np.array([r["w700"] for r in rows])
    print(f"1. content weight: ink ratio {w3.mean():.3f} with Light content, {w7.mean():.3f} with Bold")

    rows.sort(key=lambda r: r["ink_per_glyph"])
    q = len(rows) // 4
    x = np.log([r["ink_per_glyph"] for r in rows])
    y = np.log([r["soft"] for r in rows])
    parts = [("lightest quarter", rows[:q]), ("middle half", rows[q:-q]), ("heaviest quarter", rows[-q:])]
    print("2. shrinkage: " + ", ".join(f"{n} {np.mean([r['soft'] for r in p]):.2f}" for n, p in parts)
          + f" (log-log slope {np.polyfit(x, y, 1)[0]:.2f})")

    soft = np.mean([r["soft"] for r in rows])
    print(f"3. soft edges: ink ratio {soft:.3f} soft, {np.mean([r[0.5][0] for r in rows]):.3f} thresholded at 0.5")

    print("4. trace level:")
    for lv in LEVELS:
        r = np.array([row[lv] for row in rows])
        print(f"   cut at {lv:.1f}: ink ratio {r[:, 0].mean():.3f}  IoU {r[:, 1].mean():.4f}  tol-F1 {r[:, 2].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
