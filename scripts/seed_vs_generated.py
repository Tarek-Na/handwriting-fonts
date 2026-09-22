"""Do the writer's own letters and the generated ones match in the same font?

An exported font is a mixture: the ~30 letters the writer actually wrote are
passed through unchanged, and the other ~45 are drawn by the model. If the two
halves differ in stroke weight or smoothness, the font is internally
inconsistent in a way no aggregate score reports — a word like "the" can put a
passed-through `t` beside a generated `h`.

Measured on the exported OTF itself, not on the rasters, so it includes
whatever the tracer and the font builder do to each half.

    python scripts/seed_vs_generated.py --font output/myhand_diag/round2/writer1.otf
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
from hfont.data.render import FontFrame, FontRenderer, RenderConfig  # noqa: E402


def stroke_stats(image: np.ndarray) -> tuple[float, float]:
    from scipy.ndimage import distance_transform_edt

    mask = image > 0.5
    if mask.sum() < 20:
        return float("nan"), float("nan")
    widths = distance_transform_edt(mask)[mask]
    return float(np.mean(widths)), float(np.std(widths))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True)
    parser.add_argument("--seed", default="seed30")
    parser.add_argument("--out", default=str(ROOT / "output/review"))
    args = parser.parse_args()

    rc = RenderConfig()
    frame = FontFrame(scale=rc.pixels_per_em / 1000.0, baseline_y=rc.baseline_px,
                      units_per_em=1000)
    rendered, _ = FontRenderer(args.font, rc).render(LATIN_CORE, frame=frame)

    seed_chars = set(LATIN_SEED_SETS.get(args.seed, args.seed))
    groups: dict[str, list[tuple[float, float, int]]] = {"written": [], "generated": []}
    for key, glyph in rendered.items():
        char = chr(int(key.split(".")[0], 16))
        mean, spread = stroke_stats(glyph.image)
        if np.isnan(mean):
            continue
        groups["written" if char in seed_chars else "generated"].append(
            (mean, spread, int((glyph.image > 0.5).sum())))

    print(f"{'half':12s} {'n':>4s} {'stroke px':>10s} {'spread':>8s} {'ink px':>8s}")
    summary = {}
    for name, rows in groups.items():
        if not rows:
            continue
        strokes = np.array([r[0] for r in rows])
        summary[name] = {
            "n": len(rows),
            "stroke_px": float(strokes.mean()),
            "stroke_sd": float(strokes.std()),
            "ink_px": float(np.mean([r[2] for r in rows])),
        }
        print(f"{name:12s} {len(rows):4d} {strokes.mean():10.2f} {strokes.std():8.2f} "
              f"{np.mean([r[2] for r in rows]):8.0f}")

    if len(summary) == 2:
        a, b = summary["written"], summary["generated"]
        gap = b["stroke_px"] - a["stroke_px"]
        # Welch's t-test, so the two halves' different spreads are handled.
        sa = a["stroke_sd"] ** 2 / a["n"]
        sb = b["stroke_sd"] ** 2 / b["n"]
        t = gap / max(np.sqrt(sa + sb), 1e-9)
        print(f"\ngenerated strokes are {gap:+.2f}px "
              f"({gap / max(a['stroke_px'], 1e-9):+.0%}) against the writer's own, t = {t:.1f}")
        print("a reader sees this wherever a written letter sits beside a generated one")
        summary["gap_px"] = gap
        summary["t"] = float(t)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.font).stem
    (out / f"seed_vs_generated_{name}.json").write_text(json.dumps(summary, indent=1),
                                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
