"""Is the learned advance head better than not having one?

Advance widths decide letter spacing in the exported font, and the width head
is the only part of the model whose output a reader sees as *rhythm* rather
than as letterforms. It is also the least examined: it is a two-layer MLP on
(style vector, char embedding) that never sees the content image or the glyph
it is predicting for.

That makes it worth comparing against baselines a person could write in an
afternoon, all of which use only information available at inference time:

    head       the model's prediction
    scaled     the content font's advance for that character, times one
               constant fitted on the writer's own seed letters
    ink        the generated glyph's own ink width plus fixed side bearings
    constant   the median advance of the writer's seed letters

Ground truth exists only for real font files, so this runs on the ten Windows
handwriting faces, held out from the Google Fonts corpus the model trained on.
Scores are mean absolute error in em, and errors are reported on the *generated*
glyphs only, since seed letters keep the writer's own ink either way.

    python scripts/advance_head.py
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

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)

FONTS = ["segoepr", "segoesc", "Inkfree", "BRADHITC", "Gabriola",
         "MISTRAL", "PRISTINA", "FREESCPT", "JUICE___", "RAGE"]


def ink_width(image: np.ndarray) -> float:
    """Width of the ink in canvas fractions, 0 if the glyph is blank."""
    columns = np.nonzero((image > 0.5).any(axis=0))[0]
    return float(columns[-1] - columns[0] + 1) / image.shape[1] if columns.size else 0.0


def render_with_advances(name: str, fonts_dir: str = "C:/Windows/Fonts"):
    for ext in (".ttf", ".TTF"):
        try:
            rendered, _ = FontRenderer(f"{fonts_dir}/{name}{ext}", RenderConfig()).render(LATIN_CORE)
        except Exception:
            continue
        return ({k: v.image for k, v in rendered.items()},
                {k: float(v.advance) for k, v in rendered.items()})
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--out", default=str(ROOT / "output/review"))
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    content_images = content_images_from_font(args.content, model.charset)
    content_rendered, _ = FontRenderer(args.content, RenderConfig()).render(LATIN_CORE)
    content_adv = {k: float(v.advance) for k, v in content_rendered.items()}

    seed_keys = {g.key for g in LATIN_CORE if g.char in LATIN_SEED_SETS["seed30"]}
    # The side bearing the build step guarantees, expressed in canvas fractions,
    # so the "ink" baseline is the same rule the font builder falls back to.
    bearing = 2 * 8 / 1000.0 * (RenderConfig().pixels_per_em / RenderConfig().size)

    rows = []
    for name in FONTS:
        images, true_adv = render_with_advances(name)
        if images is None or not seed_keys <= set(images):
            continue
        refs = {k: images[k] for k in images if k in seed_keys}
        generated, pred = generate_rasters(model, refs, content_images)

        # One constant fitted on the seed letters, exactly as could be done for
        # a photographed hand: how much wider is this writer than the content font?
        ratios = [ink_width(refs[k]) / w for k in refs
                  if (w := ink_width(content_images.get(k, np.zeros((1, 1))))) > 0]
        scale = float(np.median(ratios)) if ratios else 1.0
        seed_median = float(np.median([true_adv[k] for k in refs]))

        keys = [k for k in generated if k not in refs and k in true_adv and k in content_adv]
        if not keys:
            continue
        err = {
            "head": [abs(pred.get(k, 0.5) - true_adv[k]) for k in keys],
            "scaled": [abs(content_adv[k] * scale - true_adv[k]) for k in keys],
            "ink": [abs(ink_width(generated[k]) + bearing - true_adv[k]) for k in keys],
            "constant": [abs(seed_median - true_adv[k]) for k in keys],
        }
        row = {"font": name, "n": len(keys), "scale": scale}
        row.update({k: float(np.mean(v)) for k, v in err.items()})
        # Correlation with the truth says whether the relative widths are right,
        # which is what survives the whole-font tracking correction.
        row["r_head"] = float(np.corrcoef([pred.get(k, 0.5) for k in keys],
                                          [true_adv[k] for k in keys])[0, 1])
        row["r_scaled"] = float(np.corrcoef([content_adv[k] * scale for k in keys],
                                            [true_adv[k] for k in keys])[0, 1])
        rows.append(row)

    if not rows:
        print("no fonts rendered")
        return 1

    print(f"{'font':10s} {'n':>4s} " + " ".join(f"{m:>9s}" for m in
          ("head", "scaled", "ink", "constant")) + f"{'r head':>9s}{'r scaled':>10s}")
    for r in rows:
        print(f"{r['font']:10s} {r['n']:4d} " +
              " ".join(f"{r[m]:9.4f}" for m in ("head", "scaled", "ink", "constant")) +
              f"{r['r_head']:9.3f}{r['r_scaled']:10.3f}")

    print(f"\n{'MEAN':10s} {'':4s} " + " ".join(
        f"{np.mean([r[m] for r in rows]):9.4f}" for m in ("head", "scaled", "ink", "constant")) +
        f"{np.mean([r['r_head'] for r in rows]):9.3f}{np.mean([r['r_scaled'] for r in rows]):10.3f}")

    from scipy import stats
    for rival in ("scaled", "ink", "constant"):
        d = np.array([r["head"] - r[rival] for r in rows])
        t, p = stats.ttest_rel([r["head"] for r in rows], [r[rival] for r in rows])
        print(f"head - {rival:9s} {d.mean():+.4f} em  "
              f"95% CI [{d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)):+.4f}, "
              f"{d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)):+.4f}]  p={p:.4f}"
              f"   ({'head better' if d.mean() < 0 else 'baseline better'})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "advance_head.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
