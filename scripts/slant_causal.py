"""Does slant *cause* the low scores, or merely correlate with them?

Round 1 attributed ~0.20 of writer 1's gap to "the model regularizing a hard
hand", resting on r = +0.50 between leave-one-out and an axis-based slant
measure across ten fonts. That is a residual plus weak correlational evidence,
and the axis measure confounded slant with glyph proportions. This tests the
claim four ways, with the stem-based measure from scripts/slant.py:

* ``--measure``     slant and LOO for the ten held-out fonts, writer 1's own
                    letters, and what the model generates from them.
* ``--shear-curve`` intervention: shear upright, high-scoring fonts by 0-25 deg
                    (references and targets alike, framing untouched) and watch
                    the score. If slant causes the drop, the curve falls.
* ``--deslant``     the inference-time remedy: shear a sample upright, generate,
                    shear the output back, score against the real letters.
* ``--dilate``      the rival explanation: thicken the strokes instead and see
                    how much of the same ground it covers.

    python scripts/slant_causal.py --measure --shear-curve --deslant --dilate
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

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.evaluate.metrics import tolerant_f1  # noqa: E402
from hfont.evaluate.report import leave_one_out  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import load_freehand_photo  # noqa: E402
from slant import font_slant, load_font, shear  # noqa: E402

FONTS = ["segoepr", "segoesc", "Inkfree", "BRADHITC", "Gabriola",
         "MISTRAL", "PRISTINA", "FREESCPT", "JUICE___", "RAGE"]
UPRIGHT = ["Gabriola", "segoepr", "JUICE___"]


def edge_touch(images: dict[str, np.ndarray]) -> int:
    return sum(1 for v in images.values()
               if (v[0] > 0.5).any() or (v[-1] > 0.5).any()
               or (v[:, 0] > 0.5).any() or (v[:, -1] > 0.5).any())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=str(ROOT / "MyHandwriting.jpeg"))
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--out", default=str(ROOT / "output/myhand_diag/round2"))
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--shear-curve", action="store_true")
    parser.add_argument("--deslant", action="store_true")
    parser.add_argument("--dilate", action="store_true")
    parser.add_argument("--controls", action="store_true",
                        help="separate the two interventions from their artefacts")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)
    mine = load_freehand_photo(args.photo, LATIN_CORE, "seed30", 128)
    results: dict[str, object] = {}

    def generated_from(refs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        made = {}
        for key in refs:
            others = {k: v for k, v in refs.items() if k != key}
            made[key] = generate_rasters(model, others, {key: content[key]})[0][key]
        return made

    # ---- 2. slant and score, measured again with the stem-based metric ----
    if args.measure:
        print("== stem slant vs leave-one-out (positive = leaning right)\n")
        print(f"{'font':10s} {'LOO':>6s} {'stem slant':>11s}")
        rows = []
        for name in FONTS:
            images = load_font(name)
            if images is None:
                continue
            loo = leave_one_out(model, images, content)["tol_f1"]
            rows.append((name, loo, font_slant(images)))
            print(f"{name:10s} {loo:6.3f} {rows[-1][2]:+11.1f}")
        loo = np.array([r[1] for r in rows])
        sl = np.array([r[2] for r in rows])
        r_signed = float(np.corrcoef(loo, sl)[0, 1])
        r_abs = float(np.corrcoef(loo, np.abs(sl))[0, 1])
        print(f"\ncorrelation LOO vs slant:      r = {r_signed:+.2f}")
        print(f"correlation LOO vs |slant|:    r = {r_abs:+.2f}   (n = {len(rows)})")

        mine_slant = font_slant(mine)
        made = generated_from(mine)
        print(f"\nwriter 1: real letters {mine_slant:+.1f} deg, "
              f"generated {font_slant(made):+.1f} deg")
        results["measure"] = {"fonts": rows, "r_signed": r_signed, "r_abs": r_abs,
                              "writer1_real": mine_slant, "writer1_generated": font_slant(made)}

    # ---- 3. intervention: shear upright fonts ----------------------------
    if args.shear_curve:
        print("\n== shearing upright fonts (references and targets alike)\n")
        print(f"{'font':10s} " + "  ".join(f"{a:>6.0f} deg" for a in (0, 10, 15, 25)))
        curves = {}
        for name in UPRIGHT:
            images = load_font(name)
            if images is None:
                continue
            row = []
            for angle in (0, 10, 15, 25):
                sheared = {k: np.clip(shear(v, angle), 0, 1) for k, v in images.items()}
                row.append(leave_one_out(model, sheared, content)["tol_f1"])
            curves[name] = row
            print(f"{name:10s} " + "  ".join(f"{v:10.3f}" for v in row))
        results["shear_curve"] = curves

    # ---- 4. test-time deslant --------------------------------------------
    if args.deslant:
        print("\n== test-time deslant: straighten, generate, lean back\n")
        print(f"{'sample':10s} {'slant':>7s} {'as-is':>7s} {'deslanted':>10s} {'change':>8s} "
              f"{'clipped':>8s}")
        deslant = {}
        subjects = {"writer1": mine}
        for name in ("BRADHITC", "segoesc"):
            images = load_font(name)
            if images is not None:
                subjects[name] = images
        for name, refs in subjects.items():
            angle = font_slant(refs)
            plain = leave_one_out(model, refs, content)["tol_f1"]
            straight = {k: np.clip(shear(v, -angle), 0, 1) for k, v in refs.items()}
            made = generated_from(straight)
            back = {k: np.clip(shear(v, angle), 0, 1) for k, v in made.items()}
            scored = float(np.mean([tolerant_f1(back[k], refs[k]) for k in refs]))
            deslant[name] = {"slant": angle, "as_is": plain, "deslanted": scored,
                             "clipped": edge_touch(straight)}
            print(f"{name:10s} {angle:+7.1f} {plain:7.3f} {scored:10.3f} "
                  f"{scored - plain:+8.3f} {edge_touch(straight):8d}")
        results["deslant"] = deslant

    # ---- 5. the rival explanation: thin strokes --------------------------
    if args.dilate:
        from skimage.morphology import dilation, disk

        print("\n== thicker strokes instead (dilating references and targets)\n")
        print(f"{'sample':10s} {'as-is':>7s} {'+1px':>8s} {'+2px':>8s}")
        thick = {}
        subjects = {"writer1": mine}
        images = load_font("BRADHITC")
        if images is not None:
            subjects["BRADHITC"] = images
        for name, refs in subjects.items():
            row = [leave_one_out(model, refs, content)["tol_f1"]]
            for radius in (1, 2):
                fat = {k: dilation(v, disk(radius)) for k, v in refs.items()}
                row.append(leave_one_out(model, fat, content)["tol_f1"])
            thick[name] = row
            print(f"{name:10s} {row[0]:7.3f} {row[1]:8.3f} {row[2]:8.3f}")
        results["dilate"] = thick

    # ---- controls: is either intervention measuring what it claims? ------
    if args.controls:
        from skimage.morphology import dilation, disk

        print("\n== control 1: how much of the shear drop is resampling?\n")
        print(f"{'font':10s} {'0 deg':>8s} {'+15/-15':>9s} {'15 deg':>8s}")
        resample = {}
        for name in UPRIGHT:
            images = load_font(name)
            if images is None:
                continue
            # Sheared there and back: two interpolations, no net slant.
            roundtrip = {k: np.clip(shear(shear(v, 15.0), -15.0), 0, 1) for k, v in images.items()}
            plain = leave_one_out(model, images, content)["tol_f1"]
            both = leave_one_out(model, roundtrip, content)["tol_f1"]
            sheared = {k: np.clip(shear(v, 15.0), 0, 1) for k, v in images.items()}
            leaned = leave_one_out(model, sheared, content)["tol_f1"]
            resample[name] = {"plain": plain, "roundtrip": both, "sheared": leaned}
            print(f"{name:10s} {plain:8.3f} {both:9.3f} {leaned:8.3f}")
        results["control_resampling"] = resample

        print("\n== control 2: are thick strokes easier, or just easier to score?\n")
        print(f"{'sample':10s} {'dilation':>9s} {'model':>7s} {'content copy':>13s} {'margin':>8s}")
        fatness = {}
        subjects = {"writer1": mine}
        images = load_font("BRADHITC")
        if images is not None:
            subjects["BRADHITC"] = images
        for name, refs in subjects.items():
            rows = []
            for radius in (0, 1, 2):
                fat = ({k: dilation(v, disk(radius)) for k, v in refs.items()} if radius
                       else dict(refs))
                scored = leave_one_out(model, fat, content)["tol_f1"]
                # The same targets, scored against an unchanging content copy.
                base = float(np.mean([tolerant_f1(content[k], fat[k]) for k in fat]))
                rows.append({"radius": radius, "model": scored, "baseline": base,
                             "margin": scored - base})
                print(f"{name:10s} {radius:9d} {scored:7.3f} {base:13.3f} {scored - base:+8.3f}")
            fatness[name] = rows
        results["control_fatness"] = fatness

    (out / "slant_causal.json").write_text(json.dumps(results, indent=1, default=float),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
