"""Full product path on a handwriting font the model has never seen.

Stands in for gate 3 until real photographed handwriting exists. The "writer"
is a handwriting-style font that is not in the training corpus (the Windows
fonts Ink Free, Segoe Print and Segoe Script are not in Google Fonts), so:

1. its 24 seed letters are written into the template and photographed —
   simulated, with perspective, uneven light, blur, noise and JPEG;
2. the photo goes through the real intake path;
3. the model generates the other 51 glyphs from those 24;
4. the result is assembled into an OTF and shape-checked;
5. every generated glyph is scored against the font's real glyph — ground truth
   that real handwriting could never provide — beside the content-copy baseline
   and the leave-one-out score on the seed letters.

Usage::

    python scripts/handwriting_e2e.py --checkpoint weights.pt \\
        --writer C:/Windows/Fonts/Inkfree.ttf --out output/inkfree
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont.charset import LATIN_CORE, seed_charset  # noqa: E402
from hfont.data.render import FontRenderer, RenderConfig  # noqa: E402
from hfont.evaluate.report import _mean, _score, leave_one_out  # noqa: E402
from hfont.evaluate.shaping import check_font, render_string  # noqa: E402
from hfont.fontbuild.build import BuildConfig, FontMetadata, rasters_to_font, save_font  # noqa: E402
from hfont.generate import content_images_from_font, generate_rasters, load_checkpoint  # noqa: E402
from hfont.intake import IntakeConfig, build_template, extract_cells, normalize_samples  # noqa: E402
from simulate_template_photo import degrade, write_into_template  # noqa: E402

SAMPLE_TEXT = "Hamburgefonstiv quick jumps"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--writer", required=True, help="handwriting-style font standing in for a person")
    parser.add_argument("--content-font", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-photo", action="store_true", help="skip the photo simulation")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = Path(args.writer)

    # 1-2. Write the template, photograph it, run intake.
    page, layout = build_template(LATIN_CORE)
    written = write_into_template(page, layout, str(writer))
    photo = written if args.no_photo else degrade(written, seed=0)
    by_char = {g.char: g for g in LATIN_CORE}
    cells = extract_cells(photo, layout)
    refs = normalize_samples({by_char[c]: img for c, img in cells.items()}, IntakeConfig()).images

    # 3. Generate.
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content_font, LATIN_CORE)
    images, advances = generate_rasters(model, refs, content)

    # 5. Score against the writer's real glyphs.
    truth_rasters, _ = FontRenderer(writer, RenderConfig()).render(LATIN_CORE)
    seed_keys = {g.key for g in seed_charset(LATIN_CORE, "seed24")}
    held_out = [k for k in truth_rasters if k not in seed_keys and k in images and k in content]
    model_rows = [_score(images[k], truth_rasters[k].image) for k in held_out]
    base_rows = [_score(content[k], truth_rasters[k].image) for k in held_out]
    loo = leave_one_out(model, refs, content)
    per_glyph = sorted(
        ((k, _score(images[k], truth_rasters[k].image)["tol_f1"]) for k in held_out),
        key=lambda kv: kv[1],
    )

    # 4. Assemble, shape-check, render a sample line beside the real font.
    fb = rasters_to_font(images, advances, LATIN_CORE,
                         FontMetadata(family=f"Generated {writer.stem}"), BuildConfig())
    font_path = save_font(fb, out / f"generated_{writer.stem}.otf")
    shaping = check_font(font_path, charset=LATIN_CORE)

    from PIL import Image

    real = render_string(writer, SAMPLE_TEXT, size=56)
    fake = render_string(font_path, SAMPLE_TEXT, size=56)
    width = max(real.shape[1], fake.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, width - a.shape[1])))
    strip = np.concatenate([pad(real), np.ones((6, width)) * 0.4, pad(fake)], axis=0)
    Image.fromarray(((1 - strip) * 255).astype(np.uint8)).save(out / "comparison.png")

    report = {
        "writer": writer.name,
        "photo_simulated": not args.no_photo,
        "held_out_glyphs": len(held_out),
        "model": _mean(model_rows),
        "content_copy_baseline": _mean(base_rows),
        "leave_one_out_on_seeds": loo,
        "worst_glyphs": [(chr(int(k.split(".")[0], 16)), round(s, 3)) for k, s in per_glyph[:8]],
        "font": str(font_path),
        "shaping_ok": shaping.ok,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    m, b = report["model"], report["content_copy_baseline"]
    print(f"{writer.name}: {len(held_out)} unseen glyphs generated from a "
          f"{'photographed' if not args.no_photo else 'clean'} template")
    print(f"  model    : IoU {m['iou']:.3f}  tolF1 {m['tol_f1']:.3f}  SSIM {m['ssim']:.3f}  coverage {m['coverage']:.2f}")
    print(f"  baseline : IoU {b['iou']:.3f}  tolF1 {b['tol_f1']:.3f}  SSIM {b['ssim']:.3f}  coverage {b['coverage']:.2f}")
    print(f"  LOO seeds: tolF1 {loo.get('tol_f1', float('nan')):.3f}")
    print(f"  worst    : {report['worst_glyphs']}")
    print(f"  font     : {font_path.name} | shaping {'PASS' if shaping.ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
