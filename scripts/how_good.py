"""How good is the model for a real person, letter by letter?

The evaluation elsewhere in this project reports one number per writer. That
hides the thing a user actually experiences: some letters come out right and
some come out wrong, and which ones matters more than the mean. A font whose
`m` is mush is unusable however well it scores overall.

Leave-one-out is the only honest measurement available for a photographed hand
-- hold out one written letter, encode the style from the other 29, generate it,
and compare against what the person actually wrote. It is scored here on all
three metrics at once, because the review established that tolerant F1 alone
cannot see stroke-level error (output/review/REVIEW.md, finding 1).

Also renders a picture, since the numbers are a proxy and the picture is not.

    python scripts/how_good.py
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
from hfont.evaluate.metrics import cl_dice, iou, tolerant_f1  # noqa: E402
from hfont.generate import (  # noqa: E402
    _stack_refs,
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import load_freehand_photo  # noqa: E402

WRITERS = {
    "writer1": ROOT / "MyHandwriting.jpeg",
    "writer2": ROOT / "output/AntoineHand/AntoineHandwriting.jpg",
    "writer3": ROOT / "output/JadHand/JadHandWriting.jpeg",
}


def leave_one_out(model, refs, content):
    """Per letter: generate it from the other 29 and score against the real one."""
    import torch

    index = {g.key: i for i, g in enumerate(model.charset)}
    rows = {}
    with torch.no_grad():
        for key in refs:
            others = [refs[k] for k in refs if k != key]
            stack, mask = _stack_refs(others, model.device)
            style = model.generator.encode_style(stack, mask)
            cont = torch.from_numpy(
                content[key][None, None].astype(np.float32)
            ).to(model.device) * 2.0 - 1.0
            out = model.generator.decode(
                cont, torch.tensor([index[key]], device=model.device), style
            )
            gen = ((out["image"].float() + 1.0) * 0.5).clamp(0, 1).cpu().numpy()[0, 0]
            rows[key] = {
                "gen": gen,
                "tol_f1": tolerant_f1(gen, refs[key]),
                "iou": iou(gen, refs[key]),
                "cl_dice": cl_dice(gen, refs[key]),
            }
    return rows


def strip(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.pad(1.0 - im, 1, constant_values=0.6) for im in images], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--out", default=str(ROOT / "output/review"))
    args = ap.parse_args()

    from PIL import Image

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, photo in WRITERS.items():
        refs = load_freehand_photo(str(photo), LATIN_CORE, "seed30", model.image_size)
        rows = leave_one_out(model, refs, content)
        by_char = {next(g.char for g in LATIN_CORE if g.key == k): v for k, v in rows.items()}

        print(f"\n=== {name} — {len(rows)} letters held out one at a time ===")
        order = sorted(by_char, key=lambda c: by_char[c]["cl_dice"], reverse=True)
        print("  best :  " + "  ".join(f"{c} {by_char[c]['cl_dice']:.2f}" for c in order[:8]))
        print("  worst:  " + "  ".join(f"{c} {by_char[c]['cl_dice']:.2f}" for c in order[-8:]))
        stats = {m: float(np.mean([r[m] for r in rows.values()])) for m in ("tol_f1", "iou", "cl_dice")}
        sd = {m: float(np.std([r[m] for r in rows.values()])) for m in ("tol_f1", "iou", "cl_dice")}
        print(f"  mean :  tol-F1 {stats['tol_f1']:.3f}+-{sd['tol_f1']:.3f}   "
              f"IoU {stats['iou']:.3f}+-{sd['iou']:.3f}   "
              f"clDice {stats['cl_dice']:.3f}+-{sd['cl_dice']:.3f}")
        good = sum(1 for r in rows.values() if r["cl_dice"] >= 0.5)
        print(f"  letters at clDice >= 0.50: {good} of {len(rows)}")
        summary[name] = {"mean": stats, "sd": sd, "n": len(rows),
                         "per_char": {c: {m: v[m] for m in ("tol_f1", "iou", "cl_dice")}
                                      for c, v in by_char.items()}}

        # Picture: what they wrote, what the model drew without seeing it, and
        # letters they never wrote at all.
        chars = sorted(by_char)
        real = strip([refs[next(g.key for g in LATIN_CORE if g.char == c)] for c in chars])
        fake = strip([by_char[c]["gen"] for c in chars])

        generated, _ = generate_rasters(model, refs, content)
        unseen = [g for g in LATIN_CORE if g.key not in refs and g.key in generated][:30]
        novel = strip([generated[g.key] for g in unseen])
        pad = np.ones((real.shape[0], max(0, real.shape[1] - novel.shape[1])))
        novel = np.concatenate([novel, pad], axis=1) if pad.shape[1] else novel[:, : real.shape[1]]

        gap = np.ones((6, real.shape[1]))
        sheet = np.concatenate([real, gap, fake, gap, gap, novel], axis=0)
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"how_good_{name}.png"
        )
        print(f"  wrote {out_dir / f'how_good_{name}.png'}  "
              f"(row 1 = written, row 2 = generated blind, row 3 = never written)")

    (out_dir / "how_good.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
