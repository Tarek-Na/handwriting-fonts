"""Does the style input actually decide whose handwriting comes out?

The gate used for the scale-augmentation experiment measured how similar two
writers' *generated* letters were to each other, and it was gameable: a worse
model produces noisier output, the outputs diverge, and the metric improves
while nothing about style got better. That is exactly what happened
(output/review/STYLE_COLLAPSE.md) and it was only caught because the style
vectors moved the wrong way at the same time.

This asks a question a worse model cannot fake. Both generations come from the
same model, so quality cancels:

    take a letter writer A actually wrote, so there is ground truth
    generate it from A's other letters   -> should look like A's
    generate it from B's letters         -> should not

If the style path works, the first is closer to A's real letter than the second.
Score it as an accuracy over letters and writer pairs. A model that ignores
style entirely scores 0.50 by construction, whatever its image quality. That
floor is what makes the number trustworthy.

    python scripts/style_identification.py --checkpoint runs/phase1c/hfont_step030000.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.evaluate.metrics import cl_dice  # noqa: E402
from hfont.generate import (  # noqa: E402
    _stack_refs,
    content_images_from_font,
    load_checkpoint,
)
from hfont.intake import load_freehand_photo  # noqa: E402

WRITERS = {
    "writer1": ROOT / "MyHandwriting.jpeg",
    "writer2": ROOT / "output/AntoineHand/AntoineHandwriting.jpg",
    "writer3": ROOT / "output/JadHand/JadHandWriting.jpeg",
}


@torch.no_grad()
def generate(model, refs, content, key, exclude=None):
    """Draw ``key`` in the style of ``refs``, optionally holding a letter out."""
    pool = [v for j, v in refs.items() if j != exclude]
    stack, mask = _stack_refs(pool, model.device)
    attends = getattr(model.generator, "ref_attention", None) is not None
    if attends:
        style, ref_feats = model.generator.encode_style_full(stack, mask)
    else:
        style, ref_feats = model.generator.encode_style(stack, mask), None
    index = {g.key: i for i, g in enumerate(model.charset)}
    cont = torch.from_numpy(content[key][None, None].astype(np.float32))
    cont = cont.to(model.device) * 2.0 - 1.0
    out = model.generator.decode(
        cont, torch.tensor([index[key]], device=model.device), style,
        ref_feats, mask if attends else None,
    )
    return ((out["image"].float() + 1.0) * 0.5).clamp(0, 1).cpu().numpy()[0, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--label", default="shipped")
    ap.add_argument("--out", default=str(ROOT / "output/review"))
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    refs = {n: load_freehand_photo(str(p), LATIN_CORE, "seed30", model.image_size)
            for n, p in WRITERS.items()}

    rows, margins = [], []
    for owner, other in itertools.permutations(refs, 2):
        shared = [k for k in refs[owner] if k in refs[other] and k in content]
        hits, gaps = 0, []
        for key in shared:
            truth = refs[owner][key]
            # Held out of its own style set, so the correct arm has no more
            # information about this letter than the wrong arm does.
            right = generate(model, refs[owner], content, key, exclude=key)
            wrong = generate(model, refs[other], content, key)
            gap = cl_dice(right, truth) - cl_dice(wrong, truth)
            gaps.append(gap)
            hits += gap > 0
        acc = hits / max(len(shared), 1)
        rows.append({"owner": owner, "other": other, "n": len(shared),
                     "accuracy": acc, "margin": float(np.mean(gaps))})
        margins += gaps
        print(f"  {owner} vs {other:8s} n={len(shared):3d}  "
              f"identified {acc:.2f}  mean margin {np.mean(gaps):+.4f}")

    acc = float(np.mean([r["accuracy"] for r in rows]))
    margin = float(np.mean(margins))
    se = float(np.std(margins, ddof=1) / np.sqrt(len(margins)))
    print(f"\n{args.label}: identification accuracy {acc:.3f} (chance 0.500), "
          f"mean margin {margin:+.4f} [{margin - 1.96 * se:+.4f}, {margin + 1.96 * se:+.4f}]")
    print("  a model that ignores style scores 0.500 however good its images are")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"style_identification_{args.label}.json").write_text(
        json.dumps({"label": args.label, "accuracy": acc, "margin": margin,
                    "ci95": [margin - 1.96 * se, margin + 1.96 * se], "pairs": rows},
                   indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
