"""Does the model actually imitate *this* writer, or a generic hand?

Every quality number this project reports is within one style: hold a letter
out, generate it, compare it to what that person wrote. That design cannot see
the failure where the model produces the *same* letters no matter whose
handwriting it was given, because the comparison is never made across styles.

So this asks the missing question directly. Generate the 45 letters nobody
wrote, once per style source, and measure how similar those outputs are to each
other. Two controls keep it honest:

    fonts       the model was trained on these; they set what a working style
                response looks like
    weight      photographed hands are ~1.2px where fonts are ~2.1px, so the
                hands are dilated and the fonts eroded to test whether stroke
                weight alone explains any clustering

Similarity is clDice throughout, because it is blind to stroke weight, which is
the one thing that reliably differs between a photograph and a rendered font.

    python scripts/style_collapse.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.evaluate.metrics import cl_dice, iou  # noqa: E402
from hfont.generate import (  # noqa: E402
    _stack_refs,
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
)
from hfont.intake import load_freehand_photo  # noqa: E402
from slant import load_font  # noqa: E402

HANDS = {
    "writer1": ROOT / "MyHandwriting.jpeg",
    "writer2": ROOT / "output/AntoineHand/AntoineHandwriting.jpg",
    "writer3": ROOT / "output/JadHand/JadHandWriting.jpeg",
}
FONTS = ("segoepr", "times", "Gabriola", "MISTRAL")


def stroke_px(images: dict[str, np.ndarray]) -> float:
    from scipy.ndimage import distance_transform_edt

    w = [float(np.mean(distance_transform_edt(m)[m]))
         for v in images.values() if (m := v > 0.5).sum() >= 20]
    return float(np.mean(w)) if w else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--out", default=str(ROOT / "output/review"))
    args = ap.parse_args()

    from skimage.morphology import dilation, disk, erosion

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)

    @torch.no_grad()
    def style_vector(refs: dict[str, np.ndarray]) -> np.ndarray:
        stack, mask = _stack_refs(list(refs.values()), model.device)
        return model.generator.encode_style(stack, mask).cpu().numpy()[0]

    styles = {n: load_freehand_photo(str(p), LATIN_CORE, "seed30", model.image_size)
              for n, p in HANDS.items()}
    styles.update({n: load_font(n) for n in FONTS})

    generated, vectors = {}, {}
    for name, refs in styles.items():
        images, _ = generate_rasters(model, refs, content)
        # Only the letters the model invented: the written ones pass through
        # unchanged and would make any two styles look different for free.
        generated[name] = {k: v for k, v in images.items() if k not in refs}
        vectors[name] = style_vector(refs)

    invented = set.intersection(*[set(g) for g in generated.values()])
    written = set.intersection(*[set(s) for s in styles.values()])

    rows = []
    for a, b in itertools.combinations(styles, 2):
        rows.append({
            "a": a, "b": b,
            "real_cl_dice": float(np.mean([cl_dice(styles[a][k], styles[b][k]) for k in written])),
            "gen_cl_dice": float(np.mean([cl_dice(generated[a][k], generated[b][k]) for k in invented])),
            "gen_iou": float(np.mean([iou(generated[a][k], generated[b][k]) for k in invented])),
            "style_cos": cosine(vectors[a], vectors[b]),
        })

    print(f"{len(invented)} invented letters, {len(written)} written letters\n")
    print(f"{'pair':24s} {'real':>7s} {'generated':>10s} {'style cos':>10s}")
    for r in rows:
        hand_pair = {r["a"], r["b"]} <= set(HANDS)
        print(f"{r['a'] + ' vs ' + r['b']:24s} {r['real_cl_dice']:7.3f} "
              f"{r['gen_cl_dice']:10.3f} {r['style_cos']:10.3f}"
              f"{'   <- two real people' if hand_pair else ''}")

    hands = [r for r in rows if {r["a"], r["b"]} <= set(HANDS)]
    fonts = [r for r in rows if not ({r["a"], r["b"]} & set(HANDS))]
    print(f"\nhand pairs (n={len(hands)}): generated similarity {np.mean([r['gen_cl_dice'] for r in hands]):.3f}, "
          f"style cos {np.mean([r['style_cos'] for r in hands]):.3f}")
    print(f"font pairs (n={len(fonts)}): generated similarity {np.mean([r['gen_cl_dice'] for r in fonts]):.3f}, "
          f"style cos {np.mean([r['style_cos'] for r in fonts]):.3f}")

    # --- control: is it just stroke weight? --------------------------------
    def mean_cos(group: dict[str, dict[str, np.ndarray]]) -> float:
        v = {n: style_vector(g) for n, g in group.items()}
        return float(np.mean([cosine(v[a], v[b]) for a, b in itertools.combinations(group, 2)]))

    hand_imgs = {n: styles[n] for n in HANDS}
    font_imgs = {n: styles[n] for n in FONTS}
    fatter = {n: {k: dilation(v, disk(1)) for k, v in g.items()} for n, g in hand_imgs.items()}
    thinner = {n: {k: erosion(v, disk(1)) for k, v in g.items()} for n, g in font_imgs.items()}
    thinner = {n: g for n, g in thinner.items() if stroke_px(g) > 0}

    control = [
        ("hands as intake gives them", hand_imgs),
        ("hands dilated toward font weight", fatter),
        ("fonts at native weight", font_imgs),
        ("fonts eroded toward hand weight", thinner),
    ]
    print(f"\n{'weight control':36s} {'stroke px':>9s} {'mean cos':>9s}")
    control_out = []
    for label, group in control:
        px, c = np.mean([stroke_px(g) for g in group.values()]), mean_cos(group)
        control_out.append({"condition": label, "stroke_px": float(px), "mean_cos": c})
        print(f"{label:36s} {px:9.2f} {c:9.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "style_collapse.json").write_text(
        json.dumps({"pairs": rows, "weight_control": control_out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
