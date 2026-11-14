"""Score a checkpoint on every photographed writer, and show what it draws.

For each handwriting photo it finds, this runs freehand intake, then
leave-one-out (hide each written letter, draw it from the other 29, compare),
and writes one sheet per writer with three rows: what they wrote, the model's
blind attempt at each of those letters, and the letters they never wrote as
they go into the font. The sheets are what showed that the shipped model draws
typeset letters for every hand; run the same script on a new checkpoint to see
whether that changed.

It also reports three style numbers across all writers (hfont.evaluate.style):
how alike the model's blind attempts are across writers, how alike the real
letters are (the target), and how often a blind attempt is nearest its own
writer's real letter (chance is one in the number of writers).

Photos are found by pattern, so no one's name is written into the code:
`*Hand*.jp*g` at the project root and `output/*Hand/*Hand*.jp*g`. The photos and the
sheets are private and ignored by git.

    python scripts/eval_writers.py --checkpoint runs/hw40k/best.pt --tag _hw40k
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.evaluate.style import style_report  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
    match_pen,
)
from hfont.intake import load_freehand_photo  # noqa: E402
from how_good import leave_one_out  # noqa: E402

PATTERNS = ("*Hand*.jp*g", "output/*Hand/*Hand*.jp*g")


def find_photos(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            name = path.stem
            for suffix in ("Handwriting", "HandWriting", "Hand"):
                name = name.removesuffix(suffix)
            found.setdefault(name or path.stem, path)
    return found


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:  # not on Windows
        return ImageFont.load_default(size)


def sheet(name, refs, blind, final, order, novel, key) -> np.ndarray:
    from PIL import Image, ImageDraw

    def tile(image):
        t = Image.fromarray(((1 - image) * 255).astype(np.uint8)).resize((64, 64), Image.LANCZOS)
        return np.pad(np.asarray(t), 1, constant_values=150)

    def strip(images, per=15):
        rows = []
        for i in range(0, len(images), per):
            chunk = [tile(x) for x in images[i : i + per]]
            chunk += [np.full((66, 66), 255, np.uint8)] * (per - len(chunk))
            rows.append(np.concatenate(chunk, 1))
        return np.concatenate(rows, 0)

    written = strip([refs[key[c]] for c in order])
    drawn = strip([blind[key[c]]["gen"] for c in order])
    never = strip([final[key[c]] for c in novel])
    font = _font(22)

    def label(text):
        im = Image.new("L", (written.shape[1], 30), 255)
        ImageDraw.Draw(im).text((4, 3), text, fill=60, font=font)
        return np.asarray(im)

    return np.concatenate([
        label(f"{name}: what they wrote (the model's input)"), written,
        label("the model's blind attempt at each letter (drawn from the other 29)"), drawn,
        label("letters they never wrote, as they go into the font"), never,
    ], 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--out", default=str(ROOT / "output/writers"))
    ap.add_argument("--tag", default="", help="suffix for this checkpoint's files")
    ap.add_argument("--pen", action="store_true",
                    help="redraw generated letters with the writer's pen weight (generate.match_pen)")
    ap.add_argument("--retrieve-cache", default=None,
                    help="content = nearest training hand in this cache (content_source='nearest')")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from PIL import Image

    photos = find_photos(ROOT)
    if not photos:
        print("no handwriting photos found (patterns: " + ", ".join(PATTERNS) + ")")
        return 1
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    order = LATIN_SEED_SETS["seed30"]
    key = {g.char: g.key for g in LATIN_CORE}
    bank = content_for = None
    if args.retrieve_cache:
        from hfont.data.dataset import GlyphStore
        from hfont.data.neighbors import Bank

        bank = Bank(GlyphStore(args.retrieve_cache), [key[c] for c in order])

        def content_for(held, others):
            return bank.content(others, {held: content[held]})[held]
    novel = [g.char for g in LATIN_CORE if g.char not in order]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    blinds, reals = {}, {}
    for name, photo in photos.items():
        try:
            refs = load_freehand_photo(str(photo), LATIN_CORE, "seed30", model.image_size)
        except Exception as error:  # a sheet intake cannot read is reported, not fatal
            print(f"{name:10s} intake failed: {error}")
            continue
        post = (lambda gen, others: match_pen({"x": gen}, others)["x"]) if args.pen else None
        blind = leave_one_out(model, refs, content, content_for, post)
        blinds[name] = {k: r["gen"] for k, r in blind.items()}
        reals[name] = refs
        score = {m: float(np.mean([r[m] for r in blind.values()])) for m in ("cl_dice", "iou", "tol_f1")}
        summary[name] = {**score, "per_char": {c: float(blind[key[c]]["cl_dice"]) for c in order}}
        final, _ = generate_rasters(model, refs, bank.content(refs, content) if bank else content,
                                    match_weight=not args.pen, pen=args.pen,
                                    replace_broken=True, match_size=True)
        Image.fromarray(sheet(name, refs, blind, final, order, novel, key)).save(
            out / f"model_{name}{args.tag}.png")
        print(f"{name:10s} leave-one-out clDice {score['cl_dice']:.3f}  IoU {score['iou']:.3f}"
              f"  tol-F1 {score['tol_f1']:.3f}")

    if summary:
        mean = float(np.mean([s["cl_dice"] for s in summary.values()]))
        print(f"{'mean':10s} leave-one-out clDice {mean:.3f} over {len(summary)} writers")
    if len(blinds) > 1:
        style = style_report(blinds, reals)
        summary["_style"] = style
        print(f"style: blind attempts alike across writers {style['generated_similarity']:.3f}"
              f" (real letters {style['real_similarity']:.3f});"
              f" attributed to their own writer {style['attribution']:.1%}"
              f" (chance {style['attribution_chance']:.1%})")
    (out / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"sheets and summary in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
