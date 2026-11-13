"""Style numbers on SD19's validation writers: the score to choose models by.

The ten photographed writers are the held-out test. Choosing experiment after
experiment by their numbers would tune on them, so models are compared here
instead: SD19 writers in the cache's ``val`` split, never trained on, given
only the seed letters a user writes. For each writer the model draws every
other letter blind, and the drawings are scored against what the writer
really wrote, with the photographed writers' style numbers (grouped by ten so
chance stays at 10%; ``hfont.evaluate.style.grouped_style_report``):

- clDice of each blind letter against the real one;
- how alike the blind letters are across writers (real hands: ``real``);
- how often a blind letter is nearest its own writer's real letter.

The nearest font (``scripts/compose_eval.py``), resized to the writer's
proportions, is scored the same way as a reference: a model that personalises
should beat it on all three. The ``test`` split stays untouched for reports.

    python scripts/sd19_style.py --cache data/cache_sd19 --fonts data/cache_local \\
        --checkpoint runs/g3/best.pt
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

from compose_eval import CHAR, KEY, font_bank, nearest_font  # noqa: E402

from hfont.charset import LATIN_SEED_SETS  # noqa: E402
from hfont.data.dataset import GlyphStore  # noqa: E402
from hfont.evaluate.metrics import cl_dice  # noqa: E402
from hfont.evaluate.style import grouped_style_report  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
    match_seed_proportions,
)
from hfont.intake import IntakeConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache holding SD19 writers")
    ap.add_argument("--fonts", default=None, help="fonts for the nearest-font reference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--split", default="val", help="comma-separated splits")
    ap.add_argument("--retrieve", action="store_true",
                    help="content = nearest training hand's letters (content_source='nearest' models)")
    ap.add_argument("--writer-bank", action="store_true",
                    help="also score the nearest SD19 *training* writer as a reference")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=None, help="write the numbers here (JSON)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    seed = [c for c in LATIN_SEED_SETS["seed30"] if c != ","]  # SD19 has no comma
    store = GlyphStore(args.cache)
    writers = [r.font_id for r in store.records
               if r.category == "SD19" and r.split in args.split.split(",")]
    fonts = GlyphStore(args.fonts) if args.fonts else None
    bank = font_bank(fonts, seed) if fonts else None
    writer_bank = (font_bank(store, seed, keep=lambda r: r.category == "SD19" and r.split == "train")
                   if args.writer_bank else None)
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    retriever = None
    if args.retrieve:
        from hfont.data.neighbors import Bank

        retriever = Bank(store, [KEY[c] for c in seed])

    real, blind, font, writer = {}, {}, {}, {}
    for fid in writers:
        chars = {CHAR[k]: store.image(fid, k) for k in store.keys(fid) if k in CHAR}
        seeds = {c: chars[c] for c in seed if c in chars}
        if len(seeds) < len(seed) - 2:
            continue
        seed_images = {KEY[c]: v for c, v in seeds.items()}
        drawn, _ = generate_rasters(model, seed_images,
                                    retriever.content(seed_images, content) if retriever else content,
                                    match_size=True)
        held = [c for c in chars if c not in seeds and KEY[c] in drawn]
        real[fid] = {c: chars[c] for c in held}
        blind[fid] = {c: drawn[KEY[c]] for c in held}
        if bank:
            near = nearest_font(seeds, seed, bank)
            glyphs = {k: fonts.image(near, k) for k in fonts.keys(near)}
            glyphs, _ = match_seed_proportions(glyphs, {}, {KEY[c]: v for c, v in seeds.items()},
                                               CHAR, IntakeConfig().baseline)
            font[fid] = {c: glyphs[KEY[c]] for c in held if KEY[c] in glyphs}
        if writer_bank:
            near = nearest_font(seeds, seed, writer_bank)
            other = {CHAR[k]: store.image(near, k) for k in store.keys(near) if k in CHAR}
            other, _ = match_seed_proportions({KEY[c]: v for c, v in other.items()}, {},
                                              {KEY[c]: v for c, v in seeds.items()}, CHAR,
                                              IntakeConfig().baseline)
            writer[fid] = {c: other[KEY[c]] for c in held if KEY[c] in other}
        if len(real) >= args.limit:
            break

    def numbers(made: dict) -> dict:
        report = grouped_style_report(made, real)
        report["cl_dice"] = float(np.mean([cl_dice(made[w][c], real[w][c])
                                           for w in made for c in made[w]]))
        return report

    if len(real) < 10:
        raise SystemExit(f"only {len(real)} usable SD19 {args.split} writers; need at least 10")
    result = {"model": numbers(blind)}
    if font:
        result["nearest_font"] = numbers(font)
    if writer:
        result["nearest_writer"] = numbers(writer)
    r = result["model"]
    print(f"{len(real)} SD19 {args.split} writers, {r['groups']} groups of 10, letters not on the sheet")
    print(f"{'':14s} clDice   alike   attributed   (real hands alike {r['real_similarity']:.3f}, chance 10%)")
    for name, n in result.items():
        print(f"{name:14s} {n['cl_dice']:.3f}    {n['generated_similarity']:.3f}   {n['attribution']:.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
