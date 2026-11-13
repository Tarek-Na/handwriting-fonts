"""Composed letters vs the model vs the nearest font, on real writers' real letters.

The letters composition builds (u w d p q 9 W) are not on the photographed
sheets, so the ten writers cannot score it. SD19 writers wrote every letter,
and none of them is one of the ten, so they can: give each held-out SD19
writer only the seed letters a user writes, build the missing letters three
ways, and compare each with the letter that writer really wrote.

- **composed**: hfont.compose, the writer's own strokes turned or mirrored;
- **model**: the checkpoint's drawing, at the writer's proportions (as in the
  font);
- **nearest font**: the corpus font whose seed letters look most like the
  writer's, its letter resized to the writer's proportions. A floor any
  generator should clear.

Scored with clDice (where the strokes run, not how heavy) on the SD19 writers
in the cache's val and test splits (scripts/sd19_cache.py); the model has
never trained on them.

    python scripts/compose_eval.py --cache data/cache_sd19 --fonts data/cache_local
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

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.compose import DONORS, compose  # noqa: E402
from hfont.data.dataset import GlyphStore  # noqa: E402
from hfont.evaluate.metrics import cl_dice  # noqa: E402
from hfont.generate import (  # noqa: E402
    content_images_from_font,
    generate_rasters,
    load_checkpoint,
    match_seed_proportions,
)
from hfont.intake import IntakeConfig  # noqa: E402

KEY = {g.char: g.key for g in LATIN_CORE}
CHAR = {g.key: g.char for g in LATIN_CORE}


def _signature(image: np.ndarray) -> np.ndarray:
    """Blurred 32x32 thumbnail, unit length: cheap shape similarity for retrieval."""
    from scipy.ndimage import gaussian_filter

    small = gaussian_filter(image.astype(np.float32), 2.0)[2::4, 2::4]
    return (small / (np.linalg.norm(small) + 1e-6)).ravel()


def font_bank(store: GlyphStore, seed: list[str], keep=None):
    """Signatures of every font's seed letters; ``keep(record)`` narrows the fonts."""
    ids = [r.font_id for r in store.records if keep is None or keep(r)]
    sigs = np.zeros((len(ids), len(seed), 32 * 32), np.float32)
    have = np.zeros((len(ids), len(seed)), bool)
    for i, fid in enumerate(ids):
        for j, ch in enumerate(seed):
            if store.has(fid, KEY[ch]):
                sigs[i, j] = _signature(store.image(fid, KEY[ch]))
                have[i, j] = True
    return ids, sigs, have


def nearest_font(seeds: dict[str, np.ndarray], seed: list[str], bank) -> str:
    ids, sigs, have = bank
    cols = [j for j, ch in enumerate(seed) if ch in seeds]
    mine = np.stack([_signature(seeds[seed[j]]) for j in cols])
    sim = (sigs[:, cols] * mine[None]).sum(-1)
    sim = np.where(have[:, cols], sim, 0.0).sum(1) / np.maximum(have[:, cols].sum(1), 1)
    return ids[int(np.argmax(sim))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache holding SD19 writers")
    ap.add_argument("--fonts", default=str(ROOT / "data/cache_local"), help="fonts to retrieve from")
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None, help="write per-writer scores here (JSON)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    seed = [c for c in LATIN_SEED_SETS["seed30"] if c != ","]  # SD19 has no comma
    store = GlyphStore(args.cache)
    writers = [r.font_id for r in store.records
               if r.category == "SD19" and r.split in args.splits.split(",")]
    fonts = GlyphStore(args.fonts)
    bank = font_bank(fonts, seed)
    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    line = model.image_size * IntakeConfig().baseline

    scores: dict[str, dict[str, list[float]]] = {t: {"composed": [], "model": [], "font": []}
                                                  for t in DONORS}
    rows = []
    used = 0
    floor = {"model": [], "font": []}   # every letter the writer did not put on the sheet
    for fid in writers:
        chars = {CHAR[k]: store.image(fid, k) for k in store.keys(fid) if k in CHAR}
        seeds = {c: chars[c] for c in seed if c in chars}
        targets = [t for t, (donor, _) in DONORS.items() if t in chars and donor in seeds]
        if len(seeds) < len(seed) - 2 or not targets:
            continue
        made = compose(seeds, line, set(targets))
        drawn, _ = generate_rasters(model, {KEY[c]: v for c, v in seeds.items()}, content,
                                    match_size=True)
        near = nearest_font(seeds, seed, bank)
        font_glyphs = {k: fonts.image(near, k) for k in fonts.keys(near)}
        font_glyphs, _ = match_seed_proportions(
            font_glyphs, {}, {KEY[c]: v for c, v in seeds.items()}, CHAR, IntakeConfig().baseline)
        row = {"writer": fid, "font": near}
        for t in targets:
            if t not in made:
                continue
            real = chars[t]
            s = {"composed": cl_dice(made[t], real), "model": cl_dice(drawn[KEY[t]], real),
                 "font": cl_dice(font_glyphs[KEY[t]], real) if KEY[t] in font_glyphs else np.nan}
            for name, v in s.items():
                scores[t][name].append(v)
            row[t] = s
        for c, real in chars.items():
            if c in seeds or KEY[c] not in drawn or KEY[c] not in font_glyphs:
                continue
            floor["model"].append(cl_dice(drawn[KEY[c]], real))
            floor["font"].append(cl_dice(font_glyphs[KEY[c]], real))
        rows.append(row)
        used += 1
        if used >= args.limit:
            break

    print(f"{used} SD19 writers ({args.splits}); clDice against the letter they really wrote\n")
    print("letter  from      n   composed   model   nearest font")
    total = {"composed": [], "model": [], "font": []}
    for t, (donor, op) in DONORS.items():
        s = scores[t]
        if not s["composed"]:
            continue
        for name in total:
            total[name] += s[name]
        wins = np.mean(np.array(s["composed"]) > np.array(s["model"]))
        print(f"  {t}     {donor} {op:6s} {len(s['composed']):4d}   {np.mean(s['composed']):.3f}"
              f"     {np.mean(s['model']):.3f}   {np.nanmean(s['font']):.3f}"
              f"   (composed better for {wins:.0%})")
    print(f"  all              {len(total['composed']):4d}   {np.mean(total['composed']):.3f}"
          f"     {np.mean(total['model']):.3f}   {np.nanmean(total['font']):.3f}")
    m, f = np.array(floor["model"]), np.array(floor["font"])
    print(f"\nevery letter not on the sheet ({m.size} letters): model {m.mean():.3f}, "
          f"nearest font {f.mean():.3f}; model closer to the real letter for {np.mean(m > f):.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
