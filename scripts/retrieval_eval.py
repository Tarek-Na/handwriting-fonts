"""How well do a writer's seed letters find a training writer who writes like them?

``sd19_style.py --oracle`` showed that for most held-out SD19 writers some
training writer's unseen letters match theirs nearly as well as their own
second copy would (attribution 36% against a 38% ceiling), but retrieval by
seed-letter thumbnails finds that writer rarely (18.6%). This compares ways
of embedding the seed letters, by the attribution their retrieved writer's
unseen letters reach, and by how far down the list the oracle's pick lands:

- ``thumb``: blurred 16x16 thumbnails (hfont.data.neighbors);
- ``style``: a checkpoint's pooled style code (``encode_style``), cosine.

    python scripts/retrieval_eval.py --cache data/cache_sd19 --checkpoint runs/g3/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS  # noqa: E402
from hfont.data.dataset import GlyphStore  # noqa: E402
from hfont.data.neighbors import hand_signatures  # noqa: E402
from hfont.evaluate.style import grouped_style_report  # noqa: E402

KEY = {g.char: g.key for g in LATIN_CORE}


def style_codes(model, hands: list[dict[str, np.ndarray]], keys: list[str]) -> np.ndarray:
    import torch

    from hfont.generate import _stack_refs

    out = []
    with torch.no_grad():
        for hand in hands:
            refs, mask = _stack_refs([hand[k] for k in keys if k in hand], model.device)
            code = model.generator.encode_style(refs, mask).float().cpu().numpy()[0]
            out.append(code / (np.linalg.norm(code) + 1e-8))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default=None, help="score only these characters")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    store = GlyphStore(args.cache)
    seed = [KEY[c] for c in LATIN_SEED_SETS["seed30"] if c in KEY and c != ","]
    sd19 = [r for r in store.records if r.category == "SD19"]
    bank = [r.font_id for r in sd19 if r.split == "train"]
    query = [r.font_id for r in sd19 if r.split in args.split.split(",")]
    unseen = sorted({k for f in query for k in store.keys(f)} - set(seed))
    if args.only:
        unseen = [k for k in unseen if any(KEY[c] == k for c in args.only)]

    def hands(ids, keys):
        return [{k: store.image(f, k) for k in keys if store.has(f, k)} for f in ids]

    def nearest(q, b):
        return np.argmax(q @ b.T, axis=1)

    seeds_q, seeds_b = hands(query, seed), hands(bank, seed)
    sq, hq = hand_signatures(seeds_q, seed)
    sb, hb = hand_signatures(seeds_b, seed)
    shared = hq @ hb.T
    thumb = (sq.reshape(len(sq), -1) @ sb.reshape(len(sb), -1).T) / np.maximum(shared, 1)
    picks = {"thumb": np.argmax(thumb, axis=1)}

    uq, hu = hand_signatures(hands(query, unseen), unseen)
    ub, hub = hand_signatures(hands(bank, unseen), unseen)
    oracle_sim = (uq.reshape(len(uq), -1) @ ub.reshape(len(ub), -1).T) / np.maximum(hu @ hub.T, 1)
    picks["oracle"] = np.argmax(oracle_sim, axis=1)
    if args.checkpoint:
        from hfont.generate import load_checkpoint

        model = load_checkpoint(args.checkpoint)
        picks["style"] = nearest(style_codes(model, seeds_q, seed), style_codes(model, seeds_b, seed))

    real = {f: {k: store.image(f, k) for k in unseen if store.has(f, k)} for f in query}
    order = np.argsort(-oracle_sim, axis=1)
    print(f"{len(query)} SD19 {args.split} writers, bank of {len(bank)} training writers, "
          f"{len(unseen)} unseen characters")
    for name, pick in picks.items():
        attempt = {f: {k: store.image(bank[j], k) for k in real[f] if store.has(bank[j], k)}
                   for f, j in zip(query, pick)}
        r = grouped_style_report(attempt, real)
        rank = [int(np.where(order[i] == j)[0][0]) for i, j in enumerate(pick)]
        print(f"{name:7s} attributed {r['attribution']:.1%}  alike {r['generated_similarity']:.3f}  "
              f"oracle-rank of pick: median {int(np.median(rank))} of {len(bank)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
