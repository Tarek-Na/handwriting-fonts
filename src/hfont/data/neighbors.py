"""Which training hands look most like a given hand, from its seed letters.

The model ignores its content image (instruction.md §4): during training the
content is the target letter in a random other font, which says which letter
to draw and nothing about its shape, so the model learned to take only that.
Retrieve-and-refine gives the content path something worth reading: the target
letter as written by the training hand (font or SD19 writer) whose seed
letters look most like the writer's. Measured on held-out SD19 writers, that
neighbour's letters alone are attributed to the right writer more often than
the model's (``scripts/sd19_style.py --writer-bank``).

Similarity is the mean over seed letters of the cosine between blurred 16x16
thumbnails, letters missing from either hand left out. The table is built
once per cache (``python -m hfont.data.neighbors <cache>``) and stored as
``neighbors.json`` beside ``meta.json``. Candidates are training hands only,
so a held-out writer can never be anyone's neighbour, and no hand is its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

FILE = "neighbors.json"
SIDE = 16


def signature(image: np.ndarray) -> np.ndarray:
    """Blurred thumbnail of a glyph, unit length: cheap shape similarity."""
    from scipy.ndimage import gaussian_filter

    im = gaussian_filter(np.asarray(image, np.float32), 3.0)
    h, w = im.shape
    small = im[: h - h % SIDE, : w - w % SIDE].reshape(SIDE, h // SIDE, SIDE, w // SIDE).mean((1, 3))
    return (small / (np.linalg.norm(small) + 1e-6)).ravel()


def hand_signatures(images: Iterable[dict[str, np.ndarray]], keys: list[str]):
    """(N, letters, SIDE*SIDE) signatures and an (N, letters) mask of letters present."""
    images = list(images)
    sigs = np.zeros((len(images), len(keys), SIDE * SIDE), np.float32)
    have = np.zeros((len(images), len(keys)), np.float32)
    for i, hand in enumerate(images):
        for j, key in enumerate(keys):
            if key in hand:
                sigs[i, j] = signature(hand[key])
                have[i, j] = 1.0
    return sigs, have


def nearest(query_sigs, query_have, bank_sigs, bank_have, k: int, exclude=None) -> np.ndarray:
    """Indices of the ``k`` most similar bank hands for each query, best first.

    ``exclude[i]`` is a bank index query ``i`` may not pick (itself), or -1.
    """
    q = query_sigs.reshape(len(query_sigs), -1)
    b = bank_sigs.reshape(len(bank_sigs), -1)
    out = []
    for start in range(0, len(q), 256):
        sim = q[start:start + 256] @ b.T
        shared = query_have[start:start + 256] @ bank_have.T
        sim = np.where(shared > 0, sim / np.maximum(shared, 1.0), -np.inf)
        if exclude is not None:
            rows = np.arange(sim.shape[0])
            ex = exclude[start:start + 256]
            sim[rows[ex >= 0], ex[ex >= 0]] = -np.inf
        top = np.argsort(-sim, axis=1)[:, :k]
        out.append(top)
    return np.concatenate(out) if out else np.zeros((0, k), int)


def build(store, seed_keys: list[str], k: int = 8,
          candidate: Callable = lambda r: r.split == "train") -> dict[str, list[str]]:
    """For every hand in ``store``, its ``k`` nearest candidate hands, best first."""
    records = list(store.records)
    hands = [{key: store.image(r.font_id, key) for key in seed_keys if store.has(r.font_id, key)}
             for r in records]
    sigs, have = hand_signatures(hands, seed_keys)
    bank = [i for i, r in enumerate(records) if candidate(r)]
    position = {i: n for n, i in enumerate(bank)}
    exclude = np.array([position.get(i, -1) for i in range(len(records))])
    top = nearest(sigs, have, sigs[bank], have[bank], k, exclude)
    return {r.font_id: [records[bank[j]].font_id for j in row] for r, row in zip(records, top)}


class Bank:
    """Training hands to retrieve from at inference, for a writer never seen."""

    def __init__(self, store, seed_keys: list[str],
                 candidate: Callable = lambda r: r.split == "train") -> None:
        self.store = store
        self.seed_keys = seed_keys
        self.ids = [r.font_id for r in store.records if candidate(r)]
        hands = [{k: store.image(f, k) for k in seed_keys if store.has(f, k)} for f in self.ids]
        self.sigs, self.have = hand_signatures(hands, seed_keys)

    def nearest(self, seeds: dict[str, np.ndarray], k: int = 8) -> list[str]:
        sigs, have = hand_signatures([seeds], self.seed_keys)
        return [self.ids[j] for j in nearest(sigs, have, self.sigs, self.have, k)[0]]

    def content(self, seeds: dict[str, np.ndarray], fallback: dict[str, np.ndarray],
                k: int = 8) -> dict[str, np.ndarray]:
        """Each letter from the nearest hand that has it; ``fallback`` where none does."""
        near = self.nearest(seeds, k)
        out = {}
        for key, image in fallback.items():
            hand = next((f for f in near if self.store.has(f, key)), None)
            out[key] = self.store.image(hand, key) if hand else image
        return out


def load(cache_dir: str | Path) -> dict[str, list[str]] | None:
    path = Path(cache_dir) / FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    import argparse

    from ..charset import LATIN_CORE, LATIN_SEED_SETS
    from .dataset import GlyphStore

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    key = {g.char: g.key for g in LATIN_CORE}
    seed = [key[c] for c in LATIN_SEED_SETS["seed30"] if c in key]
    store = GlyphStore(args.cache)
    table = build(store, seed, k=args.k)
    (Path(args.cache) / FILE).write_text(json.dumps(table), encoding="utf-8")
    log.info("neighbours for %d hands written to %s", len(table), Path(args.cache) / FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
