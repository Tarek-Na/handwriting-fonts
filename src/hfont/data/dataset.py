"""Torch dataset over the rendered glyph cache.

The sampling scheme is the part that does the real work here, so it is worth
stating explicitly. One training example is:

    target      glyph c drawn in font f          <- what the model must produce
    style refs  K glyphs of f, none of them c    <- "write in this hand"
    content     glyph c drawn in some other font <- "write this letter"

Two exclusions carry most of the weight.

*The target glyph is never among the style references.* If it were, the task
collapses into copying one of the inputs, the loss drops immediately, and the
model never learns to transfer anything. This is the single easiest way to get
a result that looks excellent in training and is worthless at inference, since
at inference the user by definition has not written the letter being generated.

*The content image comes from a different font than the style.* Taking it from
the target font would leak the answer's stroke weight and proportions through
the content path. Drawing it from a random other font each time forces the
content encoder toward what is invariant across fonts — which letter it is —
and leaves everything else to the style path.

K is randomized during training. The user supplies ~24 references at inference
but the model is trained with 1-8, so the style pooling has to be genuinely
permutation-invariant and size-agnostic rather than tuned to one K.
"""

from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..charset import Charset, get_charset, seed_charset

log = logging.getLogger(__name__)


@dataclass
class _FontData:
    """One font's decompressed cache entry."""

    images: np.ndarray
    index: dict[str, int]
    advances: np.ndarray
    hashes: list[str]


@dataclass
class FontRecord:
    font_id: str
    file: str
    family: str
    style: str
    category: str
    split: str
    n_glyphs: int
    scale: float
    baseline_y: float
    units_per_em: int


class GlyphStore:
    """Random access to the cached glyph rasters, with an LRU of open fonts.

    Fonts are the unit of caching because every sample touches many glyphs of
    one font and nothing of any other. Decompressing a whole font's ``.npz``
    (~100 glyphs, a few hundred KB) to serve one sample would be wasteful; with
    an LRU it is amortized to nearly nothing.
    """

    def __init__(self, cache_dir: str | Path, lru_size: int = 256) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no glyph cache at {self.cache_dir} (expected meta.json); "
                "run `hfont prepare` first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.charset: Charset = get_charset(meta["charset"])
        self.render_config: dict = meta["render"]
        self.image_size: int = int(self.render_config["size"])
        self.records: list[FontRecord] = [FontRecord(**r) for r in meta["fonts"]]
        self._by_id = {r.font_id: r for r in self.records}
        self._lru: OrderedDict[str, _FontData] = OrderedDict()
        self._lru_size = lru_size

        # Prefer the packed, memory-mapped layout when it exists. See
        # prepare.pack_cache for why: on a large corpus the per-font files
        # turn every sample into two decompressions.
        self._packed: dict[str, _FontData] | None = None
        packed_path = self.cache_dir / "packed_images.npy"
        index_path = self.cache_dir / "packed_index.npz"
        if packed_path.is_file() and index_path.is_file():
            self._packed = self._open_packed(packed_path, index_path)

    @staticmethod
    def _open_packed(packed_path: Path, index_path: Path) -> dict[str, "_FontData"]:
        images = np.load(packed_path, mmap_mode="r")
        with np.load(index_path) as idx:
            font_ids = [str(f) for f in idx["font_ids"]]
            starts, counts = idx["starts"], idx["counts"]
            keys = [str(k) for k in idx["keys"]]
            advances = idx["advances"]
            hashes = [str(h) for h in idx["hashes"]]
        fonts: dict[str, _FontData] = {}
        for font_id, start, count in zip(font_ids, starts, counts):
            lo, hi = int(start), int(start + count)
            fonts[font_id] = _FontData(
                images=images[lo:hi],  # zero-copy memmap view
                index={k: i for i, k in enumerate(keys[lo:hi])},
                advances=advances[lo:hi],
                hashes=hashes[lo:hi],
            )
        log.info("opened packed cache: %d fonts, %d glyphs", len(fonts), len(keys))
        return fonts

    def __len__(self) -> int:
        return len(self.records)

    def split(self, split: str) -> list[FontRecord]:
        return [r for r in self.records if r.split == split]

    def _load(self, font_id: str) -> "_FontData":
        if self._packed is not None:
            return self._packed[font_id]
        cached = self._lru.get(font_id)
        if cached is not None:
            self._lru.move_to_end(font_id)
            return cached
        record = self._by_id[font_id]
        with np.load(self.cache_dir / "fonts" / record.file) as data:
            images = data["images"]
            keys = [str(k) for k in data["keys"]]
            advances = data["advances"]
            hashes = (
                [str(h) for h in data["hashes"]] if "hashes" in data else [""] * len(keys)
            )
        entry = _FontData(
            images=images,
            index={k: i for i, k in enumerate(keys)},
            advances=advances,
            hashes=hashes,
        )
        self._lru[font_id] = entry
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return entry

    def keys(self, font_id: str) -> list[str]:
        return list(self._load(font_id).index)

    def has(self, font_id: str, key: str) -> bool:
        return key in self._load(font_id).index

    def image(self, font_id: str, key: str) -> np.ndarray:
        """Glyph raster as float32 in [0, 1]."""
        data = self._load(font_id)
        return data.images[data.index[key]].astype(np.float32) / 255.0

    def advance(self, font_id: str, key: str) -> float:
        data = self._load(font_id)
        return float(data.advances[data.index[key]])

    def pixel_hash(self, font_id: str, key: str) -> str:
        data = self._load(font_id)
        return data.hashes[data.index[key]]

    def distinct_keys(self, font_id: str, exclude_key: str) -> list[str]:
        """Keys whose raster differs from ``exclude_key``'s.

        Used to build style-reference pools: excluding by character name alone
        is not enough when two characters share an outline.
        """
        data = self._load(font_id)
        target_hash = data.hashes[data.index[exclude_key]]
        if not target_hash:
            return [k for k in data.index if k != exclude_key]
        return [
            k for k, i in data.index.items()
            if k != exclude_key and data.hashes[i] != target_hash
        ]

    def pick_content_font(self, split: str = "train") -> str:
        """Choose a neutral font to draw content images from.

        The content image is a *skeleton*, not a style cue, so a display face
        with heavy decoration makes the content encoder's job needlessly hard.
        Prefer a plain sans by name, then anything categorized as sans.

        Family names are matched exactly and the style must be exactly
        "Regular". Prefix matching picked Roboto *Condensed* on the full corpus
        once plain Roboto landed in the validation split — a narrow skeleton
        that would bias every generated width. And "Regular" cannot be assumed
        from the family: a variable font renders at its default instance, which
        is frequently a light weight (Source Sans 3 comes out ExtraLight).
        """
        records = self.split(split) or self.records
        preferred = (
            "noto sans", "open sans", "roboto", "inter", "ibm plex sans",
            "work sans", "source sans 3", "lato", "pt sans", "dejavu sans",
            "liberation sans", "arial",
        )
        by_family: dict[str, list[FontRecord]] = {}
        for record in records:
            by_family.setdefault(record.family.lower(), []).append(record)
        for want in preferred:
            for record in sorted(by_family.get(want, []), key=lambda r: r.font_id):
                if record.style == "Regular":
                    return record.font_id
        for record in sorted(records, key=lambda r: r.font_id):
            if record.category == "SANS_SERIF":
                return record.font_id
        return sorted(r.font_id for r in records)[0]


def component_weight_map(
    image: np.ndarray, balance: float, max_boost: float = 8.0, band: int = 2
) -> np.ndarray:
    """Per-pixel loss weight that gives every separate piece of a glyph a say.

    Each connected component is weighted up until its mass is at least
    ``balance`` times that of the glyph's largest component (never down, and at
    most ``max_boost``). The weight is spread over a ``band``-pixel margin, so a
    dot drawn off target is penalized as well as a dot left out.

    Why: the first trained model drew the body of ``i`` and ``j`` in the right
    hand and omitted the dot entirely, in every font. The mechanism is that L1
    is median-seeking. No seed letter shows where this writer puts a dot, so
    each candidate pixel is ink in only a minority of plausible outcomes, and
    the per-pixel median — what L1 converges to — is blank. Weight that rises
    where the *target* has ink lowers the share of outcomes needed before the
    optimum becomes "draw it". (The adversarial terms attack the same failure
    more directly: a discriminator knows a dotless ``i`` is not an ``i``.)

    The rule is relative to the glyph's own largest part rather than an
    absolute pixel count, because glyph size varies by font: a fixed floor
    boosted a thin handwritten dot 7x and Arial's heavier one under 2x.
    """
    from scipy.ndimage import grey_dilation
    from skimage.measure import label

    labels = label(image > 0.5, connectivity=2)
    if labels.max() <= 1:
        return np.ones_like(image, dtype=np.float32)  # nothing detached to balance
    areas = np.bincount(labels.ravel()).astype(np.float64)
    areas[0] = 0.0
    target = balance * areas.max()
    boost = np.clip(target / np.maximum(areas, 1.0), 1.0, max_boost)
    boost[0] = 1.0  # background
    weight = boost[labels].astype(np.float32)
    if band > 0:
        weight = grey_dilation(weight, size=(2 * band + 1, 2 * band + 1))
    return weight


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W) in [0,1] ink-on-white -> (1, H, W) in [-1, 1], ink negative.

    Signed range because the generator ends in a tanh, and 'no ink' being the
    saturating end of that tanh means blank background is easy to represent
    exactly rather than only approached.
    """
    return torch.from_numpy(img).unsqueeze(0) * 2.0 - 1.0


@dataclass
class DatasetConfig:
    split: str = "train"
    #: Number of style reference glyphs per sample. A tuple means "sample
    #: uniformly in this inclusive range each time".
    n_refs: tuple[int, int] = (1, 8)
    #: Draw the content image from a random other font ("random") or always
    #: from the same one ("fixed"). Random is the disentangling choice.
    content_source: str = "random"
    #: Restrict style references to a seed set, mimicking inference exactly.
    #: None means "any glyph of the font".
    ref_seed: str | None = None
    #: Oversampling weight for HANDWRITING-category fonts, which are the slice
    #: closest to the real inference distribution but a minority of the corpus.
    handwriting_weight: float = 3.0
    #: Length of one epoch in samples. The true space is fonts x glyphs, which
    #: is large, so an "epoch" is just a checkpoint interval.
    epoch_size: int = 20000
    #: Raise each detached part of a target glyph to at least this fraction of
    #: its largest part's loss mass; 0 disables. See component_weight_map.
    component_balance: float = 0.0
    component_max_boost: float = 8.0


class GlyphPairDataset(Dataset):
    """Yields (content, style refs, target) triples for the generator."""

    def __init__(
        self,
        store: GlyphStore,
        config: DatasetConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.config = config or DatasetConfig()
        self.charset = store.charset
        self.records = store.split(self.config.split)
        if not self.records:
            raise ValueError(f"no fonts in split {self.config.split!r}")

        self.font_ids = [r.font_id for r in self.records]
        self.font_index = {fid: i for i, fid in enumerate(self.font_ids)}
        self.glyph_index = {g.key: i for i, g in enumerate(self.charset)}

        # Sampling weights: upweight handwriting-category fonts.
        weights = np.asarray(
            [
                self.config.handwriting_weight if r.category == "HANDWRITING" else 1.0
                for r in self.records
            ],
            dtype=np.float64,
        )
        self._font_p = weights / weights.sum()
        self._font_cdf = np.cumsum(self._font_p)

        self._ref_keys: set[str] | None = None
        if self.config.ref_seed:
            sub = seed_charset(self.charset, self.config.ref_seed)
            self._ref_keys = {g.key for g in sub}

        self._base_seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.config.epoch_size

    def _pick_font(self, rng: random.Random) -> str:
        i = int(np.searchsorted(self._font_cdf, rng.random(), side="right"))
        return self.font_ids[min(i, len(self.font_ids) - 1)]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Per-item RNG so that results are reproducible regardless of how many
        # DataLoader workers are running or in what order they finish.
        rng = random.Random((self._base_seed * 1_000_003) ^ idx)
        cfg = self.config

        style_font = self._pick_font(rng)
        available = self.store.keys(style_font)
        if len(available) < 2:
            # Degenerate font; fall back to any other rather than failing a batch.
            style_font = self._pick_font(rng)
            available = self.store.keys(style_font)

        target_key = rng.choice(available)

        # Excludes both the target key and anything that renders identically to
        # it, so a reference can never be a copy of the answer.
        ref_pool = self.store.distinct_keys(style_font, target_key)
        if self._ref_keys is not None:
            restricted = [k for k in ref_pool if k in self._ref_keys]
            if len(restricted) >= 1:
                ref_pool = restricted
        if not ref_pool:
            ref_pool = [k for k in available if k != target_key]

        lo, hi = cfg.n_refs
        n_refs = rng.randint(lo, min(hi, len(ref_pool)))
        ref_keys = rng.sample(ref_pool, n_refs)

        # Content image: same character, different font.
        if cfg.content_source == "fixed":
            content_font = self.font_ids[0]
        else:
            content_font = style_font
            for _ in range(8):
                candidate = self._pick_font(rng)
                if candidate != style_font and self.store.has(candidate, target_key):
                    content_font = candidate
                    break

        target = self.store.image(style_font, target_key)
        content = self.store.image(content_font, target_key)
        refs = np.stack([self.store.image(style_font, k) for k in ref_keys])

        # Pad the reference stack to the configured maximum so that samples
        # collate into a batch; a mask tells the encoder what is real.
        max_refs = cfg.n_refs[1]
        ref_mask = torch.zeros(max_refs, dtype=torch.bool)
        ref_mask[:n_refs] = True
        padded = np.zeros((max_refs, *refs.shape[1:]), dtype=np.float32)
        padded[:n_refs] = refs

        sample = {
            "content": _to_tensor(content),
            "refs": torch.from_numpy(padded).unsqueeze(1) * 2.0 - 1.0,
            "ref_mask": ref_mask,
            "target": _to_tensor(target),
            "char_id": torch.tensor(self.glyph_index[target_key], dtype=torch.long),
            "font_id": torch.tensor(self.font_index[style_font], dtype=torch.long),
            "advance": torch.tensor(
                self.store.advance(style_font, target_key), dtype=torch.float32
            ),
        }
        if cfg.component_balance > 0:
            weight = component_weight_map(target, cfg.component_balance, cfg.component_max_boost)
            sample["target_weight"] = torch.from_numpy(weight).unsqueeze(0)
        return sample


class FontEvalDataset(Dataset):
    """Deterministic held-out evaluation: every glyph of every font, fixed refs.

    Unlike the training sampler this is fully reproducible — same references,
    same content font, same order, every run — so that two checkpoints can be
    compared without sampling noise swamping the difference.
    """

    def __init__(
        self,
        store: GlyphStore,
        split: str = "val",
        ref_seed: str = "seed24",
        max_fonts: int | None = 200,
        content_font: str | None = None,
    ) -> None:
        self.store = store
        self.charset = store.charset
        records = store.split(split)
        if max_fonts is not None:
            # Deterministic subsample, stratified by taking a regular stride so
            # the category mix is roughly preserved.
            if len(records) > max_fonts:
                stride = len(records) / max_fonts
                records = [records[int(i * stride)] for i in range(max_fonts)]
        self.records = records

        seed_set = seed_charset(self.charset, ref_seed)
        self._ref_keys = [g.key for g in seed_set]
        self._max_refs = len(self._ref_keys)
        self.glyph_index = {g.key: i for i, g in enumerate(self.charset)}

        # The content font must not be in the evaluated split.
        self.content_font = content_font or store.pick_content_font("train")

        # Reference keys are fixed per font, so resolve them once rather than
        # per item; this dataset is iterated many times.
        self._refs_for: dict[str, list[str]] = {}
        self._items: list[tuple[str, str]] = []
        content_keys = set(self.store.keys(self.content_font))
        for record in self.records:
            keys = set(self.store.keys(record.font_id))
            self._refs_for[record.font_id] = [k for k in self._ref_keys if k in keys]
            for key in sorted((keys - set(self._ref_keys)) & content_keys):
                self._items.append((record.font_id, key))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        font_id, target_key = self._items[idx]
        refs_available = self._refs_for[font_id]

        # Padded to the full seed length even though these are fixed per font:
        # a font missing one seed glyph would otherwise produce a shorter stack
        # and break collation at whatever batch boundary happens to span two
        # fonts. The mask keeps the encoder's pooling correct either way.
        n_refs = len(refs_available)
        refs = np.zeros((self._max_refs, self.store.image_size, self.store.image_size),
                        dtype=np.float32)
        for i, key in enumerate(refs_available):
            refs[i] = self.store.image(font_id, key)
        ref_mask = torch.zeros(self._max_refs, dtype=torch.bool)
        ref_mask[:n_refs] = True

        return {
            "content": _to_tensor(self.store.image(self.content_font, target_key)),
            "refs": torch.from_numpy(refs).unsqueeze(1) * 2.0 - 1.0,
            "ref_mask": ref_mask,
            "target": _to_tensor(self.store.image(font_id, target_key)),
            "char_id": torch.tensor(self.glyph_index[target_key], dtype=torch.long),
            "advance": torch.tensor(self.store.advance(font_id, target_key), dtype=torch.float32),
        }
