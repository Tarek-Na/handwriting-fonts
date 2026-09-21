"""Render a manifest into an on-disk glyph cache.

Rendering happens once, ahead of training, for two reasons. It is far too slow
to do in a DataLoader worker (a few ms per glyph, and every sample needs several
glyphs), and more importantly the cache is the thing that gets carried to the
training machine. A checkpoint is only meaningful against the exact pixels it
was trained on, so the cache is content-addressed by its render settings.

Layout::

    cache/
      meta.json              render config, charset, per-font index
      fonts/<font_id>.npz    images uint8 (N,S,S), keys, advances

One file per font rather than a few big shards, because the sampler draws all
of a sample's glyphs (target plus K style references) from a single font, so
per-font locality is exactly the access pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..charset import Charset, get_charset
from .corpus import FontEntry, Manifest
from .render import FontRenderError, FontRenderer, RenderConfig

log = logging.getLogger(__name__)


def _font_cache_name(font_id: str) -> str:
    """Filesystem-safe name for a font id like ``ofl-roboto/Roboto-Regular``."""
    return font_id.replace("/", "__").replace("\\", "__")


def render_one(
    entry: FontEntry, charset: Charset, config: RenderConfig, out_dir: Path
) -> dict | None:
    """Render one font to an ``.npz``. Returns its index record, or None."""
    out_path = out_dir / f"{_font_cache_name(entry.font_id)}.npz"
    try:
        with FontRenderer(entry.path, config) as renderer:
            rasters, frame = renderer.render(charset)
    except FontRenderError as exc:
        log.debug("render failed for %s: %s", entry.font_id, exc)
        return None
    except Exception as exc:
        log.debug("render crashed for %s: %s: %s", entry.font_id, type(exc).__name__, exc)
        return None

    keys = sorted(rasters)
    if not keys:
        return None

    images = np.stack([
        np.round(rasters[k].image * 255.0).astype(np.uint8) for k in keys
    ])
    advances = np.asarray([rasters[k].advance for k in keys], dtype=np.float32)
    blank = np.asarray([rasters[k].is_blank for k in keys], dtype=bool)

    # Pixel hash per glyph. Fonts routinely point several characters at one
    # outline (hyphen/minus, O/zero in some display faces, anything undefined
    # falling through to .notdef). Those duplicates must not be usable as a
    # style reference for each other, or the sample becomes a copy task.
    hashes = np.asarray(
        [hashlib.blake2b(images[i].tobytes(), digest_size=8).hexdigest()
         for i in range(len(keys))]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        keys=np.asarray(keys),
        advances=advances,
        blank=blank,
        hashes=hashes,
    )
    return {
        "font_id": entry.font_id,
        "file": out_path.name,
        "family": entry.family,
        "style": entry.style,
        "category": entry.category,
        "split": entry.split,
        "n_glyphs": len(keys),
        "scale": frame.scale,
        "baseline_y": frame.baseline_y,
        "units_per_em": frame.units_per_em,
    }


def _worker(args) -> dict | None:
    entry_dict, charset_name, config_dict, out_dir = args
    return render_one(
        FontEntry(**entry_dict),
        get_charset(charset_name),
        RenderConfig(**config_dict),
        Path(out_dir),
    )


def build_cache(
    manifest: Manifest,
    out_dir: str | Path,
    charset: Charset | None = None,
    config: RenderConfig | None = None,
    workers: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Render every font in ``manifest`` into ``out_dir``."""
    charset = charset or get_charset(manifest.charset)
    config = config or RenderConfig()
    out_dir = Path(out_dir)
    fonts_dir = out_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        done = {r["font_id"] for r in existing["fonts"]}
        records = list(existing["fonts"])
        log.info("resuming: %d fonts already cached", len(done))
    else:
        done, records = set(), []

    todo = [e for e in manifest.entries if e.font_id not in done]
    if not todo:
        log.info("cache already complete (%d fonts)", len(records))
        return out_dir

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log.info("rendering %d fonts with %d workers -> %s", len(todo), workers, out_dir)

    payload = [
        (asdict(e), charset.name, asdict(config), str(fonts_dir)) for e in todo
    ]

    n_failed = 0
    if workers == 1:
        results = (_worker(p) for p in payload)
        for i, record in enumerate(results, 1):
            if record is None:
                n_failed += 1
            else:
                records.append(record)
            if i % 100 == 0:
                log.info("  %d/%d rendered (%d failed)", i, len(todo), n_failed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, p) for p in payload]
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                if record is None:
                    n_failed += 1
                else:
                    records.append(record)
                if i % 100 == 0:
                    log.info("  %d/%d rendered (%d failed)", i, len(todo), n_failed)

    records.sort(key=lambda r: r["font_id"])
    meta = {
        "charset": charset.name,
        "glyph_keys": [g.key for g in charset],
        "render": asdict(config),
        "fonts": records,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log.info("cache written: %d fonts (%d failed)", len(records), n_failed)
    return out_dir


PACKED_IMAGES = "packed_images.npy"
PACKED_INDEX = "packed_index.npz"


def pack_cache(cache_dir: str | Path) -> Path:
    """Consolidate the per-font ``.npz`` files into one memory-mappable array.

    The per-font layout is right for building and resuming a cache and wrong
    for training on a large corpus. The sampler draws fonts uniformly, so with
    a few thousand fonts almost every sample misses any practical in-memory LRU
    and decompresses two fonts from disk — on a two-core Colab runtime that is
    enough to leave the GPU idle most of the time.

    Packed, every font's glyphs are a contiguous row range of a single uint8
    array. The store maps it read-only, a font becomes a zero-copy slice, the
    operating system's page cache is shared by every DataLoader worker instead
    of each worker holding its own copy, and nothing is decompressed per sample.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    size = int(meta["render"]["size"])
    records = meta["fonts"]

    counts = []
    for record in records:
        with np.load(cache_dir / "fonts" / record["file"]) as data:
            counts.append(int(data["images"].shape[0]))
    total = int(sum(counts))

    images = np.lib.format.open_memmap(
        cache_dir / PACKED_IMAGES, mode="w+", dtype=np.uint8, shape=(total, size, size)
    )
    starts = np.zeros(len(records), dtype=np.int64)
    keys: list[str] = []
    advances = np.zeros(total, dtype=np.float32)
    hashes: list[str] = []

    row = 0
    for i, record in enumerate(records):
        with np.load(cache_dir / "fonts" / record["file"]) as data:
            n = counts[i]
            images[row : row + n] = data["images"]
            advances[row : row + n] = data["advances"]
            keys.extend(str(k) for k in data["keys"])
            if "hashes" in data:
                hashes.extend(str(h) for h in data["hashes"])
            else:
                hashes.extend([""] * n)
        starts[i] = row
        row += n
        if (i + 1) % 500 == 0:
            log.info("  packed %d/%d fonts", i + 1, len(records))

    images.flush()
    del images
    np.savez(
        cache_dir / PACKED_INDEX,
        font_ids=np.asarray([r["font_id"] for r in records]),
        starts=starts,
        counts=np.asarray(counts, dtype=np.int64),
        keys=np.asarray(keys),
        advances=advances,
        hashes=np.asarray(hashes),
    )
    log.info("packed %d glyphs from %d fonts (%.2f GB)", total, len(records),
             total * size * size / 1e9)
    return cache_dir / PACKED_IMAGES
