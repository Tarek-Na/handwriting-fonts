"""Add NIST SD19's real writers to a glyph cache, one "font" per writer.

Needs ``data/sd19/by_write.zip`` and the labels from
``python scripts/sd19_labels.py --fetch``. Writes into an existing cache
(``hfont.cli prepare`` output), replacing any SD19 writers already there, then
re-packs it if it was packed. How often they are drawn in training is
``data.sd19_weight`` (1.0 by default; 0 leaves them out).

    python scripts/sd19_cache.py --cache data/cache
    python scripts/sd19_cache.py --cache data/cache --limit 20   # a quick look
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import get_charset  # noqa: E402
from hfont.data.prepare import PACKED_IMAGES, pack_cache  # noqa: E402
from hfont.data.sd19 import add_to_cache, iter_writer_fonts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--by-write", default=str(ROOT / "data/sd19/by_write.zip"))
    ap.add_argument("--labels", default=str(ROOT / "data/sd19/by_write_labels.json"))
    ap.add_argument("--min-letters", type=int, default=40,
                    help="skip writers with fewer usable characters than this")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N writers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    charset = get_charset(meta["charset"])
    size = int(meta["render"]["size"])
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    t = time.time()

    def progress(items):
        for i, item in enumerate(items, 1):
            if i % 200 == 0:
                print(f"  {i} writers ({time.time() - t:.0f}s)", flush=True)
            yield item

    writers = iter_writer_fonts(args.by_write, labels, charset, size, args.min_letters)
    n = add_to_cache(cache, progress(islice(writers, args.limit)))
    print(f"added {n} SD19 writers to {cache} ({time.time() - t:.0f}s)")
    if (cache / PACKED_IMAGES).is_file():
        pack_cache(cache)
        print("re-packed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
