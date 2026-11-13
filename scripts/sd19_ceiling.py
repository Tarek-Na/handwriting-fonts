"""The attribution ceiling: a writer's own second copy of a letter.

``scripts/sd19_style.py`` asks whether a blind attempt at a writer's letter is
closer to their real letter than to nine other writers'. Even the real
handwriting of the best-matching other person scores only ~21% there, so the
question is what the test can reach at all. SD19 writers wrote every digit
about 13 times (letters once), which answers it: score each writer's *second*
copy of a digit as if it were the blind attempt, against their first copy.
Nothing can be expected to beat the writer's own hand.

Digits on the seed sheet (0 3 6) are left out, so the rest (1 2 4 5 7 8 9)
are exactly the ones a model predicts blind. Same writers (the cache's
``val`` split, by writer-id hash), same groups of ten, same clDice.

    python scripts/sd19_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import LATIN_CORE, LATIN_SEED_SETS, get_charset  # noqa: E402
from hfont.data.sd19 import _split, index_writers, writer_glyphs  # noqa: E402
from hfont.evaluate.style import grouped_style_report  # noqa: E402

UNSEEN_DIGITS = [d for d in "0123456789" if d not in LATIN_SEED_SETS["seed30"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-write", default=str(ROOT / "data/sd19/by_write.zip"))
    ap.add_argument("--labels", default=str(ROOT / "data/sd19/by_write_labels.json"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--held-out", type=float, default=0.03, help="as sd19_cache.py")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    charset = get_charset("latin_core")
    key = {g.char: g.key for g in LATIN_CORE}
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    by_write = zipfile.ZipFile(args.by_write)
    first, second = {}, {}
    for writer in index_writers(by_write, labels):
        if _split(writer.writer_id, args.held_out) not in args.split.split(","):
            continue
        if not all(len(writer.members.get(d, [])) >= 2 for d in UNSEEN_DIGITS):
            continue
        again = replace(writer, members={c: (n[1:] if c in UNSEEN_DIGITS else n)
                                         for c, n in writer.members.items()})
        a, b = writer_glyphs(by_write, writer, charset), writer_glyphs(by_write, again, charset)
        if not all(key[d] in a and key[d] in b for d in UNSEEN_DIGITS):
            continue
        first[writer.writer_id] = {d: a[key[d]] for d in UNSEEN_DIGITS}
        second[writer.writer_id] = {d: b[key[d]] for d in UNSEEN_DIGITS}
        if len(first) >= args.limit:
            break

    r = grouped_style_report(second, first)
    print(f"{len(first)} SD19 {args.split} writers, {r['groups']} groups of 10, digits "
          f"{''.join(UNSEEN_DIGITS)}")
    print(f"own second copy: alike {r['generated_similarity']:.3f} (first copies "
          f"{r['real_similarity']:.3f}), attributed {r['attribution']:.1%} (chance 10%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
