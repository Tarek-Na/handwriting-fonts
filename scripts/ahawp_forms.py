"""Positional-form completion on AHAWP: the test, its floors and its ceiling.

The task: from a writer's isolated letters, draw that writer's initial,
medial and final forms (45 joined forms per writer). Every method is scored
the same way: ``aligned_centreline_f1`` (3 px, best offset within 6 px)
against the writer's typical copy of the form, on writers of one split
(``ahawp.split_of``), never trained on.

    python scripts/ahawp_forms.py prepare        # cache: data/ahawp/forms.npz
    python scripts/ahawp_forms.py score --split val

Methods (no model):

* ``ceiling``: the writer's own second copy of the form. No method should
  be expected above it: people do not write a letter the same way twice.
* ``isolated``: the isolated letter, unchanged, for every form.
* ``retrieve``: the form as written by the training writer whose isolated
  letters are most like this writer's.

A rule that keeps the isolated letter and adds connecting stubs was tried
and dropped: it scored 0.287 against 0.281 for the isolated letter alone on
the validation writers, because joined forms are not the isolated body with
tails (beh's bowl becomes a tooth, heh's ring a figure on the line).

The cache holds images of real handwriting: it lives under ``data/ahawp/``,
which git ignores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.evaluate.metrics import aligned_centreline_f1  # noqa: E402

log = logging.getLogger("ahawp_forms")
ARCHIVE = ROOT / "data/ahawp/isolated_alphabets_per_user.zip"
CACHE = ROOT / "data/ahawp/forms.npz"
JOINED = (Form.INIT, Form.MEDI, Form.FINA)


# ------------------------------------------------------------------ the cache
def _second(copies: list[np.ndarray], first: int) -> int:
    """The most typical copy other than ``first``."""
    rest = [i for i in range(len(copies)) if i != first]
    return rest[ahawp.typical([copies[i] for i in rest])]


def _prepare_writer(writer: str) -> dict[str, np.ndarray]:
    archive = ahawp.open_archive(ARCHIVE)
    out = {}
    for spec, names in ahawp.index(archive)[writer].items():
        copies = [ahawp.ink(archive.read(n)) for n in names]
        copies = [c if c.sum() >= 20 else np.zeros_like(c) for c in copies]
        first = ahawp.typical(copies)
        out[f"{writer}|{spec.name}|0"] = copies[first].astype(bool)
        out[f"{writer}|{spec.name}|1"] = copies[_second(copies, first)].astype(bool)
    return out


def prepare(workers: int) -> None:
    writers = sorted(ahawp.index(ahawp.open_archive(ARCHIVE)))
    t0, arrays = time.time(), {}
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_prepare_writer, writers):
            arrays.update(part)
    np.savez_compressed(CACHE, **arrays)
    log.info("%d writers, %d images -> %s (%.0fs)", len(writers), len(arrays), CACHE, time.time() - t0)


def load() -> dict[str, dict[str, list[np.ndarray]]]:
    """writer -> glyph name -> [typical copy, second copy] (bool images)."""
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    with np.load(CACHE) as data:
        for key in data.files:
            writer, name, k = key.split("|")
            out.setdefault(writer, {}).setdefault(name, [None, None])[int(k)] = data[key]
    return out


def specs_by_name() -> dict[str, GlyphSpec]:
    return {GlyphSpec(c, f).name: GlyphSpec(c, f) for c in ahawp.LETTERS.values() for f in Form}


# ------------------------------------------------------------------ the methods
def isolated_likeness(a: dict[str, list[np.ndarray]], b: dict[str, list[np.ndarray]]) -> float:
    names = [GlyphSpec(c).name for c in ahawp.LETTERS.values()]
    scores = [aligned_centreline_f1(a[n][0], b[n][0], shift=3) for n in names if n in a and n in b]
    return float(np.mean(scores)) if scores else 0.0


def _score_writer(args) -> dict:
    writer, letters, train, methods = args
    specs = specs_by_name()
    nearest = None
    if "retrieve" in methods:
        nearest = max(train, key=lambda w: isolated_likeness(letters, train[w]))
    rows = []
    for name, (first, second) in sorted(letters.items()):
        spec = specs.get(name)
        if spec is None or spec.form not in JOINED or not first.any():
            continue
        isolated = letters.get(GlyphSpec(spec.char).name)
        drawn = {}
        if "ceiling" in methods:
            drawn["ceiling"] = second
        if isolated is not None:
            if "isolated" in methods:
                drawn["isolated"] = isolated[0]
        if nearest is not None and name in train[nearest]:
            drawn["retrieve"] = train[nearest][name][0]
        rows.append({"writer": writer, "glyph": name, "form": spec.form.name,
                     **{m: (aligned_centreline_f1(im, first) if im is not None else 0.0)
                        for m, im in drawn.items()}})
    return {"writer": writer, "nearest": nearest, "rows": rows}


def score(split: str, methods: list[str], workers: int, out: Path | None) -> None:
    data = load()
    train = {w: v for w, v in data.items() if ahawp.split_of(w) == "train"}
    writers = sorted(w for w in data if ahawp.split_of(w) == split)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_score_writer, [(w, data[w], train, methods) for w in writers]))
    rows = [r for res in results for r in res["rows"]]
    print(f"{len(writers)} {split} writers, {len(rows)} joined forms ({time.time() - t0:.0f}s)")
    print(f"{'method':10} {'all':>6} {'init':>6} {'medi':>6} {'fina':>6}")
    summary = {}
    for m in methods:
        vals = {f: [r[m] for r in rows if m in r and (f == "all" or r["form"] == f)]
                for f in ("all", "INIT", "MEDI", "FINA")}
        summary[m] = {f: float(np.mean(v)) for f, v in vals.items() if v}
        print(f"{m:10} " + " ".join(f"{summary[m].get(f, float('nan')):6.3f}"
                                    for f in ("all", "INIT", "MEDI", "FINA")))
    if out:
        out.write_text(json.dumps({"split": split, "summary": summary, "results": results}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--split", default="val")
    s.add_argument("--methods", default="ceiling,isolated,retrieve")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "prepare":
        prepare(args.workers)
    else:
        score(args.split, args.methods.split(","), args.workers, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
