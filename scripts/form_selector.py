"""Choose, for each joined form, the donor writer whose form is most like the writer's own.

Drawing joined forms pixel by pixel collapses to a blur (``train_forms.py``:
0.16-0.19 on the val writers). But among a writer's 20 most alike training
writers, the best donor form scores 0.712 on average, above the writer's
own second copy (0.666): the right letter exists; the problem is to choose
it from the isolated letters alone. This learns that choice.

For each writer, joined form and candidate donor (the writer's ``K`` most
alike training writers, ``ahawp_forms.py neighbours``), the features are
how alike the two writers' isolated letters are, letter by letter
(``aligned_centreline_f1``), the same letter's likeness and size ratio, the
donor's rank, and the letter and form; the label is how alike the donor's
form is to the writer's own (the score the test uses). A gradient-boosted
regressor learns the label from training writers only; for a new writer the
donor with the highest predicted label is chosen. The output is always a
real written letter, so it cannot blur.

    python scripts/form_selector.py table          # data/ahawp/selector_table.npz
    python scripts/form_selector.py fit --split val
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ahawp_forms as af  # noqa: E402

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.evaluate.metrics import aligned_centreline_f1  # noqa: E402

log = logging.getLogger("form_selector")
TABLE = ROOT / "data/ahawp/selector_table.npz"
K = 20
CHARS = list(ahawp.LETTERS.values())
FORMS = [Form.INIT, Form.MEDI, Form.FINA]


def _size(im: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(im.any(axis=1))[0], np.nonzero(im.any(axis=0))[0]
    return (float(rows[-1] - rows[0] + 1), float(cols[-1] - cols[0] + 1)) if rows.size else (1.0, 1.0)


def features(writer_letters: dict, donor_letters: dict, likeness: np.ndarray, rank: int,
             char: str, form: Form) -> np.ndarray:
    """One candidate's features: per-letter likeness (18), this letter's likeness and size
    ratios, the donor's rank and mean likeness, the letter and form (one-hot)."""
    iso = GlyphSpec(char).name
    same = likeness[CHARS.index(char)]
    if iso in writer_letters and iso in donor_letters:
        (wh, ww), (dh, dw) = _size(writer_letters[iso][0]), _size(donor_letters[iso][0])
        ratios = [np.log(dh / wh), np.log(dw / ww)]
    else:
        ratios = [0.0, 0.0]
    onehot = np.zeros(len(CHARS) + len(FORMS))
    onehot[CHARS.index(char)] = 1
    onehot[len(CHARS) + FORMS.index(form)] = 1
    return np.concatenate([likeness, [same, *ratios, rank, np.nanmean(likeness)], onehot])


def _rows(args):
    writer, letters, donors = args
    out_x, out_y, out_key = [], [], []
    for rank, (donor, dl) in enumerate(donors):
        likeness = np.array([aligned_centreline_f1(letters[GlyphSpec(c).name][0], dl[GlyphSpec(c).name][0], shift=3)
                             if GlyphSpec(c).name in letters and GlyphSpec(c).name in dl else np.nan
                             for c in CHARS])
        for char in CHARS:
            for form in FORMS:
                name = GlyphSpec(char, form).name
                if name not in letters or name not in dl or not letters[name][0].any():
                    continue
                out_x.append(features(letters, dl, likeness, rank, char, form))
                out_y.append(aligned_centreline_f1(dl[name][0], letters[name][0]))
                out_key.append(f"{writer}|{name}|{donor}")
    return out_x, out_y, out_key


def table(workers: int) -> None:
    data, ranked = af.load(), af.load_neighbours()
    jobs = [(w, data[w], [(d, data[d]) for d in ranked[w][:K]]) for w in sorted(data)]
    t0, xs, ys, keys = time.time(), [], [], []
    with ProcessPoolExecutor(workers) as pool:
        for x, y, k in pool.map(_rows, jobs):
            xs += x
            ys += y
            keys += k
    np.savez_compressed(TABLE, x=np.array(xs, np.float32), y=np.array(ys, np.float32), key=np.array(keys))
    log.info("%d candidate rows -> %s (%.0fs)", len(ys), TABLE, time.time() - t0)


def load_table():
    with np.load(TABLE) as t:
        keys = t["key"]
        writer = np.array([k.split("|")[0] for k in keys])
        form = np.array(["|".join(k.split("|")[:2]) for k in keys])
        rank = t["x"][:, len(CHARS) + 3].astype(int)
        return t["x"], t["y"], writer, form, rank


def choose(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Index of the highest prediction in each group (one group: one writer's form)."""
    order = np.lexsort((-pred, groups))
    first = np.ones(len(order), bool)
    first[1:] = groups[order][1:] != groups[order][:-1]
    return order[first]


def fit(split: str, out: Path | None) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    x, y, writer, form, rank = load_table()
    side = np.array([ahawp.split_of(w) for w in writer])
    train, test = side == "train", side == split
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
    model.fit(x[train], y[train])
    pred = model.predict(x[test])
    groups, yt, rt = form[test], y[test], rank[test]
    picked = choose(pred, groups)
    nearest = choose(-rt.astype(float), groups)
    best = choose(yt, groups)
    print(f"{len(np.unique(writer[test]))} {split} writers, {len(picked)} joined forms, {K} candidates each")
    print(f"  nearest donor (retrieve)   {yt[nearest].mean():.3f}")
    print(f"  learned choice             {yt[picked].mean():.3f}")
    print(f"  best candidate (oracle)    {yt[best].mean():.3f}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pickle.dumps(model))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("fit")
    f.add_argument("--split", default="val")
    f.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "table":
        table(args.workers)
    else:
        fit(args.split, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
