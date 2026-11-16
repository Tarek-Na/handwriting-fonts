"""Template skeletons and writers' moves on AHAWP (step A5, day 1).

For each form, the template is the pen path of the training writer whose
skeleton is nearest all the other training writers' (the medoid, by
symmetric chamfer distance). Every writer's form is then described as the
template's points moved onto that writer's skeleton (``hfont.skeleton``).

    python scripts/skeleton_moves.py fit --split train   # data/ahawp/moves_train.npz
    python scripts/skeleton_moves.py fit --split val
    python scripts/skeleton_moves.py report --split val

``report`` scores, on one split, against the writer's typical copy
(``aligned_centreline_f1``, the positional-form task's measure):

* ``template``: the template redrawn with the writer's pen, no moves;
* ``affine``: the template with only the writer's affine fit (size, slant);
* ``fit``: the template with all the writer's moves: how much of a writer
  this representation can hold (its ceiling; a predictor can only lose);
* ``mean``: the template with the training writers' average moves (the
  default skeleton).

Everything written here is derived from real handwriting: it stays in
``data/ahawp/``, which git ignores.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# One maths thread per worker: with the default (one per core) in every
# process, 6 workers on 8 cores ran over 100 times slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont import skeleton  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.evaluate.metrics import aligned_centreline_f1  # noqa: E402

log = logging.getLogger("skeleton_moves")
TEMPLATES = ROOT / "data/ahawp/templates.npz"


def moves_path(split: str) -> Path:
    return ROOT / f"data/ahawp/moves_{split}.npz"


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    return float(d.min(1).mean() + d.min(0).mean())


def _points_of(data: dict, writers: list[str]) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = {}
    for w in writers:
        for name, (first, _) in data[w].items():
            got = skeleton.skeleton_points(first.astype(np.float32)) if first.any() else None
            if got is not None:
                out.setdefault(name, {})[w] = got
    return out


def _medoid(args) -> tuple[str, str, np.ndarray, np.ndarray]:
    name, by_writer = args
    writers = sorted(by_writer)
    cost = [sum(_chamfer(by_writer[a][0], by_writer[b][0]) for b in writers if b != a) for a in writers]
    best = writers[int(np.argmin(cost))]
    return name, best, *by_writer[best]


def templates(workers: int) -> None:
    from ahawp_forms import load

    data = load()
    train = sorted(w for w in data if ahawp.split_of(w) == "train")
    t0 = time.time()
    points = _points_of(data, train)
    arrays = {}
    with ProcessPoolExecutor(workers) as pool:
        for name, writer, pts, ids in pool.map(_medoid, sorted(points.items())):
            arrays[f"{name}|points"], arrays[f"{name}|ids"] = pts, ids
            log.info("%-14s template from %s (%d points)", name, writer, len(pts))
    np.savez_compressed(TEMPLATES, **arrays)
    log.info("%d templates -> %s (%.0fs)", len(arrays) // 2, TEMPLATES, time.time() - t0)


def load_templates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(TEMPLATES) as z:
        names = {k.split("|")[0] for k in z.files}
        return {n: (z[f"{n}|points"], z[f"{n}|ids"]) for n in names}


def _fit_writer(args) -> dict[str, np.ndarray]:
    writer, letters, temps, kw = args
    out = {}
    for name, (first, _) in letters.items():
        if name not in temps or not first.any():
            continue
        got = skeleton.skeleton_points(first.astype(np.float32))
        if got is None:
            continue
        tpl = temps[name][0]
        b, t = skeleton.affine(tpl, got[0], kw.get("affine_outliers", 0.01))
        moved = skeleton.register(tpl, got[0], **kw)
        out[f"{writer}|{name}|affine"] = np.concatenate([b.ravel(), t])
        out[f"{writer}|{name}|moved"] = moved.astype(np.float32)
    return out


def fit(split: str, workers: int, kw: dict) -> None:
    from ahawp_forms import load

    data = load()
    temps = load_templates()
    writers = sorted(w for w in data if ahawp.split_of(w) == split)
    t0, arrays = time.time(), {}
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(_fit_writer, [(w, data[w], temps, kw) for w in writers]), 1):
            arrays.update(part)
            if i % 10 == 0:
                log.info("%d/%d writers (%.0fs)", i, len(writers), time.time() - t0)
    np.savez_compressed(moves_path(split), **arrays)
    log.info("%d writers, %d forms -> %s (%.0fs)", len(writers), len(arrays) // 2, moves_path(split),
             time.time() - t0)


def load_moves(split: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """writer -> glyph name -> {"affine": (6,), "moved": (M, 2)}."""
    out: dict = {}
    with np.load(moves_path(split)) as z:
        for key in z.files:
            w, name, kind = key.split("|")
            out.setdefault(w, {}).setdefault(name, {})[kind] = z[key]
    return out


def writer_radius(letters: dict[str, list[np.ndarray]]) -> float:
    """The writer's pen half-width from their isolated letters (what a user hands in)."""
    from ahawp_forms import specs_by_name

    specs = specs_by_name()
    iso = {specs[n]: v[0].astype(np.float32) for n, v in letters.items()
           if n in specs and specs[n].form.name == "ISOL"}
    return ahawp.pen_radius(iso)


def _report_writer(args) -> list[dict]:
    writer, letters, temps, moves, mean = args
    radius = writer_radius(letters)
    rows = []
    for name, got in moves.items():
        target = letters[name][0]
        tpl, ids = temps[name]
        a = got["affine"]
        drawn = {
            "template": tpl,
            "affine": tpl @ a[:4].reshape(2, 2).T + a[4:],
            "fit": got["moved"],
            "mean": tpl + mean[name],
        }
        row = {"writer": writer, "glyph": name, "ceiling": aligned_centreline_f1(letters[name][1], target)}
        for k, pts in drawn.items():
            row[k] = aligned_centreline_f1(skeleton.render_points(pts, ids, radius), target)
        rows.append(row)
    return rows


def report(split: str, workers: int, limit: int | None) -> None:
    from ahawp_forms import load

    data = load()
    temps = load_templates()
    train = load_moves("train")
    mean = {}
    for name, (tpl, _) in temps.items():
        d = [m[name]["moved"] - tpl for m in train.values() if name in m]
        mean[name] = np.mean(d, axis=0)
    moves = load_moves(split)
    writers = sorted(moves)[:limit]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for part in pool.map(_report_writer, [(w, data[w], temps, moves[w], mean) for w in writers])
                for r in part]
    print(f"{len(writers)} {split} writers, {len(rows)} forms ({time.time() - t0:.0f}s)")
    joined = [r for r in rows if "." in r["glyph"]]
    for label, rs in (("all", rows), ("joined", joined)):
        print(f"{label:7} " + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.3f}"
                                       for k in ("ceiling", "template", "mean", "affine", "fit")))


FEATURES = "full"


def _features(moves: dict, names: list[str], fill: dict[str, np.ndarray]) -> np.ndarray:
    """A writer's style from their isolated letters' affine fits.

    ``full``: all fits side by side (training mean where one is missing);
    ``compact``: the median over letters of the 4 linear terms (width and
    height scale, the two shears), robust to one badly fitted letter.
    """
    if FEATURES == "compact":
        return np.median(np.stack([moves[n]["affine"][:4] for n in names if n in moves]), axis=0)
    return np.concatenate([moves[n]["affine"] if n in moves else fill[n] for n in names])


def predicted(split: str, alpha: float, residual: bool) -> dict[str, dict[str, np.ndarray]]:
    """writer -> joined glyph -> predicted points, by ridge regression from isolated affine fits.

    One ridge per joined form, trained on the training writers: input the
    writer's isolated letters' affine fits (18 x 6 numbers), output the
    form's affine fit (6). With ``residual``, the training writers' mean
    local moves (in the template's frame, after their affine) are added.
    """
    from sklearn.linear_model import Ridge

    temps = load_templates()
    train = load_moves("train")
    iso = sorted(n for n in temps if "." not in n)
    fill = {n: np.mean([m[n]["affine"] for m in train.values() if n in m], axis=0) for n in iso}
    target = load_moves(split)
    out: dict[str, dict[str, np.ndarray]] = {w: {} for w in target}
    for name in sorted(n for n in temps if "." in n):
        rows = [(w, m) for w, m in train.items() if name in m]
        x = np.stack([_features(m, iso, fill) for _, m in rows])
        y = np.stack([m[name]["affine"] for _, m in rows])
        mu, sd = x.mean(0), x.std(0) + 1e-6
        model = Ridge(alpha=alpha).fit((x - mu) / sd, y)
        tpl = temps[name][0]
        local = np.zeros_like(tpl)
        if residual:
            res = []
            for _, m in rows:
                a = m[name]["affine"]
                b = a[:4].reshape(2, 2)
                res.append((m[name]["moved"] - a[4:]) @ np.linalg.inv(b).T - tpl)
            local = np.mean(res, axis=0)
        for w, m in target.items():
            a = model.predict(((_features(m, iso, fill) - mu) / sd)[None])[0]
            out[w][name] = (tpl + local) @ a[:4].reshape(2, 2).T + a[4:]
    return out


def _score_predicted(args) -> list[dict]:
    writer, letters, temps, drawn = args
    radius = writer_radius(letters)
    return [{"glyph": n, "score": aligned_centreline_f1(skeleton.render_points(p, temps[n][1], radius),
                                                         letters[n][0])}
            for n, p in drawn.items() if n in letters]


def score_predicted(split: str, alpha: float, residual: bool, workers: int) -> float:
    from ahawp_forms import load

    data = load()
    temps = load_templates()
    pred = predicted(split, alpha, residual)
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for part in pool.map(_score_predicted, [(w, data[w], temps, pred[w]) for w in sorted(pred)])
                for r in part]
    by_form = {f: np.mean([r["score"] for r in rows if r["glyph"].endswith(f)]) for f in (".init", ".medi", ".fina")}
    total = float(np.mean([r["score"] for r in rows]))
    print(f"ridge alpha={alpha} residual={residual}: {split} joined {total:.3f}  "
          + "  ".join(f"{k[1:]} {v:.3f}" for k, v in by_form.items()))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("templates")
    t.add_argument("--workers", type=int, default=6)
    f = sub.add_parser("fit")
    f.add_argument("--split", default="train")
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--beta", type=float, default=12.0)
    f.add_argument("--smooth", type=float, default=2.0)
    f.add_argument("--outliers", type=float, default=0.01)
    f.add_argument("--affine-outliers", type=float, default=0.01)
    r = sub.add_parser("report")
    r.add_argument("--split", default="val")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("ridge")
    p.add_argument("--split", default="val")
    p.add_argument("--alpha", type=float, nargs="+", default=[10.0])
    p.add_argument("--residual", action="store_true")
    p.add_argument("--features", choices=["full", "compact"], default="full")
    p.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "ridge":
        global FEATURES
        FEATURES = args.features
        for a in args.alpha:
            score_predicted(args.split, a, args.residual, args.workers)
    elif args.cmd == "templates":
        templates(args.workers)
    elif args.cmd == "fit":
        fit(args.split, args.workers, {"beta": args.beta, "smooth": args.smooth, "outliers": args.outliers,
                                       "affine_outliers": args.affine_outliers})
    else:
        report(args.split, args.workers, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
