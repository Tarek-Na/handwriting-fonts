"""Handwriting variation on AHAWP: many plausible versions of each form (step A6).

A writer never writes a letter the same way twice: on AHAWP, one writer's
two copies of a form differ in size about as much as two writers do
(``skeleton_moves.py``). So instead of one exact letter, a model here
gives a distribution of versions of each form in a writer's style, and a
font carries several per letter.

    python scripts/variation.py copies          # every copy's moves -> data/ahawp/variation.npz
    python scripts/variation.py pca             # per-form components -> data/ahawp/variation_pca.npz
    python scripts/variation.py train --out runs/v1
    python scripts/variation.py score --split val --methods template,gauss,other,own --model runs/v1/best.pt

``copies`` registers every copy (ten per form) of every writer to the
form's template (``data/ahawp/templates.npz``) and keeps the moves
(fitted points minus template, (M, 2) per copy). For validation and test
writers it also keeps the copies' ink, cut to the ink, to score drawn
versions against. All of it is derived from handwriting and stays in
``data/ahawp/``, which git ignores.

``score`` draws versions of each joined form for every writer of a split
and compares them with the writer's real copies (even-numbered copies;
the odd ones are the ``own`` ceiling's versions), by
``aligned_centreline_f1``:

* realism: for each version, its best match among the real copies;
* coverage: for each real copy, its best match among the versions;
* other-writer realism: realism against another writer's copies instead.
  A version in this writer's style should match them less well.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont import skeleton  # noqa: E402
from hfont.data import ahawp  # noqa: E402

log = logging.getLogger("variation")
ARCHIVE = ROOT / "data/ahawp/isolated_alphabets_per_user.zip"
CACHE = ROOT / "data/ahawp/variation.npz"
PCA = ROOT / "data/ahawp/variation_pca.npz"
K = 24
VERSIONS = 5


def _cut(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _copies_of_writer(args) -> dict[str, np.ndarray]:
    writer, temps = args
    archive = ahawp.open_archive(ARCHIVE)
    keep_ink = ahawp.split_of(writer) != "train"
    out = {}
    for spec, files in ahawp.index(archive)[writer].items():
        if spec.name not in temps:
            continue
        tpl = temps[spec.name][0]
        for c, f in enumerate(files):
            ink = ahawp.ink(archive.read(f))
            if ink.sum() < 20:
                continue
            got = skeleton.skeleton_points(ink)
            if got is None:
                continue
            key = f"{writer}|{spec.name}|{c}"
            out[f"{key}|moves"] = (skeleton.register(tpl, got[0], beta=4.0, smooth=0.5) - tpl).astype(np.float32)
            if keep_ink:
                out[f"{key}|ink"] = _cut(ink > 0.5)
    return out


def copies(workers: int) -> None:
    from skeleton_moves import load_templates

    temps = load_templates()
    writers = sorted(ahawp.index(ahawp.open_archive(ARCHIVE)))
    t0, arrays = time.time(), {}
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(_copies_of_writer, [(w, temps) for w in writers]), 1):
            arrays.update(part)
            if i % 5 == 0:
                log.info("%d/%d writers (%.0fs)", i, len(writers), time.time() - t0)
    np.savez_compressed(CACHE, **arrays)
    n = sum(k.endswith("|moves") for k in arrays)
    log.info("%d writers, %d copies -> %s (%.0fs)", len(writers), n, CACHE, time.time() - t0)


# ------------------------------------------------------------------ data
def load_cache() -> tuple[dict, dict]:
    """(writer -> glyph -> {copy: moves (M, 2)}, writer -> glyph -> {copy: ink (bool)})."""
    moves: dict = {}
    inks: dict = {}
    with np.load(CACHE) as z:
        for key in z.files:
            w, name, c, kind = key.split("|")
            (moves if kind == "moves" else inks).setdefault(w, {}).setdefault(name, {})[int(c)] = z[key]
    return moves, inks


def fit_pca(moves: dict) -> None:
    """Per form: mean, first K components and their spreads, on the training writers' copies."""
    from sklearn.decomposition import PCA as Pca

    arrays = {}
    names = sorted({n for m in moves.values() for n in m})
    for name in names:
        rows = np.stack([d.ravel() for w, m in moves.items() if ahawp.split_of(w) == "train" and name in m
                         for d in m[name].values()])
        pca = Pca(n_components=K).fit(rows)
        arrays[f"{name}|mean"] = pca.mean_.astype(np.float32)
        arrays[f"{name}|comps"] = pca.components_.astype(np.float32)
        arrays[f"{name}|scale"] = np.sqrt(pca.explained_variance_).astype(np.float32)
        log.info("%-14s %5d copies, %d components keep %.0f%% of the variance", name, len(rows), K,
                 100 * pca.explained_variance_ratio_.sum())
    np.savez_compressed(PCA, **arrays)


def load_pca() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(PCA) as z:
        names = {k.split("|")[0] for k in z.files}
        return {n: (z[f"{n}|mean"], z[f"{n}|comps"], z[f"{n}|scale"]) for n in names}


def encode(pca, name: str, moves: np.ndarray) -> np.ndarray:
    mean, comps, scale = pca[name]
    return ((moves.ravel() - mean) @ comps.T) / scale


def decode(pca, name: str, x: np.ndarray) -> np.ndarray:
    mean, comps, scale = pca[name]
    return (mean + (np.asarray(x) * scale) @ comps).reshape(-1, 2)


def form_names(pca) -> list[str]:
    return sorted(pca)


def isolated_names(pca) -> list[str]:
    return [n for n in form_names(pca) if "." not in n]


def encoded(moves: dict, pca) -> dict:
    """writer -> glyph -> {copy: x (K,)}."""
    return {w: {n: {c: encode(pca, n, d) for c, d in m.items()} for n, m in wm.items() if n in pca}
            for w, wm in moves.items()}


def style_inputs(xs: dict, pca, pick) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One writer's isolated letters as the style encoder's set: (L, K) x, (L,) form ids, (L,) mask.

    ``pick(copies)`` chooses which copy of each letter (a dict copy -> x).
    """
    names, iso = form_names(pca), isolated_names(pca)
    x = np.zeros((len(iso), K), np.float32)
    mask = np.zeros(len(iso), bool)
    for i, n in enumerate(iso):
        if n in xs and xs[n]:
            x[i], mask[i] = pick(xs[n]), True
    return x, np.array([names.index(n) for n in iso]), mask


# ------------------------------------------------------------------ the model
def drop_letters(mask: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Hide a share ``rate`` of each writer's isolated letters from the style encoder.

    Training only: the encoder cannot lean on any one letter or memorise a
    writer's exact set. A row keeps at least one letter, and a missing
    letter is never made present. (Val writers, letter dropout 0.3 against
    none: val loss 9.757 against 9.845; realism +0.004, specificity +0.004.)
    """
    keep = (rng.random(mask.shape) >= rate) & mask
    empty = keep.sum(1) == 0
    keep[empty] = mask[empty]
    return keep


def train(out: Path, steps: int, batch: int, style: bool, beta: float, seed: int,
          letter_dropout: float = 0.3, style_dim: int = 32) -> None:
    import torch

    from hfont.models.variation import VariationConfig, VariationModel

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    moves, _ = load_cache()
    pca = load_pca()
    names = form_names(pca)
    xs = encoded(moves, pca)
    splits = {s: sorted(w for w in xs if ahawp.split_of(w) == s) for s in ("train", "val")}
    items = {s: [(w, n, c) for w in splits[s] for n in xs[w] for c in xs[w][n]] for s in splits}
    model = VariationModel(VariationConfig(forms=len(names), k=K, style=style, beta=beta, style_dim=style_dim))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def make(rows, r):
        x = np.stack([xs[w][n][c] for w, n, c in rows]).astype(np.float32)
        form = np.array([names.index(n) for _, n, _ in rows])
        sty = [style_inputs(xs[w], pca, lambda d: d[list(d)[r.integers(len(d))]]) for w, _, _ in rows]
        t = torch.from_numpy
        return (t(x), t(form), t(np.stack([s[0] for s in sty])), t(np.stack([s[1] for s in sty])),
                t(np.stack([s[2] for s in sty])))

    vrng = np.random.default_rng(1)
    val_rows = [items["val"][i] for i in vrng.choice(len(items["val"]), 4096, replace=False)]
    val_batch = make(val_rows, vrng)
    out.mkdir(parents=True, exist_ok=True)
    best, t0 = float("inf"), time.time()
    for step in range(1, steps + 1):
        rows = [items["train"][i] for i in rng.integers(len(items["train"]), size=batch)]
        x, form, ix, iform, imask = make(rows, rng)
        if letter_dropout:
            imask = torch.from_numpy(drop_letters(imask.numpy(), letter_dropout, rng))
        loss, parts = model.loss(x, model.condition(form, ix, iform, imask))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                x, form, ix, iform, imask = val_batch
                vloss, vparts = model.loss(x, model.condition(form, ix, iform, imask))
            model.train()
            log.info("step %d  train %.2f (rec %.2f kl %.2f)  val %.2f (rec %.2f kl %.2f)  %.0fs", step,
                     float(loss), parts["rec"], parts["kl"], float(vloss), vparts["rec"], vparts["kl"],
                     time.time() - t0)
            if float(vloss) < best:
                best = float(vloss)
                torch.save(model.checkpoint(), out / "best.pt")
    log.info("best val loss %.3f -> %s", best, out / "best.pt")


def model_versions(path: str, xs: dict, pca, writers: list[str], n: int, temperature: float,
                   seed: int = 0) -> dict:
    """writer -> joined glyph -> (n, K) versions from a trained model; style from each writer's copy 0."""
    import torch

    from hfont.models.variation import VariationModel

    torch.manual_seed(seed)
    model = VariationModel.from_checkpoint(torch.load(path)).eval()
    names = form_names(pca)
    out = {}
    for w in writers:
        joined = [nm for nm in names if "." in nm and nm in xs[w]]
        ix, iform, imask = style_inputs(xs[w], pca, lambda d: d[min(d)])
        b = len(joined)
        t = torch.from_numpy
        with torch.no_grad():
            cond = model.condition(torch.tensor([names.index(nm) for nm in joined]),
                                   t(np.repeat(ix[None], b, 0)), t(np.repeat(iform[None], b, 0)),
                                   t(np.repeat(imask[None], b, 0)))
        drawn = model.sample(cond, n, temperature).numpy()
        out[w] = dict(zip(joined, drawn))
    return out


# ------------------------------------------------------------------ scoring
def _score_writer(args) -> list[dict]:
    from hfont.evaluate.metrics import aligned_centreline_f1
    from skeleton_moves import load_templates

    writer, versions, real, other, radius = args
    temps = load_templates()
    rows = []
    for name, drawn_list in versions.items():
        if name not in real or name not in other:
            continue
        reals = [real[name][c] for c in sorted(real[name]) if c % 2 == 0]
        others = [other[name][c] for c in sorted(other[name]) if c % 2 == 0]
        if not reals or not others:
            continue
        ids = temps[name][1]
        drawn = [d if d.dtype == bool else skeleton.render_points(d, ids, radius) for d in drawn_list]
        f = np.array([[aligned_centreline_f1(d, r) for r in reals] for d in drawn])
        g = np.array([[aligned_centreline_f1(d, r) for r in others] for d in drawn])
        rows.append({"writer": writer, "glyph": name, "realism": float(f.max(1).mean()), "coverage": float(f.max(0).mean()),
                     "other": float(g.max(1).mean())})
    return rows


def score(split: str, methods: list[str], model: str | None, temperature: float, workers: int,
          limit: int | None, out: Path | None = None) -> None:
    from skeleton_moves import load_templates

    moves, inks = load_cache()
    pca = load_pca()
    temps = load_templates()
    xs = encoded(moves, pca)
    writers = sorted(w for w in inks if ahawp.split_of(w) == split)[:limit]
    train_writers = sorted(w for w in xs if ahawp.split_of(w) == "train")
    rng = np.random.default_rng(0)
    radius = {w: ahawp.pen_radius({n: inks[w][n][min(inks[w][n])].astype(np.float32)
                                   for n in isolated_names(pca) if n in inks[w]}) for w in writers}

    def pts(n, x):
        return temps[n][0] + decode(pca, n, x)

    by_model = model_versions(model, xs, pca, writers, VERSIONS, temperature) if model else {}
    for method in methods + (["model"] if model else []):
        jobs = []
        for i, w in enumerate(writers):
            versions = {}
            for n in (n for n in xs[w] if "." in n):
                if method == "template":
                    got = [pts(n, np.zeros(K))]
                elif method == "gauss":
                    got = [pts(n, rng.standard_normal(K) * temperature) for _ in range(VERSIONS)]
                elif method == "other":
                    donor = train_writers[rng.integers(len(train_writers))]
                    got = [temps[n][0] + d for d in list(moves[donor].get(n, {}).values())[:VERSIONS]]
                elif method == "own":
                    got = [inks[w][n][c] for c in sorted(inks[w].get(n, {})) if c % 2 == 1]
                elif method == "model":
                    got = [pts(n, x) for x in by_model.get(w, {}).get(n, [])]
                else:
                    raise ValueError(method)
                if got:
                    versions[n] = got
            other = writers[(i + 1) % len(writers)]
            jobs.append((w, versions, inks[w], inks[other], radius[w]))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool:
            rows = [r for part in pool.map(_score_writer, jobs) for r in part]
        m = {k: np.mean([r[k] for r in rows]) for k in ("realism", "coverage", "other")}
        print(f"{method:9} {split} {len(writers)} writers {len(rows)} forms: realism {m['realism']:.3f}  "
              f"coverage {m['coverage']:.3f}  other-writer realism {m['other']:.3f}  "
              f"(own - other {m['realism'] - m['other']:+.3f})  {time.time() - t0:.0f}s", flush=True)
        if out:
            import json
            label = method if method != "model" else f"model:{model}:{temperature}"
            done = json.loads(out.read_text()) if out.exists() else {}
            done[label] = rows
            out.write_text(json.dumps(done))


# ------------------------------------------------------------------ fonts
def sheet_font(writer: str, model: str | None, versions: int, temperature: float, seed: int = 0):
    """(ttfont, glyphs) for a writer from their isolated letters only (copy 0 of each).

    Joined forms come from the model (``versions`` drawn versions each, the
    font cycling through them) or, with ``model`` None, are the template
    skeletons alone (one version). Isolated letters are the writer's own ink.
    """
    import torch

    from ahawp_font import versions_font
    from ahawp_forms import specs_by_name
    from skeleton_moves import load_templates

    moves, inks = load_cache()
    pca = load_pca()
    temps = load_templates()
    xs = encoded(moves, pca)
    specs = specs_by_name() | {s.name: s for s in ahawp.EXTRA.values()}
    iso = {specs[n]: inks[writer][n][min(inks[writer][n])].astype(np.float32)
           for n in isolated_names(pca) if n in inks[writer] and n in specs}
    radius = ahawp.pen_radius(iso)
    joined = [n for n in form_names(pca) if "." in n and n in specs]
    if model is None:
        drawn = {n: np.zeros((1, K)) for n in joined}
    else:
        from hfont.models.variation import VariationModel

        torch.manual_seed(seed)
        net = VariationModel.from_checkpoint(torch.load(model)).eval()
        names = form_names(pca)
        ix, iform, imask = style_inputs(xs[writer], pca, lambda d: d[min(d)])
        t = torch.from_numpy
        b = len(joined)
        with torch.no_grad():
            cond = net.condition(torch.tensor([names.index(n) for n in joined]), t(np.repeat(ix[None], b, 0)),
                                 t(np.repeat(iform[None], b, 0)), t(np.repeat(imask[None], b, 0)))
        drawn = dict(zip(joined, net.sample(cond, versions, temperature).numpy()))
    sets = []
    for v in range(len(next(iter(drawn.values())))):
        letters = dict(iso) if v == 0 else {}
        for n in joined:
            pts = temps[n][0] + decode(pca, n, drawn[n][v])
            letters[specs[n]] = skeleton.render_points(pts, temps[n][1], radius)
        sets.append(letters)
    return versions_font(sets, f"AHAWP {writer} {'model' if model else 'template'}")


def sheet_preview(writer: str, model: str, versions: int, temperature: float, out: Path) -> None:
    """Words set in three fonts for one writer: their real full sheet, template only, model.

    An image of real handwriting (and drawn from it): ``out`` must be under
    ``output/``, which git ignores.
    """
    from PIL import Image, ImageDraw

    from ahawp_font import COMPOSED_WORDS, WORDS, render, writer_font

    archive = ahawp.open_archive(ARCHIVE)
    fonts = [("real full sheet (66 forms written)", writer_font(archive, ahawp.index(archive)[writer],
                                                                 f"AHAWP {writer}", copy=0)[0]),
             ("18 isolated letters + template", sheet_font(writer, None, 1, temperature)[0]),
             (f"18 isolated letters + model ({versions} versions)", sheet_font(writer, model, versions,
                                                                             temperature)[0])]
    text = [" ".join(WORDS[:3]), " ".join(WORDS[3:]), " ".join(COMPOSED_WORDS[:5]), "ببببب سسسس ممممم"]
    panels = []
    for label, font in fonts:
        buffer = io.BytesIO()
        font.save(buffer)
        im = Image.fromarray(render(buffer.getvalue(), text)).convert("RGB")
        head = Image.new("RGB", (im.width, 34), "white")
        ImageDraw.Draw(head).text((10, 8), label, fill=(200, 0, 0))
        panels += [head, im]
    width = max(p.width for p in panels)
    sheet = Image.new("RGB", (width, sum(p.height for p in panels)), "white")
    y = 0
    for p in panels:
        sheet.paste(p, (width - p.width, y))
        y += p.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    log.info("preview -> %s", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copies")
    c.add_argument("--workers", type=int, default=7)
    sub.add_parser("pca")
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--no-style", action="store_true")
    t.add_argument("--beta", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--letter-dropout", type=float, default=0.3,
                   help="share of isolated letters hidden from the style encoder per training example")
    t.add_argument("--style-dim", type=int, default=32)
    s = sub.add_parser("score")
    s.add_argument("--split", default="val")
    s.add_argument("--methods", default="template,gauss,other,own")
    s.add_argument("--model", default=None)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--workers", type=int, default=7)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out", type=Path, default=None, help="add every form's scores to this JSON")
    p = sub.add_parser("preview")
    p.add_argument("writer")
    p.add_argument("--model", default="runs/v1_drop/best.pt")
    p.add_argument("--versions", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "preview":
        if "output" not in args.out.parts:
            raise SystemExit("a preview shows handwriting: write it under output/ (git-ignored)")
        sheet_preview(args.writer, args.model, args.versions, args.temperature, args.out)
    elif args.cmd == "copies":
        copies(args.workers)
    elif args.cmd == "pca":
        fit_pca(load_cache()[0])
    elif args.cmd == "train":
        train(args.out, args.steps, args.batch, not args.no_style, args.beta, args.seed, args.letter_dropout,
              args.style_dim)
    else:
        methods = [m for m in args.methods.split(",") if m]
        score(args.split, methods, args.model, args.temperature, args.workers, args.limit, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
