"""Several templates per form: the variation model chooses how a letter is built (step A6b).

One template per form fixes a letter's structure: a writer whose medial
seen has three clear teeth, or whose medial ain is a closed loop, can only
be approximated by moving a template that is built differently. Here each
form gets up to ``K`` templates, the medoids of clusters of the training
writers' skeletons (k-medoids on symmetric chamfer distance), and every
copy is described by the template it fits best plus its moves on it.

    python scripts/variation_mix.py templates     # data/ahawp/templates_mix.npz
    python scripts/variation_mix.py copies        # data/ahawp/variation_mix.npz
    python scripts/variation_mix.py pca           # data/ahawp/variation_mix_pca.npz

A "pair" below is a form with one of its templates, named ``<form>#<t>``.
All files are derived from handwriting and stay in ``data/ahawp/``.
"""

from __future__ import annotations

import argparse
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

log = logging.getLogger("variation_mix")
ARCHIVE = ROOT / "data/ahawp/isolated_alphabets_per_user.zip"
TEMPLATES = ROOT / "data/ahawp/templates_mix.npz"
CACHE = ROOT / "data/ahawp/variation_mix.npz"
PCA = ROOT / "data/ahawp/variation_mix_pca.npz"
K_TEMPLATES = 3
MIN_CLUSTER = 6          # training writers a template must stand for
K = 24


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    return float(d.min(1).mean() + d.min(0).mean())


def kmedoids(dist: np.ndarray, k: int, min_size: int, iterations: int = 50) -> list[int]:
    """Medoid indices of up to ``k`` clusters (PAM-style swaps); clusters under ``min_size`` are dropped."""
    n = len(dist)
    medoids = [int(np.argmin(dist.sum(1)))]
    while len(medoids) < min(k, n):                       # greedy start: farthest-first by gain
        gain = [(dist[:, medoids].min(1) - np.minimum(dist[:, medoids].min(1), dist[:, j])).sum()
                if j not in medoids else -1 for j in range(n)]
        medoids.append(int(np.argmax(gain)))
    for _ in range(iterations):
        label = np.argmin(dist[:, medoids], axis=1)
        new = []
        for c in range(len(medoids)):
            members = np.nonzero(label == c)[0]
            new.append(int(members[np.argmin(dist[np.ix_(members, members)].sum(1))]) if members.size else medoids[c])
        if new == medoids:
            break
        medoids = new
    while True:
        label = np.argmin(dist[:, medoids], axis=1)
        sizes = np.bincount(label, minlength=len(medoids))
        if len(medoids) == 1 or sizes.min() >= min_size:
            return medoids
        medoids.pop(int(np.argmin(sizes)))


def _cluster_form(args) -> tuple[str, list]:
    name, by_writer = args
    writers = sorted(by_writer)
    dist = np.array([[_chamfer(by_writer[a][0], by_writer[b][0]) if a != b else 0.0 for b in writers]
                     for a in writers])
    return name, [(writers[m], *by_writer[writers[m]]) for m in kmedoids(dist, K_TEMPLATES, MIN_CLUSTER)]


def templates(workers: int) -> None:
    from ahawp_forms import load
    from skeleton_moves import _points_of

    data = load()
    train = sorted(w for w in data if ahawp.split_of(w) == "train")
    points = _points_of(data, train)
    arrays, t0 = {}, time.time()
    with ProcessPoolExecutor(workers) as pool:
        for name, chosen in pool.map(_cluster_form, sorted(points.items())):
            for t, (writer, pts, ids) in enumerate(chosen):
                arrays[f"{name}#{t}|points"], arrays[f"{name}#{t}|ids"] = pts, ids
            log.info("%-14s %d templates (%s)", name, len(chosen), ", ".join(c[0] for c in chosen))
    np.savez_compressed(TEMPLATES, **arrays)
    log.info("%d pairs -> %s (%.0fs)", len(arrays) // 2, TEMPLATES, time.time() - t0)


def load_templates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """pair name ``form#t`` -> (points, stroke ids)."""
    with np.load(TEMPLATES) as z:
        names = {k.split("|")[0] for k in z.files}
        return {n: (z[f"{n}|points"], z[f"{n}|ids"]) for n in names}


def pairs_of(temps: dict) -> dict[str, list[str]]:
    """form -> its pair names, in template order."""
    out: dict[str, list[str]] = {}
    for p in sorted(temps, key=lambda p: (p.split("#")[0], int(p.split("#")[1]))):
        out.setdefault(p.split("#")[0], []).append(p)
    return out


def best_pair(pairs: list[str], temps: dict, target: np.ndarray) -> str:
    """The template a copy is built like: smallest chamfer after each template's affine fit."""
    if len(pairs) == 1:
        return pairs[0]
    cost = []
    for p in pairs:
        tpl = temps[p][0]
        b, t = skeleton.affine(tpl, target)
        cost.append(_chamfer(tpl @ b.T + t, target))
    return pairs[int(np.argmin(cost))]


def _copies_of_writer(args) -> dict[str, np.ndarray]:
    writer, temps = args
    by_form = pairs_of(temps)
    archive = ahawp.open_archive(ARCHIVE)
    keep_ink = ahawp.split_of(writer) != "train"
    out = {}
    for spec, files in ahawp.index(archive)[writer].items():
        if spec.name not in by_form:
            continue
        for c, f in enumerate(files):
            ink = ahawp.ink(archive.read(f))
            if ink.sum() < 20:
                continue
            got = skeleton.skeleton_points(ink)
            if got is None:
                continue
            pair = best_pair(by_form[spec.name], temps, got[0])
            tpl = temps[pair][0]
            key = f"{writer}|{pair}|{c}"
            out[f"{key}|moves"] = (skeleton.register(tpl, got[0], beta=4.0, smooth=0.5) - tpl).astype(np.float32)
            if keep_ink:
                rows, cols = np.nonzero(ink > 0.5)
                out[f"{key}|ink"] = (ink > 0.5)[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return out


def copies(workers: int) -> None:
    temps = load_templates()
    writers = sorted(ahawp.index(ahawp.open_archive(ARCHIVE)))
    t0, arrays = time.time(), {}
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(_copies_of_writer, [(w, temps) for w in writers]), 1):
            arrays.update(part)
            if i % 5 == 0:
                log.info("%d/%d writers (%.0fs)", i, len(writers), time.time() - t0)
    np.savez_compressed(CACHE, **arrays)
    log.info("%d copies -> %s (%.0fs)", sum(k.endswith("|moves") for k in arrays), CACHE, time.time() - t0)


# ------------------------------------------------------------------ data in the model's terms
def _v():
    """The single-template pipeline's helpers, pointed at this cache and PCA."""
    import variation as v

    v.CACHE, v.PCA = CACHE, PCA
    return v


def fit_pca() -> None:
    """Per pair: mean, components and spreads on the training copies that use it.

    A template few copies use (6 to 49 for 21 of 191 pairs) gets only as
    many components as a third of its copies support; the rest are zero, so
    that template varies less instead of fitting noise.
    """
    from sklearn.decomposition import PCA as Pca

    moves, _ = _v().load_cache()
    arrays = {}
    for name in sorted({p for m in moves.values() for p in m}):
        rows = np.stack([d.ravel() for w, m in moves.items() if ahawp.split_of(w) == "train" and name in m
                         for d in m[name].values()])
        k = max(1, min(K, len(rows) // 3))
        pca = Pca(n_components=k).fit(rows)
        comps = np.zeros((K, rows.shape[1]), np.float32)
        scale = np.ones(K, np.float32)
        comps[:k], scale[:k] = pca.components_, np.sqrt(pca.explained_variance_)
        arrays[f"{name}|mean"] = pca.mean_.astype(np.float32)
        arrays[f"{name}|comps"] = comps
        arrays[f"{name}|scale"] = scale
    np.savez_compressed(PCA, **arrays)
    log.info("%d pairs -> %s", len({k.split('|')[0] for k in arrays}), PCA)


class Data:
    """Everything the model needs: pairs, their components, and each writer's copies as x."""

    def __init__(self):
        v = _v()
        moves, pair_inks = v.load_cache()
        # writer -> base form -> {copy: ink}: the real copies of val/test writers
        self.inks: dict[str, dict[str, dict]] = {}
        for w, wm in pair_inks.items():
            for p, cs in wm.items():
                self.inks.setdefault(w, {}).setdefault(p.split("#")[0], {}).update(cs)
        self.pca = v.load_pca()
        self.temps = load_templates()
        self.pairs = sorted(self.pca)                                  # model form ids
        by_form = pairs_of({p: None for p in self.pairs})
        self.bases = sorted(by_form)                                   # base forms
        self.base_of = np.array([self.bases.index(p.split("#")[0]) for p in self.pairs])
        self.t_of = np.array([int(p.split("#")[1]) for p in self.pairs])
        self.valid = np.zeros((len(self.bases), K_TEMPLATES), bool)
        for p in self.pairs:
            self.valid[self.bases.index(p.split("#")[0]), int(p.split("#")[1])] = True
        self.iso = [b for b in self.bases if "." not in b]
        self.joined = [b for b in self.bases if "." in b]
        xs = v.encoded(moves, self.pca)                                # writer -> pair -> {copy: x}
        # writer -> base form -> [(pair id, x)] over all its copies
        self.copies: dict[str, dict[str, list]] = {}
        for w, wm in xs.items():
            for p, cs in wm.items():
                b = p.split("#")[0]
                for c, x in sorted(cs.items()):
                    self.copies.setdefault(w, {}).setdefault(b, []).append((self.pairs.index(p), c, x))

    def style_inputs(self, writer: str, pick) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The writer's isolated letters, one copy each (``pick`` chooses among (pair, copy, x))."""
        x = np.zeros((len(self.iso), K), np.float32)
        pid = np.zeros(len(self.iso), np.int64)
        mask = np.zeros(len(self.iso), bool)
        for i, b in enumerate(self.iso):
            got = self.copies.get(writer, {}).get(b)
            if got:
                p, _, xi = pick(got)
                x[i], pid[i], mask[i] = xi, p, True
        return x, pid, mask

    def points(self, pair: int, x: np.ndarray) -> np.ndarray:
        name = self.pairs[pair]
        return self.temps[name][0] + _v().decode(self.pca, name, x)


def first_copy(got):
    return min(got, key=lambda r: r[1])


# ------------------------------------------------------------------ a new writer (the live demo)
_TEMPLATES = None


def templates_once() -> "Templates":
    """``Templates()``, loaded the first time and kept: the live demo draws every sheet with it."""
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = Templates()
    return _TEMPLATES


class Templates:
    """The pairs, their components and templates, without the copies cache and without
    repointing ``variation``'s files (``_v``), so the single-template model can run alongside."""

    def __init__(self):
        with np.load(PCA) as z:
            names = {k.split("|")[0] for k in z.files}
            self.pca = {n: (z[f"{n}|mean"], z[f"{n}|comps"], z[f"{n}|scale"]) for n in names}
        self.temps = load_templates()
        self.pairs = sorted(self.pca)
        self.by_form = pairs_of({p: None for p in self.pairs})
        self.bases = sorted(self.by_form)
        self.valid = np.zeros((len(self.bases), K_TEMPLATES), bool)
        for p in self.pairs:
            self.valid[self.bases.index(p.split("#")[0]), int(p.split("#")[1])] = True
        self.iso = [b for b in self.bases if "." not in b]


def _fit_pair_job(job):
    """One isolated letter: (its templates [(pair, points)], ink) -> (pair it is built like, moves), or
    None. Module level, so a process pool can run it."""
    candidates, ink = job
    got = skeleton.skeleton_points(np.asarray(ink) > 0.5)
    if got is None:
        return None
    cost = []
    for _, tpl in candidates:
        b, t = skeleton.affine(tpl, got[0])
        cost.append(_chamfer(tpl @ b.T + t, got[0]))
    pair, tpl = candidates[int(np.argmin(cost))]
    return pair, skeleton.register(tpl, got[0], beta=4.0, smooth=0.5) - tpl


def photo_style(iso: dict, tp: Templates, map_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A new writer's isolated letters (spec -> ink) as this model's style input: each letter
    registered to the template it is built like (as ``copies`` does), rescaled first into the
    training writers' size range as the single-template pipeline does. Returns (x, pair ids,
    mask, size factor)."""
    import variation as v

    factor = v.size_factor(iso)
    iso = v.resize_letters(iso, factor)
    by_name = {spec.name: ink for spec, ink in iso.items()}
    x = np.zeros((len(tp.iso), K), np.float32)
    pid = np.zeros(len(tp.iso), np.int64)
    mask = np.zeros(len(tp.iso), bool)
    letters = [i for i, b in enumerate(tp.iso) if b in by_name]
    jobs = [([(p, tp.temps[p][0]) for p in tp.by_form[tp.iso[i]]], by_name[tp.iso[i]]) for i in letters]
    for i, got in zip(letters, (map_fn or map)(_fit_pair_job, jobs)):
        if got is not None:
            pair, moves = got
            x[i], pid[i], mask[i] = v.encode(tp.pca, pair, moves), tp.pairs.index(pair), True
    return x, pid, mask, factor


def photo_versions(iso: dict, model: str, forms: dict[str, int | None], n: int = 4, temperature: float = 0.4,
                   seed: int = 0, tp: Templates | None = None, map_fn=None) -> dict[str, tuple[int, np.ndarray, list]]:
    """For each joined form, ``n`` versions drawn from template ``forms[form]`` (its index; None:
    the one the model's choice head likes best for this writer). Returns form -> (template index,
    the choice head's probabilities over the form's templates, [(points, stroke ids)])."""
    import torch

    import variation as v
    from hfont.models.variation import VariationModel

    tp = tp or templates_once()
    x, pid, mask, factor = photo_style(iso, tp, map_fn)
    torch.manual_seed(seed)
    net = VariationModel.from_checkpoint(torch.load(model)).eval()
    t = torch.from_numpy
    out = {}
    with torch.no_grad():
        s = net.style_vector(t(x[None]), t(pid[None]), t(mask[None]))
        for form, chosen in forms.items():
            b = tp.bases.index(form)
            probs = torch.softmax(net.choice_logits(torch.tensor([b]), s, t(tp.valid[b:b + 1])), -1)[0].numpy()
            k = int(np.argmax(probs)) if chosen is None else int(chosen)
            pair = f"{form}#{k}"
            cond = net.condition(torch.tensor([tp.pairs.index(pair)] * n), None, None, None, style=s.expand(n, -1))
            xs = net.sample(cond, 1, temperature)[:, 0].numpy()
            drawn = [((tp.temps[pair][0] + v.decode(tp.pca, pair, xi)) / factor, tp.temps[pair][1]) for xi in xs]
            out[form] = (k, probs[:len(tp.by_form[form])], drawn)
    return out


# ------------------------------------------------------------------ training
def train(out: Path, steps: int, batch: int, style: bool, seed: int, letter_dropout: float) -> None:
    import torch
    import torch.nn.functional as F

    from hfont.models.variation import VariationConfig, VariationModel
    from variation import drop_letters

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = Data()
    split = {s: sorted(w for w in d.copies if ahawp.split_of(w) == s) for s in ("train", "val")}
    items = {s: [(w, b, i) for w in split[s] for b in d.copies[w] for i in range(len(d.copies[w][b]))]
             for s in split}
    model = VariationModel(VariationConfig(forms=len(d.pairs), k=K, style=style, templates=K_TEMPLATES,
                                           base_forms=len(d.bases)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    valid = torch.from_numpy(d.valid)
    t = torch.from_numpy

    def make(rows, r):
        got = [d.copies[w][b][i] for w, b, i in rows]
        pair = np.array([g[0] for g in got])
        sty = [d.style_inputs(w, lambda c: c[r.integers(len(c))]) for w, _, _ in rows]
        return (t(np.stack([g[2] for g in got]).astype(np.float32)), t(pair), t(d.base_of[pair]),
                t(d.t_of[pair]), t(np.stack([s[0] for s in sty])), t(np.stack([s[1] for s in sty])),
                t(np.stack([s[2] for s in sty])))

    def step_loss(b):
        x, pair, base, tmpl, ix, ip, im = b
        s = model.style_vector(ix, ip, im) if style else None
        vae, parts = model.loss(x, model.condition(pair, None, None, None, style=s))
        ce = F.cross_entropy(model.choice_logits(base, s, valid[base]), tmpl)
        return vae + ce, {**parts, "choice": float(ce)}

    vrng = np.random.default_rng(1)
    val_batch = make([items["val"][i] for i in vrng.choice(len(items["val"]), 4096, replace=False)], vrng)
    out.mkdir(parents=True, exist_ok=True)
    best, t0 = float("inf"), time.time()
    for step in range(1, steps + 1):
        b = list(make([items["train"][i] for i in rng.integers(len(items["train"]), size=batch)], rng))
        if letter_dropout and style:
            b[6] = t(drop_letters(b[6].numpy(), letter_dropout, rng))
        loss, parts = step_loss(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vloss, vparts = step_loss(val_batch)
            model.train()
            log.info("step %d  train %.2f  val %.2f (rec %.2f kl %.2f choice %.3f)  %.0fs", step, float(loss),
                     float(vloss), vparts["rec"], vparts["kl"], vparts["choice"], time.time() - t0)
            if float(vloss) < best:
                best = float(vloss)
                torch.save(model.checkpoint(), out / "best.pt")
    log.info("best val loss %.3f -> %s", best, out / "best.pt")


# ------------------------------------------------------------------ versions and scoring
def model_versions(path: str, d: Data, writers: list[str], n: int, temperature: float,
                   choice: str = "sample", seed: int = 0) -> dict:
    """writer -> joined base form -> [(pair id, x)] * n, from a trained mixture model."""
    import torch

    from hfont.models.variation import VariationModel

    torch.manual_seed(seed)
    model = VariationModel.from_checkpoint(torch.load(path)).eval()
    valid = torch.from_numpy(d.valid)
    t = torch.from_numpy
    out = {}
    for w in writers:
        ix, ip, im = d.style_inputs(w, first_copy)
        base = torch.tensor([d.bases.index(b) for b in d.joined])
        with torch.no_grad():
            s = model.style_vector(t(ix[None]), t(ip[None]), t(im[None])).expand(len(base), -1) \
                if model.config.style else None
            probs = torch.softmax(model.choice_logits(base, s, valid[base]), -1)
            if choice == "argmax":
                tmpl = probs.argmax(-1, keepdim=True).expand(-1, n)
            else:
                tmpl = torch.multinomial(probs, n, replacement=True)
            out[w] = {}
            for j, b in enumerate(d.joined):
                pairs = [d.pairs.index(f"{b}#{int(k)}") for k in tmpl[j]]
                cond = model.condition(torch.tensor(pairs), None, None, None,
                                       style=s[j:j + 1].expand(n, -1) if s is not None else None)
                xs = model.sample(cond, 1, temperature)[:, 0].numpy()
                out[w][b] = list(zip(pairs, xs))
    return out


def baseline_versions(d: Data, writers: list[str], n: int, temperature: float, kind: str) -> dict:
    """No network: templates drawn by how often training copies use them; x Gaussian (or zero)."""
    rng = np.random.default_rng(0)
    use = np.zeros(len(d.pairs))
    for w, wm in d.copies.items():
        if ahawp.split_of(w) == "train":
            for got in wm.values():
                for p, _, _ in got:
                    use[p] += 1
    out = {}
    for w in writers:
        out[w] = {}
        for b in d.joined:
            ids = [i for i, p in enumerate(d.pairs) if p.split("#")[0] == b]
            prob = use[ids] / use[ids].sum()
            if kind == "template":
                out[w][b] = [(ids[int(np.argmax(prob))], np.zeros(K))]
            else:
                out[w][b] = [(ids[rng.choice(len(ids), p=prob)], rng.standard_normal(K) * temperature)
                             for _ in range(n)]
    return out


def score(split: str, methods: list[str], model: str | None, temperature: float, choice: str, workers: int,
          out: Path | None) -> None:
    import json

    import variation as v1

    d = Data()
    inks = d.inks
    writers = sorted(w for w in inks if ahawp.split_of(w) == split)
    radius = {w: ahawp.pen_radius({b: inks[w][b][min(inks[w][b])].astype(np.float32)
                                   for b in d.iso if b in inks[w]}) for w in writers}
    runs = [(m, baseline_versions(d, writers, 5, temperature, m)) for m in methods]
    if model:
        runs.append((f"model:{model}:{temperature}:{choice}", model_versions(model, d, writers, 5, temperature, choice)))
    for label, versions in runs:
        jobs = []
        for i, w in enumerate(writers):
            drawn = {b: [skeleton.render_points(d.points(p, x), d.temps[d.pairs[p]][1], radius[w]) for p, x in vs]
                     for b, vs in versions[w].items() if b in inks[w]}
            jobs.append((w, drawn, inks[w], inks[writers[(i + 1) % len(writers)]], radius[w]))
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool:
            rows = [r for part in pool.map(v1._score_writer, jobs) for r in part]
        m = {k: np.mean([r[k] for r in rows]) for k in ("realism", "coverage", "other")}
        print(f"{label:40} {split} {len(writers)} writers {len(rows)} forms: realism {m['realism']:.3f}  "
              f"coverage {m['coverage']:.3f}  other {m['other']:.3f}  (spec {m['realism'] - m['other']:+.3f})  "
              f"{time.time() - t0:.0f}s", flush=True)
        if out:
            done = json.loads(out.read_text()) if out.exists() else {}
            done[f"mix:{label}"] = rows
            out.write_text(json.dumps(done))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("templates", "copies"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=7)
    sub.add_parser("pca")
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--no-style", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--letter-dropout", type=float, default=0.3)
    s = sub.add_parser("score")
    s.add_argument("--split", default="val")
    s.add_argument("--methods", default="template,gauss")
    s.add_argument("--model", default=None)
    s.add_argument("--temperature", type=float, default=0.4)
    s.add_argument("--choice", choices=["sample", "argmax"], default="sample")
    s.add_argument("--workers", type=int, default=7)
    s.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "templates":
        templates(args.workers)
    elif args.cmd == "copies":
        copies(args.workers)
    elif args.cmd == "pca":
        fit_pca()
    elif args.cmd == "train":
        train(args.out, args.steps, args.batch, not args.no_style, args.seed, args.letter_dropout)
    else:
        score(args.split, [m for m in args.methods.split(",") if m], args.model, args.temperature, args.choice,
              args.workers, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
