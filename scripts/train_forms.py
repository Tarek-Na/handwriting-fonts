"""Train the positional-form model (``hfont.models.formgen``) on AHAWP.

    python scripts/train_forms.py canvas                     # data/ahawp/forms_canvas.npz
    python scripts/train_forms.py train --out runs/f1 --steps 20000

Each sample is one training writer's joined form: the input is a random
copy of the writer's isolated form of that letter, the style a random copy
of each of their isolated letters, the target a random copy of the form, so
the ten copies of every letter act as augmentation. Only ``train`` writers
(``ahawp.split_of``) are used. Validation loss is on the ``val`` writers;
the score that decides is ``scripts/ahawp_forms.py score --checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.models.formgen import FormConfig, FormGenerator, to_canvas  # noqa: E402

log = logging.getLogger("train_forms")
ARCHIVE = ROOT / "data/ahawp/isolated_alphabets_per_user.zip"
CANVAS = ROOT / "data/ahawp/forms_canvas.npz"
#: Every isolated glyph a writer gives: the 18 letters, alef with hamza, lam-alef.
ISOLATED = [GlyphSpec(c) for c in ahawp.LETTERS.values()] + [ahawp.EXTRA["alif_hamza"], ahawp.LAM_ALEF]
LETTER_ID = {spec.char: i for i, spec in enumerate(ISOLATED)}
FORM_ID = {f: i for i, f in enumerate(Form)}
JOINED = (Form.INIT, Form.MEDI, Form.FINA)


def _canvas_writer(writer: str) -> dict[str, np.ndarray]:
    archive = ahawp.open_archive(ARCHIVE)
    out = {}
    for spec, names in ahawp.index(archive)[writer].items():
        copies = [to_canvas(ahawp.ink(archive.read(n))) for n in names]
        copies = [c for c in copies if c.sum() >= 5]
        if copies:
            out[f"{writer}|{spec.name}"] = (np.stack(copies) * 255).round().astype(np.uint8)
    return out


def make_canvas(workers: int) -> None:
    writers = sorted(ahawp.index(ahawp.open_archive(ARCHIVE)))
    t0, arrays = time.time(), {}
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_canvas_writer, writers):
            arrays.update(part)
    np.savez_compressed(CANVAS, **arrays)
    log.info("%d writers, %d glyphs -> %s (%.0fs)", len(writers), len(arrays), CANVAS, time.time() - t0)


def load_canvas(split: str) -> dict[str, dict[GlyphSpec, np.ndarray]]:
    """writer -> glyph -> (copies, 64, 256) uint8, for one split's writers."""
    specs = {s.name: s for s in ISOLATED} | {GlyphSpec(c, f).name: GlyphSpec(c, f)
                                            for c in ahawp.LETTERS.values() for f in JOINED}
    out: dict[str, dict[GlyphSpec, np.ndarray]] = {}
    with np.load(CANVAS) as data:
        for key in data.files:
            writer, name = key.split("|")
            if ahawp.split_of(writer) == split and name in specs:
                out.setdefault(writer, {})[specs[name]] = data[key]
    return out


def inputs_for(letters: dict[GlyphSpec, np.ndarray], spec: GlyphSpec, pick) -> tuple:
    """(letter, refs, ref_mask, letter_id, form_id) for drawing ``spec``; ``pick(copies)`` chooses a copy."""
    letter = pick(letters[GlyphSpec(spec.char)])
    refs = np.zeros((len(ISOLATED), 64, 256), np.uint8)
    mask = np.zeros(len(ISOLATED), bool)
    for i, s in enumerate(ISOLATED):
        if s in letters:
            refs[i], mask[i] = pick(letters[s]), True
    return letter, refs, mask, LETTER_ID[spec.char], FORM_ID[spec.form]


class FormData(torch.utils.data.Dataset):
    def __init__(self, split: str, length: int = 10 ** 7, seed: int = 0) -> None:
        self.writers = load_canvas(split)
        self.samples = [(w, s) for w, d in sorted(self.writers.items()) for s in d
                        if s.form in JOINED and GlyphSpec(s.char) in d]
        self.length, self.seed = length, seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _):
        writer, spec = self.samples[self.rng.integers(len(self.samples))]
        letters = self.writers[writer]
        pick = lambda copies: copies[self.rng.integers(len(copies))]  # noqa: E731
        letter, refs, mask, lid, fid = inputs_for(letters, spec, pick)
        return letter, refs, mask, lid, fid, pick(letters[spec])


def collate(batch):
    f = lambda i: torch.from_numpy(np.stack([b[i] for b in batch]))  # noqa: E731
    img = lambda t: t.float()[:, None] / 255 if t.dim() == 3 else t.float()[:, :, None] / 255  # noqa: E731
    return (img(f(0)), img(f(1)), f(2), torch.tensor([b[3] for b in batch]),
            torch.tensor([b[4] for b in batch]), img(f(5)))


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.dataset.seed * 1000 + worker_id + int(time.time()))


def run_batch(model, batch, device):
    letter, refs, mask, lid, fid, target = (b.to(device, non_blocking=True) for b in batch)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(letter, refs, mask, lid, fid)
    return model.loss(logits.float(), target)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = FormData("train")
    val = FormData("val", length=args.batch * 10, seed=1)
    log.info("train: %d writers, %d forms; val: %d writers", len(data.writers), len(data.samples),
             len(val.writers))
    loader = torch.utils.data.DataLoader(data, args.batch, num_workers=args.workers, collate_fn=collate,
                                         worker_init_fn=worker_init, persistent_workers=args.workers > 0,
                                         pin_memory=device.type == "cuda")
    model = FormGenerator(FormConfig(width=args.width)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (
        1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    log.info("%.1fM parameters on %s", sum(p.numel() for p in model.parameters()) / 1e6, device)
    step, best, t0, running = 0, math.inf, time.time(), []
    for batch in loader:
        if step >= args.steps:
            break
        model.train()
        losses = run_batch(model, batch, device)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        running.append(losses["loss"].item())
        if step % args.log_every == 0:
            log.info("step %d loss %.4f lr %.2e (%.2fs/step)", step, np.mean(running),
                     sched.get_last_lr()[0], (time.time() - t0) / args.log_every)
            running, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val.rng = np.random.default_rng(val.seed)            # the same samples every pass
            with torch.no_grad():
                v = float(np.mean([run_batch(model, b, device)["loss"].item()
                                   for b in torch.utils.data.DataLoader(val, args.batch, collate_fn=collate)]))
            torch.save({**model.checkpoint(), "step": step}, out / "last.pt")
            if v < best:
                best = v
                torch.save({**model.checkpoint(), "step": step}, out / "best.pt")
            log.info("step %d val loss %.4f (best %.4f)", step, v, best)
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val": v}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canvas")
    c.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "canvas":
        make_canvas(args.workers)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
