"""Train the stroke model (step A3) on a packed cache with strokes.

The cache needs ``strokes.npz`` (``scripts/stroke_cache.py``). Each sample is
one letter of one hand (font or SD19 writer) in the ``train`` split: the
letter's pen rows are the target, the same letter in the plain content font
is the content, and 8-30 of the hand's other letters are the references.
Validation loss is on the ``val`` split; the score that decides is
``scripts/sd19_style.py --strokes`` (centreline F1 on SD19 validation, G3:
0.419).

    python scripts/train_strokes.py --cache data/cache_sd19 --out runs/s1 --steps 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stroke_cache import load as load_strokes  # noqa: E402

from hfont.data.dataset import GlyphStore  # noqa: E402
from hfont.generate import content_images_from_font  # noqa: E402
from hfont.models.strokegen import StrokeConfig, StrokeGenerator, rows_for_training  # noqa: E402

log = logging.getLogger("train_strokes")
CONTENT_FONT = ROOT / "data/fonts/NotoSans[wdth,wght].ttf"


def ref_count(rng: np.random.Generator, refs: tuple[int, int], available: int) -> int:
    """How many reference letters to show: ``refs`` (low, high), as far as the hand has them.

    A hand is admitted for having enough letters, but the pool for one target
    leaves out that letter's look-alikes, so it can hold fewer than ``low``.
    """
    high = min(refs[1], available)
    return int(rng.integers(min(refs[0], high), high + 1))


class StrokeData(Dataset):
    """Random (content, refs, rows) samples from one split of a packed cache."""

    def __init__(self, cache: str | Path, split: str, refs: tuple[int, int] = (8, 30),
                 max_len: int = 256, length: int = 10 ** 7, seed: int = 0,
                 categories: set[str] | None = None) -> None:
        self.store = GlyphStore(cache)
        self.points, self.offsets, self.step = load_strokes(cache)
        with np.load(Path(cache) / "packed_index.npz") as idx:
            self.start = {str(f): int(s) for f, s in zip(idx["font_ids"], idx["starts"])}
        self.content = content_images_from_font(CONTENT_FONT, self.store.charset)
        self.refs, self.max_len, self.length, self.seed = refs, max_len, length, seed
        self.hands = []
        for r in self.store.split(split):
            if categories and r.category not in categories:
                continue
            keys = [k for k in self.store.keys(r.font_id)
                    if k in self.content and self._n(r.font_id, k) + 1 <= max_len
                    and self._n(r.font_id, k) > 1]
            if len(keys) > refs[0]:
                self.hands.append((r.font_id, keys))
        if not self.hands:
            raise ValueError(f"no usable hands in split {split!r}")
        self.rng = np.random.default_rng(seed)

    def _row(self, font_id: str, key: str) -> int:
        return self.start[font_id] + self.store._load(font_id).index[key]

    def _n(self, font_id: str, key: str) -> int:
        i = self._row(font_id, key)
        return int(self.offsets[i + 1] - self.offsets[i])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _):
        while True:
            font_id, keys = self.hands[self.rng.integers(len(self.hands))]
            key = keys[self.rng.integers(len(keys))]
            pool = [k for k in self.store.distinct_keys(font_id, key)
                    if k in self.store._load(font_id).index]
            if pool:
                break
        i = self._row(font_id, key)
        rows = rows_for_training(self.points[self.offsets[i]:self.offsets[i + 1]].astype(np.float32),
                                 self.step, self.max_len)
        k = ref_count(self.rng, self.refs, len(pool))
        picks = self.rng.choice(len(pool), size=k, replace=False)
        refs = np.stack([self.store.image(font_id, pool[j]) for j in picks])
        return self.content[key], refs, rows


def fixed_batches(dataset, batch: int) -> DataLoader:
    """The same samples on every pass: validation restarts its draw from its seed.

    Without this each pass drew from where the last left off, and the loss
    moved by more between passes (-3.17, -2.65, -3.04 at steps 6-10K) than
    the model did.
    """
    dataset.rng = np.random.default_rng(dataset.seed)
    return DataLoader(dataset, batch, collate_fn=collate)


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.dataset.seed * 1000 + worker_id + int(time.time()))


def collate(batch):
    content = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
    k = max(len(b[1]) for b in batch)
    refs = torch.zeros(len(batch), k, 1, *batch[0][1].shape[1:])
    ref_mask = torch.zeros(len(batch), k, dtype=torch.bool)
    t = max(len(b[2]) for b in batch)
    rows = torch.zeros(len(batch), t, 3)
    mask = torch.zeros(len(batch), t, dtype=torch.bool)
    for j, (_, r, s) in enumerate(batch):
        refs[j, :len(r), 0] = torch.from_numpy(r)
        ref_mask[j, :len(r)] = True
        rows[j, :len(s)] = torch.from_numpy(s)
        mask[j, :len(s)] = True
    return content, refs, ref_mask, rows, mask


def shrink(images: torch.Tensor, size: int) -> torch.Tensor:
    """Letters at the encoders' size (area average keeps thin strokes)."""
    shape = images.shape
    flat = images.reshape(-1, 1, *shape[-2:])
    if shape[-1] != size:
        flat = F.adaptive_avg_pool2d(flat, size)
    return flat.reshape(*shape[:-2], size, size)


def run_batch(model, batch, device, size):
    content, refs, ref_mask, rows, mask = (b.to(device, non_blocking=True) for b in batch)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(shrink(content, size), shrink(refs, size), ref_mask, rows)
    return model.loss(out, rows, mask)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--sd19-only", action="store_true", help="train on SD19 writers only, no fonts")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = {"SD19"} if args.sd19_only else None
    config = StrokeConfig(d_model=args.d_model, layers=args.layers, position=True)
    train = StrokeData(args.cache, "train", max_len=config.max_len, categories=cats)
    val = StrokeData(args.cache, "val", max_len=config.max_len, length=args.batch * 20, seed=1,
                     categories=cats)
    log.info("train hands %d, val hands %d", len(train.hands), len(val.hands))
    config.step = train.step
    loader = DataLoader(train, args.batch, num_workers=args.workers, collate_fn=collate,
                        worker_init_fn=worker_init, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")
    model = StrokeGenerator(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    step, best = 0, math.inf
    if args.resume:
        data = torch.load(args.resume, map_location=device)
        model.load_state_dict(data["model"])
        opt.load_state_dict(data["opt"])
        step, best = data["step"], data.get("best", math.inf)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    for _ in range(step):
        sched.step()
    log.info("%.1fM parameters on %s", sum(p.numel() for p in model.parameters()) / 1e6, device)

    t0, running = time.time(), []
    for batch in loader:
        if step >= args.steps:
            break
        model.train()
        losses = run_batch(model, batch, device, config.image_size)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        running.append([losses[k].item() for k in ("loss", "move", "pen")])
        if step % args.log_every == 0:
            m = np.mean(running, axis=0)
            log.info("step %d loss %.3f move %.3f pen %.3f lr %.2e (%.2fs/step)", step, *m,
                     sched.get_last_lr()[0], (time.time() - t0) / args.log_every)
            running, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vals = [run_batch(model, b, device, config.image_size)["loss"].item()
                        for b in fixed_batches(val, args.batch)]
            v = float(np.mean(vals))
            state = {**model.checkpoint(), "opt": opt.state_dict(), "step": step, "best": min(best, v)}
            torch.save(state, out / "last.pt")
            if v < best:
                best = v
                torch.save(model.checkpoint(), out / "best.pt")
            log.info("step %d val loss %.3f (best %.3f)", step, v, best)
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val": v}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
