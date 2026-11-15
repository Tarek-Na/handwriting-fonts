"""Train the sheet reader's form classifier (``hfont.models.formclass``) on AHAWP.

    python scripts/train_formclass.py --out runs/c1 --epochs 12
    python scripts/train_formclass.py --eval runs/c1/best.pt --split test

Every copy of every form of the training writers is a sample (about 31,800);
validation is the val writers' copies, never trained on. Reports top-1 and
top-3 accuracy, and the two numbers the reader acts on: how often a correct
box would be flagged (false alarms) and how often a box holding another
form would pass (missed errors), for the flag rule ``p(label) < FLAG``: against a
random other form, and against the form the classifier finds most alike.
Needs ``scripts/train_forms.py canvas``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont.models.formclass import ClassConfig, FormClassifier, augment  # noqa: E402

log = logging.getLogger("train_formclass")
#: A box is flagged when the classifier gives its label less than this. On the val
#: writers: 3.1% of correct boxes flagged (about 2 a sheet), 9.1% of boxes holding
#: the most alike other form passed (0.05: 0.9% and 22.9%; 0.5: 7.2% and 3.3%).
FLAG = 0.2


def load(split: str):
    """(images uint8 (N,64,256), labels, writers, class names) for one split."""
    from train_forms import load_canvas

    writers = load_canvas(split)
    names = sorted({s.name for d in load_canvas("train").values() for s in d})
    xs, ys, ws = [], [], []
    for w, d in sorted(writers.items()):
        for spec, copies in d.items():
            if spec.name in names:
                xs.append(copies)
                ys += [names.index(spec.name)] * len(copies)
                ws += [w] * len(copies)
    return np.concatenate(xs), np.array(ys), np.array(ws), names


def predict(model, images: np.ndarray, device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).float()[:, None].to(device) / 255
            out.append(F.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out)


def report(prob: np.ndarray, y: np.ndarray, label: str) -> dict:
    top = np.argsort(-prob, 1)
    top1 = float((top[:, 0] == y).mean())
    top3 = float((top[:, :3] == y[:, None]).any(1).mean())
    false_alarm = float((prob[np.arange(len(y)), y] < FLAG).mean())
    # A box holding another form: each sample scored against a label it is not.
    rng = np.random.default_rng(0)
    wrong = (y + rng.integers(1, prob.shape[1], len(y))) % prob.shape[1]
    missed = float((prob[np.arange(len(y)), wrong] >= FLAG).mean())
    # The hardest case: the letter in a box labelled with the form most like it
    # (an initial noon in beh's box: the dots alone differ).
    other = prob.copy()
    other[np.arange(len(y)), y] = -1
    nearest = other.argmax(1)
    missed_near = float((prob[np.arange(len(y)), nearest] >= FLAG).mean())
    print(f"{label}: top-1 {top1:.3f}  top-3 {top3:.3f}  correct boxes flagged {false_alarm:.3f}  "
          f"wrong forms passed {missed:.3f} (random), {missed_near:.3f} (the most alike form)")
    return {"top1": top1, "top3": top3, "false_alarm": false_alarm, "missed": missed, "missed_near": missed_near}


def train(args) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, _, names = load("train")
    vx, vy, _, _ = load("val")
    log.info("train %d images, val %d, %d forms, on %s", len(y), len(vy), len(names), device)
    model = FormClassifier(ClassConfig(classes=len(names), width=args.width)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(y) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng, best = np.random.default_rng(0), 0.0
    for epoch in range(args.epochs):
        model.train()
        t0, order, losses = time.time(), rng.permutation(len(y)), []
        for i in range(0, len(order) - args.batch + 1, args.batch):
            idx = order[i:i + args.batch]
            xb = augment(torch.from_numpy(x[idx]).float()[:, None].to(device) / 255)
            loss = F.cross_entropy(model(xb), torch.from_numpy(y[idx]).to(device), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        acc = float((predict(model, vx, device).argmax(1) == vy).mean())
        log.info("epoch %d loss %.3f val top-1 %.3f (%.0fs)", epoch + 1, np.mean(losses), acc, time.time() - t0)
        torch.save(model.checkpoint(names), out / "last.pt")
        if acc > best:
            best = acc
            torch.save(model.checkpoint(names), out / "best.pt")


def evaluate(path: str, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names = FormClassifier.from_checkpoint(torch.load(path, map_location=device))
    model.to(device)
    x, y, w, split_names = load(split)
    assert split_names == names
    report(predict(model, x, device), y, f"{len(np.unique(w))} {split} writers, {len(y)} letters")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--eval", default=None, help="a checkpoint to score instead of training")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.eval:
        evaluate(args.eval, args.split)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
