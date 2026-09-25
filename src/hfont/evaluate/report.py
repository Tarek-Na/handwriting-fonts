"""The Phase 1 gate report.

Produces the numbers the README's gate is written against, with the baselines
needed to read them. A score is only meaningful next to what doing nothing
achieves, so every metric is reported beside the *content-copy* baseline: the
same glyph drawn in the neutral content font, ignoring the style references
entirely. A model that does not clearly beat that has not learned style
transfer, whatever its absolute IoU.

Two kinds of measurement live here:

* **Held-out generation** on corpus fonts: every non-seed glyph is generated
  from the 24 seed glyphs and compared with the font's real glyph.
* **Leave-one-out on the seed letters**: each seed letter is generated from the
  other 23 and compared with the real one. This is the only quantitative check
  available on real handwriting, where only the seed letters have ground truth
  — and running it on corpus fonts too puts the corpus-vs-handwriting domain
  gap on one scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..charset import seed_charset
from .metrics import cl_dice, iou, ink_coverage_ratio, ssim, tolerant_f1

log = logging.getLogger(__name__)


def _score(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "iou": iou(pred, target),
        "tol_f1": tolerant_f1(pred, target),
        # Scale-invariant, unlike tol_f1: identical glyphs scored at full and at
        # half size give clDice 0.581 both times, while tol_f1's fixed 1.5 px
        # tolerance moves it from 0.639 to 0.731. Reported alongside so a change
        # in intake scale cannot silently move the quality verdict.
        "cl_dice": cl_dice(pred, target),
        "ssim": ssim(pred, target),
        "coverage": ink_coverage_ratio(pred, target),
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


@dataclass
class FontResult:
    font_id: str
    category: str
    model: dict[str, float]
    baseline: dict[str, float]
    loo: dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def leave_one_out(model, reference_images: dict[str, np.ndarray],
                  content_images: dict[str, np.ndarray]) -> dict[str, float]:
    """Generate each reference glyph from the others and score it.

    Works on anything with a reference set — a corpus font's seed glyphs or a
    photographed template — because it never needs a glyph the writer did not
    provide.
    """
    from ..generate import _stack_refs

    keys = [k for k in reference_images if k in content_images]
    if len(keys) < 2:
        return {}
    glyph_index = {g.key: i for i, g in enumerate(model.charset)}
    device = model.device
    rows = []
    for held_out in keys:
        others = [reference_images[k] for k in keys if k != held_out]
        refs, mask = _stack_refs(others, device)
        attends = getattr(model.generator, "ref_attention", None) is not None
        if attends:
            style, ref_feats = model.generator.encode_style_full(refs, mask)
        else:
            style, ref_feats = model.generator.encode_style(refs, mask), None
        content = torch.from_numpy(content_images[held_out][None, None].astype(np.float32))
        content = content.to(device) * 2.0 - 1.0
        char = torch.tensor([glyph_index[held_out]], device=device)
        out = model.generator.decode(content, char, style, ref_feats,
                                     mask if attends else None)
        pred = ((out["image"].float() + 1) * 0.5).clamp(0, 1).cpu().numpy()[0, 0]
        rows.append(_score(pred, reference_images[held_out]))
    return _mean(rows)


def evaluate_split(
    checkpoint: str | Path,
    cache_dir: str | Path,
    split: str = "val",
    max_fonts: int | None = None,
    ref_seed: str = "seed24",
    with_loo: bool = True,
) -> list[FontResult]:
    from ..data.dataset import GlyphStore
    from ..generate import generate_rasters, load_checkpoint

    store = GlyphStore(cache_dir)
    model = load_checkpoint(checkpoint)
    records = store.split(split)
    if max_fonts:
        records = records[:max_fonts]

    seed_keys = [g.key for g in seed_charset(store.charset, ref_seed)]
    content_font = store.pick_content_font("train")
    content = {k: store.image(content_font, k) for k in store.keys(content_font)}

    results = []
    for i, record in enumerate(records):
        available = set(store.keys(record.font_id))
        refs = {k: store.image(record.font_id, k) for k in seed_keys if k in available}
        if not refs:
            continue
        predicted, _ = generate_rasters(model, refs, content)

        model_rows, base_rows = [], []
        for key in sorted(available - set(seed_keys)):
            if key not in predicted or key not in content:
                continue
            truth = store.image(record.font_id, key)
            model_rows.append(_score(predicted[key], truth))
            base_rows.append(_score(content[key], truth))

        results.append(FontResult(
            font_id=record.font_id,
            category=record.category,
            model=_mean(model_rows),
            baseline=_mean(base_rows),
            loo=leave_one_out(model, refs, content) if with_loo else {},
        ))
        if (i + 1) % 25 == 0:
            log.info("  evaluated %d/%d fonts", i + 1, len(records))
    return results


def summarize(results: list[FontResult]) -> dict:
    """Aggregate per-font results into the gate numbers, overall and by category."""

    def block(rs: list[FontResult]) -> dict:
        out = {"n_fonts": len(rs)}
        for part in ("model", "baseline", "loo"):
            rows = [getattr(r, part) for r in rs if getattr(r, part)]
            if not rows:
                continue
            out[part] = {}
            for metric in rows[0]:
                values = np.asarray([row[metric] for row in rows])
                out[part][metric] = {
                    "mean": float(values.mean()),
                    "p10": float(np.percentile(values, 10)),
                    "min": float(values.min()),
                }
        return out

    categories = sorted({r.category for r in results})
    return {
        "overall": block(results),
        "by_category": {c: block([r for r in results if r.category == c]) for c in categories},
    }


def format_summary(summary: dict) -> str:
    """Plain-text table of the headline numbers."""
    lines = []
    header = f"{'':14s} {'fonts':>5s} | {'IoU':>6s} {'tolF1':>6s} {'SSIM':>6s} {'cov':>5s} | " \
             f"{'base tolF1':>10s} | {'LOO tolF1':>9s}"
    lines.append(header)
    lines.append("-" * len(header))

    def row(name: str, b: dict) -> str:
        m, base, loo = b.get("model", {}), b.get("baseline", {}), b.get("loo", {})
        get = lambda d, k: d.get(k, {}).get("mean", float("nan"))
        return (f"{name:14s} {b['n_fonts']:5d} | {get(m, 'iou'):6.3f} {get(m, 'tol_f1'):6.3f} "
                f"{get(m, 'ssim'):6.3f} {get(m, 'coverage'):5.2f} | {get(base, 'tol_f1'):10.3f} | "
                f"{get(loo, 'tol_f1'):9.3f}")

    lines.append(row("ALL", summary["overall"]))
    for name, b in summary["by_category"].items():
        lines.append(row(name[:14], b))
    o = summary["overall"].get("model", {})
    if o:
        lines.append("")
        lines.append(f"p10 across fonts: IoU {o['iou']['p10']:.3f}  tolF1 {o['tol_f1']['p10']:.3f}  "
                     f"SSIM {o['ssim']['p10']:.3f}")
    return "\n".join(lines)


def write_report(results: list[FontResult], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "per_font.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=1), encoding="utf-8")
    (out_dir / "summary.txt").write_text(format_summary(summary), encoding="utf-8")
    return summary
