"""Quantitative evaluation against held-out fonts.

The brief asks for one harness serving both phases so that the Phase 1 gate
rests on numbers comparable to Phase 2's. The metrics here are therefore
script-agnostic: nothing below knows what a Latin letter is.

A note on what these numbers are worth. Held-out typeset fonts are an easier
distribution than photographed handwriting, so a good score here is necessary
but not sufficient — it is the cheap check, and :mod:`hfont.evaluate.shaping`
plus human judgement on real samples are the expensive ones. Reporting IoU
alone would be exactly the quiet failure the brief warns about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def _ink(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    a, b = pred > threshold, target > threshold
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def ssim(pred: np.ndarray, target: np.ndarray, window: int = 7) -> float:
    """Structural similarity, computed with uniform windows.

    Included alongside IoU because the two fail differently: IoU is blind to
    how wrong a miss is, while SSIM notices local contrast and structure, which
    is closer to what "looks like the same hand" means.
    """
    from scipy.ndimage import uniform_filter

    p, t = pred.astype(np.float64), target.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2

    mu_p = uniform_filter(p, window)
    mu_t = uniform_filter(t, window)
    sigma_p = uniform_filter(p * p, window) - mu_p**2
    sigma_t = uniform_filter(t * t, window) - mu_t**2
    sigma_pt = uniform_filter(p * t, window) - mu_p * mu_t

    numerator = (2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p**2 + mu_t**2 + c1) * (sigma_p + sigma_t + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def tolerant_f1(
    pred: np.ndarray, target: np.ndarray, tolerance: float = 1.5, threshold: float = 0.5
) -> float:
    """Shape F1 where ink within ``tolerance`` px of the other shape counts as a hit.

    IoU is the wrong primary metric for handwriting. On a 4px stroke, a 1px
    offset — invisible in text, and within the noise of any real hand — takes
    IoU from 1.0 to about 0.6, and the handwriting category is precisely where
    strokes are thin. Held to an IoU bar, the fonts closest to the real target
    look like the worst failures. Tolerance-based F1 is the standard measure for
    thin structures (it is how edge and boundary detection are scored): it asks
    whether the strokes are in the right place, to within a pixel and a half.
    """
    from scipy.ndimage import distance_transform_edt

    p, t = pred > threshold, target > threshold
    if not p.any() and not t.any():
        return 1.0
    if not p.any() or not t.any():
        return 0.0
    dist_to_t = distance_transform_edt(~t)
    dist_to_p = distance_transform_edt(~p)
    precision = float((dist_to_t[p] <= tolerance).mean())
    recall = float((dist_to_p[t] <= tolerance).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ink_coverage_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """Generated ink divided by true ink. 1.0 is correct weight."""
    denominator = float(_ink(target).sum())
    return float(_ink(pred).sum() / denominator) if denominator > 1e-6 else 0.0


@dataclass
class RasterComparison:
    n: int = 0
    iou: float = 0.0
    l1: float = 0.0
    ssim: float = 0.0
    coverage_ratio: float = 0.0
    per_glyph: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"n={self.n}  IoU={self.iou:.4f}  L1={self.l1:.4f}  "
            f"SSIM={self.ssim:.4f}  coverage={self.coverage_ratio:.3f}"
        )

    def worst(self, k: int = 8) -> list[tuple[str, float]]:
        return sorted(self.per_glyph.items(), key=lambda kv: kv[1])[:k]


def compare_rasters(
    predicted: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    with_ssim: bool = True,
) -> RasterComparison:
    """Compare two sets of glyph rasters keyed by glyph key."""
    keys = sorted(set(predicted) & set(truth))
    if not keys:
        return RasterComparison()

    ious, l1s, ssims, covers = [], [], [], []
    per_glyph: dict[str, float] = {}
    for key in keys:
        p, t = _ink(predicted[key]), _ink(truth[key])
        value = iou(p, t)
        ious.append(value)
        per_glyph[key] = value
        l1s.append(float(np.abs(p - t).mean()))
        covers.append(ink_coverage_ratio(p, t))
        if with_ssim:
            ssims.append(ssim(p, t))

    return RasterComparison(
        n=len(keys),
        iou=float(np.mean(ious)),
        l1=float(np.mean(l1s)),
        ssim=float(np.mean(ssims)) if ssims else 0.0,
        coverage_ratio=float(np.mean(covers)),
        per_glyph=per_glyph,
    )


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    cache_dir: str | Path,
    split: str = "val",
    ref_seed: str = "seed24",
    max_fonts: int = 60,
    with_ssim: bool = True,
) -> dict[str, float]:
    """Score a checkpoint across held-out fonts, one font at a time.

    Evaluated per font rather than per glyph because that is the unit the
    product delivers: a user gets a whole typeface, and a model that nails 90%
    of letters but mangles ``g`` in every font has a different problem from one
    that is uniformly mediocre. The spread across fonts is reported for the
    same reason.
    """
    from ..charset import seed_charset
    from ..data.dataset import GlyphStore
    from ..generate import generate_rasters, load_checkpoint

    store = GlyphStore(cache_dir)
    model = load_checkpoint(checkpoint)

    records = store.split(split)[:max_fonts]
    if not records:
        raise ValueError(f"no fonts in split {split!r}")

    seed = seed_charset(store.charset, ref_seed)
    seed_keys = {g.key for g in seed}

    content_font = store.pick_content_font("train")
    content_keys = set(store.keys(content_font))
    content_images = {
        g.key: store.image(content_font, g.key)
        for g in store.charset
        if g.key in content_keys
    }

    per_font: list[RasterComparison] = []
    for record in records:
        available = set(store.keys(record.font_id))
        references = {k: store.image(record.font_id, k) for k in seed_keys & available}
        if not references:
            continue

        predicted, _ = generate_rasters(model, references, content_images)

        # Score only glyphs the model actually had to invent. Including the
        # reference glyphs would be scoring the input: they are passed through
        # unchanged, so each one is a free 1.0 and inflates the result by
        # roughly the seed fraction.
        truth = {
            k: store.image(record.font_id, k)
            for k in available - seed_keys
            if k in predicted
        }
        if truth:
            per_font.append(compare_rasters(predicted, truth, with_ssim))

    if not per_font:
        raise ValueError("no fonts produced a comparable result")

    ious = np.asarray([c.iou for c in per_font])
    return {
        "n_fonts": len(per_font),
        "n_glyphs": int(sum(c.n for c in per_font)),
        "iou": float(ious.mean()),
        "iou_std": float(ious.std()),
        "iou_p10": float(np.percentile(ious, 10)),
        "iou_min": float(ious.min()),
        "l1": float(np.mean([c.l1 for c in per_font])),
        "ssim": float(np.mean([c.ssim for c in per_font])),
        "coverage_ratio": float(np.mean([c.coverage_ratio for c in per_font])),
    }
