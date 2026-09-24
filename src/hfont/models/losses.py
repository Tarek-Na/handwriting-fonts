"""Training objectives.

Weighting notes, since the balance here is doing real work:

*L1 is the backbone.* The task is deterministic — one correct letterform per
(character, style) — so a strong pixel loss is appropriate, and the usual
objection to L1 (it blurs multi-modal outputs) does not apply when the output
is not supposed to be multi-modal.

*Ink is weighted above background.* A 128x128 glyph is roughly 90% empty. Plain
L1 lets a model score well by predicting white everywhere, and early training
does exactly that. Weighting ink pixels keeps gradient on the strokes.

*Dice complements L1 rather than duplicating it.* L1 is a per-pixel intensity
match; soft Dice measures overlap of the whole shape and is scale-free with
respect to how much ink there is, so a thin letter like ``l`` contributes as
much shape signal as a heavy one like ``M``.

*Adversarial and feature-matching terms are off by default.* They sharpen edges,
which matters because the raster gets vectorized, but they also destabilize.
The intended sequence is to get reconstruction working first, then switch them
on to crisp it up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    l1: float = 10.0
    #: Extra multiplier applied to L1 on pixels that are ink in the target.
    ink_weight: float = 4.0
    dice: float = 2.0
    advance: float = 1.0
    adversarial: float = 0.0
    feature_matching: float = 0.0
    char_aux: float = 0.0
    font_aux: float = 0.0
    #: Pulls the style code of a generated glyph toward the style code of the
    #: references it came from. Guards against the decoder ignoring style.
    style_consistency: float = 0.0
    #: Pushes style codes of different fonts apart and two views of the same
    #: font together (see style_contrastive). Trains the property the
    #: reconstruction loss never asks for. 0 disables.
    style_contrastive: float = 0.0
    style_temperature: float = 0.1


def to_ink(x: torch.Tensor) -> torch.Tensor:
    """Map the model's [-1, 1] signed output to [0, 1] ink coverage."""
    return (x + 1.0) * 0.5


def weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    ink_weight: float,
    band: int = 2,
    extra_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 with extra weight on a band around the target's strokes.

    The band is the target ink *dilated* by ``band`` pixels, so it covers both
    sides of every stroke edge. An earlier version weighted the target ink
    itself, which is one-sided: a pixel of missing ink cost four times a pixel
    of extra ink, so wherever the model was unsure where an edge was, painting
    too much was the cheaper bet. Trained that way, the model produced ~1.5x the
    true ink across the validation set — every generated font came out bolder
    than the hand it was imitating. Weighting both sides of the edge equally
    removes the incentive while still keeping the loss focused on the strokes
    rather than the 90% of the canvas that is empty paper.

    ``extra_weight`` multiplies in a per-pixel map from the data loader — the
    small-component boost (see data/dataset.py:component_weight_map).
    """
    diff = (pred - target).abs()
    if ink_weight <= 1.0 and extra_weight is None:
        return diff.mean()
    ink = to_ink(target)
    if band > 0:
        ink = F.max_pool2d(ink, kernel_size=2 * band + 1, stride=1, padding=band)
    weight = 1.0 + (max(ink_weight, 1.0) - 1.0) * ink
    if extra_weight is not None:
        weight = weight * extra_weight
    return (diff * weight).sum() / weight.sum()


def soft_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = to_ink(pred).flatten(1)
    t = to_ink(target).flatten(1)
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def feature_matching(real_feats: list[torch.Tensor], fake_feats: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.l1_loss(f, r.detach()) for r, f in zip(real_feats, fake_feats)) / max(
        len(real_feats), 1
    )


class GlyphLoss(nn.Module):
    """Assembles the generator objective and reports its parts for logging."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LossConfig()

    def reconstruction(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_advance: torch.Tensor | None = None,
        target_advance: torch.Tensor | None = None,
        target_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        parts: dict[str, float] = {}

        l1 = weighted_l1(pred_image, target_image, cfg.ink_weight, extra_weight=target_weight)
        total = cfg.l1 * l1
        parts["l1"] = float(l1.detach())

        if cfg.dice > 0:
            dice = soft_dice(pred_image, target_image)
            total = total + cfg.dice * dice
            parts["dice"] = float(dice.detach())

        if cfg.advance > 0 and pred_advance is not None and target_advance is not None:
            adv = F.l1_loss(pred_advance, target_advance)
            total = total + cfg.advance * adv
            parts["advance"] = float(adv.detach())

        return total, parts

    def forward(self, *args, **kwargs):  # pragma: no cover - use reconstruction()
        return self.reconstruction(*args, **kwargs)


@torch.no_grad()
def pixel_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Quick quality read-outs that are comparable across runs."""
    p, t = to_ink(pred), to_ink(target)
    l1 = (p - t).abs().mean()
    rmse = torch.sqrt(((p - t) ** 2).mean())

    pb, tb = p > threshold, t > threshold
    inter = (pb & tb).flatten(1).sum(dim=1).float()
    union = (pb | tb).flatten(1).sum(dim=1).float()
    iou = (inter / union.clamp_min(1.0)).mean()

    # Ink-coverage ratio catches a model that is systematically too light or
    # too heavy even when IoU looks acceptable.
    coverage = (p.flatten(1).sum(dim=1) / t.flatten(1).sum(dim=1).clamp_min(1e-6)).mean()

    return {
        "l1": float(l1),
        "rmse": float(rmse),
        "iou": float(iou),
        "coverage_ratio": float(coverage),
    }


def style_contrastive(
    view_a: torch.Tensor, view_b: torch.Tensor, valid: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Same hand -> same code; different hand -> different code.

    Nothing in the reconstruction objective asks the style encoder to *separate*
    styles. It only asks that the decoded glyph match its target, and the
    content path plus the character embedding already carry most of what that
    needs. Measured consequence: style codes for three different people sit at
    cosine 0.99+ and the model cannot tell them apart
    (output/review/STYLE_COLLAPSE.md). Reference attention lifted held-out
    corpus IoU by 0.087 and moved that number not at all, which is what
    motivates training the property directly instead of hoping it emerges.

    The two views are encoded from *disjoint halves* of one sample's references,
    so a positive pair is the same hand seen through different letters rather
    than the same letters twice.

    It is not free. The halves are encoded *in addition to* the full reference
    set the decoder needs, so the style backbone -- the most expensive part of
    the model, since it sees every reference of every sample -- does twice the
    work. Measured on a T4 at batch 32: 67 img/s against 99 without, about 32%
    slower end to end. An earlier version of this docstring claimed the cost was
    flat, which was wrong: it confused the cost *within* the split with the cost
    of the split plus the pass that was already there.

    ``valid`` marks samples with at least two references, since a sample with
    one cannot be split into two views.
    """
    if valid.sum() < 2:
        return view_a.new_zeros(())

    a = F.normalize(view_a[valid].float(), dim=1)
    b = F.normalize(view_b[valid].float(), dim=1)
    n = a.shape[0]

    z = torch.cat([a, b], dim=0)
    sim = (z @ z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))

    # Each view's positive is the other view of the same sample; every other
    # column is a different font, and therefore a negative.
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)
