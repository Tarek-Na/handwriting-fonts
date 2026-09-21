"""Multi-task patch discriminator.

Adversarial training here is a *sharpener*, not the main objective. L1 alone
produces correct but slightly soft letterforms, and softness is specifically
damaging in this project because the raster is subsequently traced into Bézier
curves — a blurry stem edge becomes a wobbly outline, and the wobble survives
into the installed font. The discriminator exists to keep stroke edges hard.

It carries two auxiliary classification heads, following zi2zi and DG-Font:

    char head   which character is this?   -> keeps content legible
    font head   which font is this?        -> keeps style distinctive

The font head only knows training fonts, which is fine: it is a training-time
signal to stop the generator collapsing every style toward the corpus mean, not
something used at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


@dataclass
class DiscriminatorConfig:
    image_size: int = 128
    n_chars: int = 75
    n_fonts: int = 1
    base_channels: int = 48
    max_channels: int = 384
    n_scales: int = 4
    use_spectral_norm: bool = True


def _conv(in_ch: int, out_ch: int, k: int, s: int, p: int, sn: bool) -> nn.Module:
    layer = nn.Conv2d(in_ch, out_ch, k, s, p)
    return spectral_norm(layer) if sn else layer


class PatchDiscriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or DiscriminatorConfig()
        sn = cfg.use_spectral_norm

        ch = cfg.base_channels
        layers: list[nn.Module] = [_conv(1, ch, 4, 2, 1, sn), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(cfg.n_scales - 1):
            nxt = min(ch * 2, cfg.max_channels)
            layers += [
                _conv(ch, nxt, 4, 2, 1, sn),
                nn.InstanceNorm2d(nxt, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = nxt
        self.backbone = nn.Sequential(*layers)

        # Patch-level real/fake map rather than a single scalar: local stroke
        # texture is what we are trying to fix, and a patch verdict gives a
        # gradient at every stem edge instead of one number for the glyph.
        self.to_patch = _conv(ch, 1, 3, 1, 1, sn)
        self.char_head = nn.Linear(ch, cfg.n_chars)
        self.font_head = nn.Linear(ch, cfg.n_fonts) if cfg.n_fonts > 1 else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        out = {"patch": self.to_patch(feats), "char_logits": self.char_head(pooled)}
        if self.font_head is not None:
            out["font_logits"] = self.font_head(pooled)
        return out


def hinge_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def hinge_g_loss(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()
