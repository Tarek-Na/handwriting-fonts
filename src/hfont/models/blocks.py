"""Building blocks for the glyph generator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive instance normalization driven by a style vector.

    Instance norm strips per-feature-map mean and variance — which for a glyph
    feature map is close to "how much ink, how heavy" — and the style vector
    puts them back. That is exactly the split this project wants: the content
    path decides *where* strokes go, the style path decides what they look like.
    """

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.to_scale_shift = nn.Linear(style_dim, channels * 2)
        # Start as identity: scale 1, shift 0, so early training is a plain
        # autoencoder and the style path ramps in rather than fighting it.
        nn.init.zeros_(self.to_scale_shift.weight)
        with torch.no_grad():
            self.to_scale_shift.bias[:channels].fill_(1.0)
            self.to_scale_shift.bias[channels:].fill_(0.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(style).chunk(2, dim=1)
        return self.norm(x) * scale.unsqueeze(-1).unsqueeze(-1) + shift.unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    """Conv -> norm -> activation, optionally changing resolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm: str = "instance",
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=norm == "none")
        if norm == "instance":
            self.norm: nn.Module = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(out_ch)
        else:
            self.norm = nn.Identity()
        self.act: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True) if activation == "lrelu"
            else nn.ReLU(inplace=True) if activation == "relu"
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Pre-activation residual block with no style conditioning."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.norm1(x), 0.2))
        h = self.conv2(F.leaky_relu(self.norm2(h), 0.2))
        return x + h


class AdaINResBlock(nn.Module):
    """Residual block whose two normalizations are style-driven."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = AdaIN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        # Zero-init the residual branch's last conv so the block starts as a
        # no-op; with a dozen of these stacked, random init otherwise destroys
        # the content signal before any gradient arrives.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.norm1(x, style), 0.2))
        h = self.conv2(F.leaky_relu(self.norm2(h, style), 0.2))
        return x + h


class UpBlock(nn.Module):
    """2x upsample, then fuse an optional style-modulated skip."""

    def __init__(
        self, in_ch: int, out_ch: int, style_dim: int, skip_ch: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm = AdaIN(out_ch, style_dim)
        self.skip_ch = skip_ch
        if skip_ch:
            # The skip carries the *content font's* stroke weight as well as its
            # structure. Projecting it and re-normalizing under the target style
            # lets the decoder keep the structure without importing the weight.
            self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1)
            self.skip_norm = AdaIN(out_ch, style_dim)
            self.fuse = nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1)

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = F.leaky_relu(self.norm(self.conv(x), style), 0.2)
        if self.skip_ch and skip is not None:
            s = F.leaky_relu(self.skip_norm(self.skip_proj(skip), style), 0.2)
            h = F.leaky_relu(self.fuse(torch.cat([h, s], dim=1)), 0.2)
        return h


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim 1 of ``x`` (B, K, D) using a boolean (B, K) mask."""
    weight = mask.to(x.dtype).unsqueeze(-1)
    total = weight.sum(dim=1).clamp_min(1.0)
    return (x * weight).sum(dim=1) / total


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over dim 1 of ``x`` (B, K, D) using a boolean (B, K) mask."""
    neg_inf = torch.finfo(x.dtype).min
    filled = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
    out = filled.max(dim=1).values
    # A row with no valid entries would be all -inf; zero it instead.
    return torch.where(mask.any(dim=1, keepdim=True), out, torch.zeros_like(out))
