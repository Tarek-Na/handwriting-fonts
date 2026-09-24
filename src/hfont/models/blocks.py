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


class ReferenceAttention(nn.Module):
    """Let each position of the target glyph read from the reference glyphs.

    The pooled style vector averages K references into one code, and averaging
    is the documented failure mode of this architecture: a survey of the field
    states that "an average operation is usually performed on the extracted
    features, which easily weakens the local information and results in the loss
    of fine-grained details", and this project measured exactly that -- three
    different people's handwriting received style codes at cosine 0.98+ and
    near-identical generated letters (output/review/STYLE_COLLAPSE.md).

    This is the alternative from FS-Font (CVPR 2022): content features are the
    queries, reference features are the keys and values, so a stem in the target
    can attend to the stems in the references rather than to their average. It
    is *added alongside* the pooled path rather than replacing it -- the pooled
    code still drives AdaIN and the advance head -- because replacing a working
    subsystem on an untested hypothesis is how the last two rounds went wrong.

    The output projection is zero-initialised, so an untrained block is exactly
    the identity and a model can be warm-started into this architecture without
    its existing behaviour being destroyed on the first step.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, refs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, C, H, W); refs: (B, K, C, h, w); mask: (B, K) -> (B, C, H, W)."""
        b, c, h, w = x.shape
        k = refs.shape[1]

        q = self.to_q(self.norm(x)).reshape(b, self.heads, c // self.heads, h * w)
        q = q.transpose(-2, -1)

        tokens = refs.permute(0, 1, 3, 4, 2).reshape(b, -1, c)
        keys, values = self.to_kv(tokens).chunk(2, dim=-1)
        n = tokens.shape[1]
        keys = keys.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        values = values.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)

        # Padded reference slots carry zeros, not silence; mask them out or the
        # attention learns to read whatever the padding happens to encode.
        per_ref = n // max(k, 1)
        valid = mask.repeat_interleave(per_ref, dim=1)
        attn_mask = valid[:, None, None, :].expand(b, self.heads, h * w, n)

        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=attn_mask)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)
