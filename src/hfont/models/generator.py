"""The few-shot glyph generator.

Three paths meet in the decoder:

    content   a rendering of the target character in some other font
    identity  a learned embedding of which character it is
    style     a pooled code from K reference glyphs of the target hand

The content image supplies spatial structure, which matters at this scale: with
a few hundred training fonts there is not enough data to learn letterforms from
a class embedding alone, and a skeleton to deform is a much easier starting
point than a blank canvas. The identity embedding is carried alongside it
because the content image is drawn in an arbitrary font and can be ambiguous
(l/I, O/0), and the model should not have to guess.

The style path pools to a single vector. Pooling rather than attending is a
deliberate simplicity: it is exactly permutation-invariant and indifferent to
how many references it gets, which is what lets a model trained on 1-8
references accept the ~24 a user actually writes.

The width head is not decoration. A raster alone cannot be assembled into a
font — glyphs need advance widths, or the exported font sets text at uniform
spacing and looks obviously wrong regardless of how good the letterforms are.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    AdaIN,
    AdaINResBlock,
    ConvBlock,
    ReferenceAttention,
    ResBlock,
    UpBlock,
    masked_max,
    masked_mean,
)


@dataclass
class GeneratorConfig:
    image_size: int = 128
    n_chars: int = 75
    base_channels: int = 48
    max_channels: int = 384
    style_dim: int = 256
    char_embed_dim: int = 64
    n_bottleneck: int = 4
    n_style_blocks: int = 4
    #: Encoder/decoder depth. 4 takes 128 -> 8.
    n_scales: int = 4
    predict_advance: bool = True
    #: Let the target attend to the references directly, alongside the pooled
    #: style code. Off by default: the shipped model does not have these
    #: weights. See models/blocks.ReferenceAttention.
    style_attention: bool = False
    attention_heads: int = 4


class StyleEncoder(nn.Module):
    """Pools K reference glyphs into one style vector."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        layers: list[nn.Module] = [ConvBlock(1, ch, stride=1)]
        for _ in range(cfg.n_scales + 1):
            nxt = min(ch * 2, cfg.max_channels)
            layers.append(ConvBlock(ch, nxt, stride=2))
            ch = nxt
        self.backbone = nn.Sequential(*layers)
        self.out_channels = ch
        self.shape_bucket = 32
        # mean and max are concatenated: the mean describes the hand on average,
        # the max catches features that appear in only one or two references
        # (a single sharp terminal, one unusually long descender).
        self.head = nn.Sequential(
            nn.Linear(ch * 2, cfg.style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cfg.style_dim, cfg.style_dim),
        )

    def forward(
        self, refs: torch.Tensor, mask: torch.Tensor, return_spatial: bool = False
    ):
        """refs: (B, K, 1, H, W), mask: (B, K) -> (B, style_dim).

        Only the real references go through the backbone. Stacks are padded to
        the maximum K so they batch, but training draws 1-8 references, so
        about 44% of padded slots are empty — and this backbone is the most
        expensive part of the model, since it sees every reference of every
        sample. Instance norm is per-sample, so encoding the subset produces
        exactly the features encoding everything would.
        """
        b, k = refs.shape[:2]
        flat = refs.reshape(b * k, *refs.shape[2:])
        valid = mask.reshape(b * k)

        # Round the encoded count up to a multiple of `shape_bucket`, topping up
        # with padded slots whose outputs are then discarded.
        #
        # Encoding exactly the valid references gives the backbone a different
        # batch size on nearly every training step, and with cudnn.benchmark on,
        # cuDNN re-runs algorithm autotuning for every new input shape. On a T4
        # that took training from ~200 images/s to 9. Bucketing caps the number
        # of distinct shapes at k*b/bucket (8 for the default batch) while still
        # skipping most of the padding.
        valid_idx = valid.nonzero(as_tuple=True)[0]
        n_valid = valid_idx.numel()
        n_encode = min(-(-n_valid // self.shape_bucket) * self.shape_bucket, b * k)
        if n_encode > n_valid:
            filler = (~valid).nonzero(as_tuple=True)[0][: n_encode - n_valid]
            encode_idx = torch.cat([valid_idx, filler])
        else:
            encode_idx = valid_idx

        maps = self.backbone(flat[encode_idx])
        encoded = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        feats = encoded.new_zeros(b * k, encoded.shape[1])
        feats[valid_idx] = encoded[:n_valid]
        feats = feats.reshape(b, k, -1)
        pooled = torch.cat([masked_mean(feats, mask), masked_max(feats, mask)], dim=1)
        style = self.head(pooled)
        if not return_spatial:
            return style
        # The same features before they were averaged away, for attention to read.
        ch, fh, fw = maps.shape[1:]
        spatial = maps.new_zeros(b * k, ch, fh, fw)
        spatial[valid_idx] = maps[:n_valid]
        return style, spatial.reshape(b, k, ch, fh, fw)


class ContentEncoder(nn.Module):
    """Encodes the content glyph into a bottleneck plus skip features."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        self.stem = ConvBlock(1, ch, stride=1)
        self.downs = nn.ModuleList()
        self.skip_channels: list[int] = []
        for _ in range(cfg.n_scales):
            self.skip_channels.append(ch)
            nxt = min(ch * 2, cfg.max_channels)
            self.downs.append(ConvBlock(ch, nxt, stride=2))
            ch = nxt
        self.out_channels = ch
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(cfg.n_bottleneck)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            skips.append(h)
            h = down(h)
        return self.blocks(h), skips


class Decoder(nn.Module):
    def __init__(self, cfg: GeneratorConfig, in_channels: int, skip_channels: list[int]) -> None:
        super().__init__()
        self.style_blocks = nn.ModuleList(
            [AdaINResBlock(in_channels, cfg.style_dim) for _ in range(cfg.n_style_blocks)]
        )
        self.ups = nn.ModuleList()
        ch = in_channels
        for skip_ch in reversed(skip_channels):
            nxt = max(ch // 2, cfg.base_channels)
            self.ups.append(UpBlock(ch, nxt, cfg.style_dim, skip_ch=skip_ch))
            ch = nxt
        self.out_norm = AdaIN(ch, cfg.style_dim)
        self.to_image = nn.Conv2d(ch, 1, 7, 1, 3)

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        for block in self.style_blocks:
            x = block(x, style)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, style, skip)
        return torch.tanh(self.to_image(F.leaky_relu(self.out_norm(x, style), 0.2)))


class GlyphGenerator(nn.Module):
    """Content + style -> glyph raster (and its advance width)."""

    def __init__(self, cfg: GeneratorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or GeneratorConfig()
        self.style_encoder = StyleEncoder(cfg)
        self.content_encoder = ContentEncoder(cfg)

        self.char_embed = nn.Embedding(cfg.n_chars, cfg.char_embed_dim)
        bottleneck = self.content_encoder.out_channels
        self.merge_char = nn.Conv2d(bottleneck + cfg.char_embed_dim, bottleneck, 1)

        self.decoder = Decoder(cfg, bottleneck, self.content_encoder.skip_channels)

        self.ref_attention = (
            ReferenceAttention(bottleneck, cfg.attention_heads)
            if cfg.style_attention else None
        )

        if cfg.predict_advance:
            self.advance_head = nn.Sequential(
                nn.Linear(cfg.style_dim + cfg.char_embed_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1),
            )

    def encode_style(self, refs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(refs, mask)

    def encode_style_full(self, refs: torch.Tensor, mask: torch.Tensor):
        """Pooled code *and* the per-reference features it was pooled from."""
        return self.style_encoder(refs, mask, return_spatial=True)

    def decode(
        self,
        content: torch.Tensor,
        char_id: torch.Tensor,
        style: torch.Tensor,
        ref_feats: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feats, skips = self.content_encoder(content)
        char = self.char_embed(char_id)
        char_map = char.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *feats.shape[-2:])
        feats = self.merge_char(torch.cat([feats, char_map], dim=1))

        if self.ref_attention is not None and ref_feats is not None:
            feats = self.ref_attention(feats, ref_feats, ref_mask)

        image = self.decoder(feats, style, skips)
        out = {"image": image}
        if self.cfg.predict_advance:
            out["advance"] = self.advance_head(torch.cat([style, char], dim=1)).squeeze(-1)
        return out

    def forward(
        self,
        content: torch.Tensor,
        refs: torch.Tensor,
        ref_mask: torch.Tensor,
        char_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.ref_attention is not None:
            style, ref_feats = self.encode_style_full(refs, ref_mask)
            out = self.decode(content, char_id, style, ref_feats, ref_mask)
        else:
            style = self.encode_style(refs, ref_mask)
            out = self.decode(content, char_id, style)
        out["style"] = style
        return out

    @torch.no_grad()
    def generate_set(
        self,
        contents: torch.Tensor,
        char_ids: torch.Tensor,
        refs: torch.Tensor,
        ref_mask: torch.Tensor,
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Generate a whole charset from one style, reusing the style code.

        The style encoder runs once for the entire font rather than once per
        glyph: it is the same references every time, and re-encoding them would
        be both wasteful and a source of drift between glyphs of one font.
        """
        self.eval()
        style = self.encode_style(refs, ref_mask)
        images, advances = [], []
        for i in range(0, contents.shape[0], batch_size):
            chunk = contents[i : i + batch_size]
            ids = char_ids[i : i + batch_size]
            out = self.decode(chunk, ids, style.expand(chunk.shape[0], -1))
            images.append(out["image"])
            if "advance" in out:
                advances.append(out["advance"])
        result = {"image": torch.cat(images)}
        if advances:
            result["advance"] = torch.cat(advances)
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
