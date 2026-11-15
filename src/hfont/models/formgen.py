"""A writer's joined letter forms from their isolated letters (positional-form completion).

Arabic letters change shape with their place in a word: beh's bowl becomes
a tooth on the line, heh's ring a figure resting on it. The shape of a form
is mostly the letter's (borrowing another writer's form scores 0.435 on the
AHAWP validation writers, against 0.281 for the isolated letter unchanged);
what is the writer's is the hand it is drawn in (their own second copy:
0.666). This model learns the letter's change of shape and applies it in
the writer's hand:

* ``letter``: the writer's isolated form of the letter to draw;
* ``refs``: all of the writer's isolated letters, pooled into a style;
* letter and form identity, as embeddings;

and draws the form's ink. The images are AHAWP scans at half size: 64 rows
(the cell's own rows, so where the letter sits in its cell is kept) by 256
columns, the ink centred across (``to_canvas``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 64, 256


def to_canvas(image: np.ndarray) -> np.ndarray:
    """An AHAWP ink image (128 rows, any width) at half size, ink centred across a 64 x 256 canvas."""
    from PIL import Image

    im = np.asarray(image, np.float32)
    half = np.asarray(Image.fromarray(im).resize((max(1, im.shape[1] // 2), H), Image.BOX), np.float32)
    cols = np.nonzero((half > 0.25).any(axis=0))[0]
    out = np.zeros((H, W), np.float32)
    if cols.size == 0:
        return out
    ink = half[:, cols[0]:cols[-1] + 1]
    if ink.shape[1] > W:                                   # wider than any AHAWP letter
        c = (ink.shape[1] - W) // 2
        ink = ink[:, c:c + W]
    x = (W - ink.shape[1]) // 2
    out[:, x:x + ink.shape[1]] = ink
    return out


def from_canvas(canvas: np.ndarray) -> np.ndarray:
    """A drawn canvas back at AHAWP size (128 rows), for scoring against the scans."""
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(canvas, np.float32)).resize((2 * W, 2 * H), Image.BILINEAR))


@dataclass
class FormConfig:
    letters: int = 20
    forms: int = 4                  # Form enum order: isol, init, medi, fina
    width: int = 64                 # channels of the first block
    d: int = 256                    # bottleneck channels
    mix_layers: int = 2             # transformer layers over the 8 x 32 bottleneck
    #: The style letters are halved this many times before encoding. Style is
    #: one pooled vector of up to 20 letters a sample; at full size they made
    #: the style encoder the cost of the model (0.8 s/step at batch 64 on an L4).
    style_pool: int = 1
    #: Retrieve and refine: also read the same form as written by the most
    #: alike training writer (a second input channel), with U-Net skips from
    #: it to the output so its strokes can be kept and adjusted. Without it
    #: the model drew one centred blob for every form (val 0.189 at 2K steps).
    donor: bool = False


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class Encoder(nn.Module):
    """channels x 64 x 256 -> d x 8 x 32 (and the two finer levels, for skips)."""

    def __init__(self, c: FormConfig, channels: int = 1) -> None:
        super().__init__()
        w = c.width
        self.blocks = nn.ModuleList([_block(channels, w, 2), _block(w, 2 * w, 2), _block(2 * w, c.d, 2)])

    def forward(self, x: torch.Tensor, levels: bool = False):
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out if levels else x


class FiLM(nn.Module):
    def __init__(self, cond: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class FormGenerator(nn.Module):
    def __init__(self, config: FormConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or FormConfig()
        d = c.d
        self.inputs = 2 if c.donor else 1
        self.letter_encoder = Encoder(c, self.inputs)
        self.style_encoder = Encoder(c)
        self.letter_emb = nn.Embedding(c.letters, d)
        self.form_emb = nn.Embedding(c.forms, d)
        cond = 3 * d
        self.film_in = FiLM(cond, d)
        ys, xs = torch.meshgrid(torch.arange(H // 8), torch.arange(W // 8), indexing="ij")
        self.register_buffer("grid", torch.stack([ys, xs]).float()[None] / torch.tensor([H // 8, W // 8])[:, None, None])
        self.pos = nn.Conv2d(2, d, 1)
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), c.mix_layers)
        w = c.width
        # With a donor, each decoder level also takes the encoder's features at
        # its size (and the inputs themselves at full size).
        skip = (2 * w, w, self.inputs) if c.donor else (0, 0, 0)
        self.up = nn.ModuleList([_block(d + skip[0], 2 * w, 1), _block(2 * w + skip[1], w, 1),
                                 _block(w + skip[2], w, 1)])
        self.films = nn.ModuleList([FiLM(cond, 2 * w), FiLM(cond, w), FiLM(cond, w)])
        self.out = nn.Conv2d(w, 1, 3, 1, 1)

    def style(self, refs: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        """refs (B,K,1,H,W), ref_mask (B,K) -> (B,d): the mean of the present letters' features."""
        b, k = refs.shape[:2]
        x = refs.flatten(0, 1)
        if self.config.style_pool:
            x = F.avg_pool2d(x, 2 ** self.config.style_pool)
        f = self.style_encoder(x).mean(dim=(2, 3)).reshape(b, k, -1)
        m = ref_mask.float()[..., None]
        return (f * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, letter: torch.Tensor, refs: torch.Tensor, ref_mask: torch.Tensor,
                letter_id: torch.Tensor, form_id: torch.Tensor) -> torch.Tensor:
        """Logits of the form's ink, (B,1,H,W).

        ``letter`` is (B,1,H,W), the writer's isolated letter, or with
        ``config.donor`` (B,2,H,W): that letter and the donor's form.
        """
        cond = torch.cat([self.style(refs, ref_mask), self.letter_emb(letter_id), self.form_emb(form_id)], 1)
        levels = self.letter_encoder(letter, levels=True)
        x = self.film_in(levels[-1], cond) + self.pos(self.grid)
        b, d, h, w = x.shape
        x = self.mix(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, d, h, w)
        skips = [levels[1], levels[0], letter] if self.config.donor else [None] * 3
        for up, film, skip in zip(self.up, self.films, skips):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], 1)
            x = film(up(x), cond)
        return self.out(x)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> dict:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return {"loss": bce + dice.mean(), "bce": bce, "dice": dice.mean()}

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, data: dict) -> "FormGenerator":
        model = cls(FormConfig(**data["config"]))
        model.load_state_dict(data["model"])
        return model
