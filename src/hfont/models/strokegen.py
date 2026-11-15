"""A letter drawn as pen moves, in the style of a few written letters (step A3).

The image model (``generator.py``) paints a 128 px picture and, unsure where a
stroke goes, paints the average of the places it could go: the "average
letter" every arm so far has converged to. This model instead writes the
letter as a sequence of pen moves (``hfont.strokes``), one at a time, each
drawn from a mixture of Gaussians, so it commits to one stroke path and never
blurs two. The design follows SDT (Dai et al., CVPR 2023): convolutional
encoders for the letter to write (``content``, a plain font's glyph) and for
the writer's letters (``refs``), and a Transformer decoder that writes the
pen sequence attending to both.

Sequence rows are ``(dx, dy, pen)`` with ``dx, dy`` in units of the stroke
step (``JUMP`` steps for the move onto a stroke's first point) and ``pen``
0 = keep drawing, 1 = lift after this point, 2 = the letter is done (a final
row with no move).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DRAW, LIFT, END = 0, 1, 2
#: A jump to a stroke's first point (from the canvas corner, or after a lift)
#: is 10-40 steps long, a drawn move about one: jumps are written in units of
#: JUMP steps, so both are of a size the output head reaches easily.
JUMP = 10.0
#: The letters' canvas (the glyph cache's), pixels.
CANVAS_PX = 128


@dataclass
class StrokeConfig:
    d_model: int = 256
    heads: int = 8
    layers: int = 6
    style_layers: int = 2
    mixtures: int = 20
    max_len: int = 256
    image_size: int = 64          # encoders see letters at this size
    ref_tokens: int = 4           # each reference letter becomes ref_tokens^2 tokens
    dropout: float = 0.1
    step: float = 3.0             # pixels per unit of dx, dy (the cache's resampling step)
    #: Also give the decoder where the pen is (the sum of the moves so far),
    #: not only the last move. Off in checkpoints made before it existed.
    position: bool = False


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class LetterEncoder(nn.Module):
    """1 x 64 x 64 letter -> d x 8 x 8 features.

    Halves the size in its first block, not its last: a training step encodes
    batch x up to 30 reference letters, and full-resolution layers made the
    encoder the cost of the whole model (0.68 -> 0.23 GMAC a letter).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_block(1, 64, 2), _block(64, 128, 2), _block(128, 256, 2),
                                 _block(256, d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _grid_embedding(n: int, d: int) -> torch.Tensor:
    """Fixed 2-D sine positions for an n x n grid, (n*n, d)."""
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    freq = torch.exp(torch.arange(0, d // 4) * (-math.log(1000.0) / max(d // 4, 1)))
    parts = [torch.sin(ys.flatten()[:, None] * freq), torch.cos(ys.flatten()[:, None] * freq),
             torch.sin(xs.flatten()[:, None] * freq), torch.cos(xs.flatten()[:, None] * freq)]
    return torch.cat(parts, dim=1)[:, :d]


class StrokeGenerator(nn.Module):
    def __init__(self, config: StrokeConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or StrokeConfig()
        d = c.d_model
        self.content_encoder = LetterEncoder(d)
        self.style_encoder = LetterEncoder(d)
        self.register_buffer("content_pos", _grid_embedding(c.image_size // 8, d), persistent=False)
        self.register_buffer("ref_pos", _grid_embedding(c.ref_tokens, d), persistent=False)
        self.kind = nn.Embedding(2, d)                     # 0 content token, 1 style token
        self.style_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, c.heads, 4 * d, c.dropout, batch_first=True, norm_first=True),
            c.style_layers)
        self.move_in = nn.Sequential(nn.Linear(8 if c.position else 6, d), nn.GELU(), nn.Linear(d, d))
        self.step_pos = nn.Embedding(c.max_len + 1, d)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, c.heads, 4 * d, c.dropout, batch_first=True, norm_first=True),
            c.layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 6 * c.mixtures + 3)

    # ------------------------------------------------------------------ memory
    def memory(self, content: torch.Tensor, refs: torch.Tensor, ref_mask: torch.Tensor) -> tuple:
        """content (B,1,H,W), refs (B,K,1,H,W), ref_mask (B,K) True = present."""
        b, k = refs.shape[:2]
        c = self.content_encoder(content).flatten(2).transpose(1, 2) + self.content_pos
        c = c + self.kind.weight[0]
        s = self.style_encoder(refs.flatten(0, 1))
        s = F.adaptive_avg_pool2d(s, self.config.ref_tokens).flatten(2).transpose(1, 2)
        s = (s + self.ref_pos + self.kind.weight[1]).reshape(b, k * s.shape[1], -1)
        pad = ~ref_mask.repeat_interleave(self.config.ref_tokens ** 2, dim=1)
        s = self.style_mixer(s, src_key_padding_mask=pad)
        mem = torch.cat([c, s], dim=1)
        mem_pad = torch.cat([torch.zeros(b, c.shape[1], dtype=torch.bool, device=pad.device), pad], 1)
        return mem, mem_pad

    def inputs(self, rows: torch.Tensor) -> torch.Tensor:
        """Decoder input for target rows (B,T,3): a start row, then rows[:-1].

        Features: the move, the pen state (3), a start flag, and with
        ``config.position`` where the pen then is, in canvases (the moves
        summed, jumps back in steps): a relative move alone leaves the model
        to add up its whole history to know where on the letter it is.
        """
        b, t, _ = rows.shape
        pen = F.one_hot(rows[..., 2].long().clamp(0, 2), 3).float()
        parts = [rows[..., :2], pen, torch.zeros(b, t, 1, device=rows.device)]
        if self.config.position:
            lifted = rows[..., 2] == LIFT
            jump = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=rows.device), lifted[:, :-1]], 1)
            moves = rows[..., :2] * (1.0 + (JUMP - 1.0) * jump.float())[..., None]
            parts.append(torch.cumsum(moves, 1) * self.config.step / CANVAS_PX)
        feats = torch.cat(parts, -1)
        start = torch.zeros(b, 1, feats.shape[-1], device=rows.device)
        start[..., 5] = 1.0
        return torch.cat([start, feats[:, :-1]], dim=1)

    def decode(self, inputs: torch.Tensor, mem: torch.Tensor, mem_pad: torch.Tensor) -> torch.Tensor:
        t = inputs.shape[1]
        x = self.move_in(inputs) + self.step_pos(torch.arange(t, device=inputs.device))
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=inputs.device), 1)
        h = self.decoder(x, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.head(self.norm(h))

    def forward(self, content, refs, ref_mask, rows):
        mem, mem_pad = self.memory(content, refs, ref_mask)
        return self.decode(self.inputs(rows), mem, mem_pad)

    # -------------------------------------------------------------------- loss
    def split(self, out: torch.Tensor):
        m = self.config.mixtures
        logit_pi, mx, my, sx, sy, rho, pen = torch.split(out, [m, m, m, m, m, m, 3], dim=-1)
        sx = F.softplus(sx) + 1e-3
        sy = F.softplus(sy) + 1e-3
        return logit_pi, mx, my, sx, sy, torch.tanh(rho) * 0.95, pen

    def loss(self, out: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor) -> dict:
        """Mixture NLL of the moves (drawn rows only) plus pen cross-entropy (all rows)."""
        logit_pi, mx, my, sx, sy, rho, pen = self.split(out.float())
        x, y = rows[..., :1], rows[..., 1:2]
        zx, zy = (x - mx) / sx, (y - my) / sy
        z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
        one_minus = 1 - rho ** 2
        log_n = -z / (2 * one_minus) - torch.log(2 * math.pi * sx * sy * torch.sqrt(one_minus))
        log_p = torch.logsumexp(F.log_softmax(logit_pi, -1) + log_n, dim=-1)
        moves = mask & (rows[..., 2] != END)
        move_nll = -(log_p * moves).sum() / moves.sum().clamp(min=1)
        pen_ce = F.cross_entropy(pen[mask], rows[..., 2][mask].long())
        return {"loss": move_nll + pen_ce, "move": move_nll, "pen": pen_ce}

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def write(self, content, refs, ref_mask, max_len: int | None = None,
              temperature: float = 0.0) -> list[np.ndarray]:
        """Write each letter of the batch; returns (T, 3) rows ``(dx, dy, lift)`` per letter.

        ``temperature`` 0 takes the likeliest component's mean at every step.
        """
        self.eval()
        max_len = max_len or self.config.max_len
        mem, mem_pad = self.memory(content, refs, ref_mask)
        b = content.shape[0]
        jumps: list[torch.Tensor] = []
        rows = torch.zeros(b, 0, 3, device=content.device)
        done = torch.zeros(b, dtype=torch.bool, device=content.device)
        jumping = torch.ones(b, dtype=torch.bool, device=content.device)
        for _ in range(max_len):
            nxt_in = self.inputs(torch.cat([rows, torch.zeros(b, 1, 3, device=rows.device)], 1))
            out = self.decode(nxt_in, mem, mem_pad)[:, -1]
            logit_pi, mx, my, sx, sy, rho, pen = self.split(out.float())
            if temperature > 0:
                k = torch.distributions.Categorical(logits=logit_pi / temperature).sample()
            else:
                k = logit_pi.argmax(-1)
            pick = lambda t: t.gather(-1, k[:, None])[:, 0]  # noqa: E731
            dx, dy = pick(mx), pick(my)
            if temperature > 0:
                ex, ey = torch.randn_like(dx), torch.randn_like(dy)
                r = pick(rho)
                dx = dx + pick(sx) * ex * math.sqrt(temperature)
                dy = dy + pick(sy) * (r * ex + torch.sqrt(1 - r ** 2) * ey) * math.sqrt(temperature)
            state = pen.argmax(-1)
            state = torch.where(done, torch.full_like(state, END), state)
            row = torch.stack([dx, dy, state.float()], -1)
            row[state == END, :2] = 0
            rows = torch.cat([rows, row[:, None]], 1)
            done |= state == END
            jumps.append(jumping.clone())
            jumping = state == LIFT
            if done.all():
                break
        scale = 1.0 + (JUMP - 1.0) * torch.stack(jumps, 1).float()[..., None].cpu().numpy()
        out = []
        for seq, k in zip(rows.cpu().numpy(), scale):
            seq = seq.copy()
            seq[:, :2] *= k
            ends = np.nonzero(seq[:, 2] == END)[0]
            seq = seq[:ends[0]] if ends.size else seq
            seq = seq.copy()
            seq[:, 2] = (seq[:, 2] == LIFT).astype(np.float32)
            if len(seq):
                seq[-1, 2] = 1.0
            out.append(seq)
        return out

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, data: dict) -> "StrokeGenerator":
        model = cls(StrokeConfig(**data["config"]))
        model.load_state_dict(data["model"])
        return model


def rows_for_training(seq: np.ndarray, step: float, max_len: int) -> np.ndarray | None:
    """Cache rows (pixels, lift 0/1) -> model rows (step units, pen 0/1/2 + END row)."""
    if len(seq) == 0 or len(seq) + 1 > max_len:
        return None
    rows = np.zeros((len(seq) + 1, 3), np.float32)
    rows[:-1, :2] = seq[:, :2] / step
    rows[:-1, 2] = seq[:, 2]
    jump = np.concatenate([[True], seq[:-1, 2] > 0.5])
    rows[:-1][jump, :2] /= JUMP
    rows[-1, 2] = END
    return rows


def pen_radius(images) -> float:
    """A hand's pen half-width in pixels: median ink depth along its pen paths."""
    from scipy.ndimage import distance_transform_edt

    from ..generate import centreline

    depths = []
    for im in images:
        line = centreline(im)
        if line is not None:
            depths.append(distance_transform_edt(im > 0.5)[line])
    return float(np.median(np.concatenate(depths))) if depths else 2.0


def draw_letters(model: StrokeGenerator, reference_images: dict[str, np.ndarray],
                 content_images: dict[str, np.ndarray], chars: dict[str, str],
                 keys: list[str] | None = None, batch_size: int = 32,
                 match_size: bool = True, temperature: float = 0.0) -> dict[str, np.ndarray]:
    """Write ``keys`` (default: every content key) in the hand of ``reference_images``.

    Each letter is written as pen moves, drawn at the writer's measured pen
    width, and (``match_size``) brought to the writer's proportions as the
    image model's letters are (``generate.match_seed_proportions``). Images are
    128 px, ink 1.
    """
    from .. import strokes as pen
    from ..generate import match_seed_proportions
    from ..intake import IntakeConfig

    device = next(model.parameters()).device
    size = model.config.image_size
    keys = [k for k in (keys or list(content_images)) if k in content_images]

    def small(images: list[np.ndarray]) -> torch.Tensor:
        t = torch.from_numpy(np.stack(images).astype(np.float32))[:, None]
        return F.adaptive_avg_pool2d(t, size) if t.shape[-1] != size else t

    refs = small(list(reference_images.values()))[None].to(device)
    radius = pen_radius(list(reference_images.values()))
    canvas = next(iter(content_images.values())).shape
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        n = len(chunk)
        content = small([content_images[k] for k in chunk]).to(device)
        rows = model.write(content, refs.expand(n, -1, -1, -1, -1),
                           torch.ones(n, refs.shape[1], dtype=torch.bool, device=device),
                           temperature=temperature)
        for key, seq in zip(chunk, rows):
            seq = seq.copy()
            seq[:, :2] *= model.config.step
            out[key] = pen.render(pen.from_sequence(seq), canvas, radius)
    if match_size:
        out, _ = match_seed_proportions(out, {}, reference_images, chars, IntakeConfig().baseline)
    return out
