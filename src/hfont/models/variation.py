"""A conditional VAE over a writer's skeleton moves (step A6).

Each form ``f`` has a template pen path; a written copy of it is the
template's points moved, and the moves are reduced to ``K`` principal
components per form, scaled to unit variance on the training writers
(``x`` below, shape (K,)). The model learns the distribution of ``x`` for
a form given the writer:

* a style encoder reads the writer's isolated letters (one copy of each,
  as on a sheet: their ``x`` and which letter it is) and pools them, with
  attention, into a style vector ``s``;
* a conditional prior p(z | f, s), an encoder q(z | x, f, s) and a decoder
  giving a Gaussian over x (a mean and a variance per component), all
  small MLPs; the loss is the Gaussian negative log-likelihood of ``x``
  plus ``beta`` times KL(q || prior).

The decoder's variance is learned, not fixed: with a fixed one, variation
between copies smaller than it is treated as noise and the model draws
the same letter every time (posterior collapse; the test shows it).
Sampling z from the prior, and x from the decoder's Gaussian, gives as
many versions of a form as wanted, in the writer's style. With ``style=False`` the writer is ignored: the same
model learns one distribution per form, for the ablation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class VariationConfig:
    forms: int = 65
    k: int = 24                 # components per form
    latent: int = 8
    style_dim: int = 32
    hidden: int = 128
    form_dim: int = 16
    style: bool = True
    beta: float = 1.0
    #: Templates per form (step A6b). Above 1, ``forms`` counts (form,
    #: template) pairs and a choice head picks the template for a base form.
    templates: int = 1
    base_forms: int = 0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class StyleEncoder(nn.Module):
    """A set of (letter, x) -> style vector, by attention pooling (missing letters masked)."""

    def __init__(self, c: VariationConfig):
        super().__init__()
        self.item = _mlp(c.k + c.form_dim, c.hidden, c.style_dim)
        self.score = nn.Linear(c.style_dim, 1)

    def forward(self, x: torch.Tensor, form: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, L, K), form (B, L, form_dim), mask (B, L) bool
        h = self.item(torch.cat([x, form], -1))
        a = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
        return (torch.softmax(a, -1).unsqueeze(-1) * h).sum(1)


class VariationModel(nn.Module):
    def __init__(self, config: VariationConfig):
        super().__init__()
        c = self.config = config
        self.form = nn.Embedding(c.forms, c.form_dim)
        self.style = StyleEncoder(c)
        cond = c.form_dim + (c.style_dim if c.style else 0)
        self.prior = _mlp(cond, c.hidden, 2 * c.latent)
        self.encoder = _mlp(c.k + cond, c.hidden, 2 * c.latent)
        self.decoder = _mlp(c.latent + cond, c.hidden, 2 * c.k)
        if c.templates > 1:
            self.base = nn.Embedding(c.base_forms, c.form_dim)
            self.choice = _mlp(c.form_dim + (c.style_dim if c.style else 0), c.hidden, c.templates)

    def style_vector(self, iso_x: torch.Tensor, iso_form: torch.Tensor, iso_mask: torch.Tensor) -> torch.Tensor:
        return self.style(iso_x, self.form(iso_form), iso_mask)

    def choice_logits(self, base: torch.Tensor, style: torch.Tensor | None, valid: torch.Tensor) -> torch.Tensor:
        """(B,) base forms -> (B, templates) logits for which template the writer builds it like.

        ``valid`` (B, templates) masks templates a form does not have.
        """
        parts = [self.base(base)] + ([style] if self.config.style else [])
        return self.choice(torch.cat(parts, -1)).masked_fill(~valid, -1e9)

    def _decode(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, lv = self.decoder(torch.cat([z, cond], -1)).chunk(2, -1)
        return mu, lv.clamp(-8.0, 4.0)

    def condition(self, form: torch.Tensor, iso_x: torch.Tensor | None, iso_form: torch.Tensor | None,
                  iso_mask: torch.Tensor | None, style: torch.Tensor | None = None) -> torch.Tensor:
        parts = [self.form(form)]
        if self.config.style:
            parts.append(style if style is not None else self.style_vector(iso_x, iso_form, iso_mask))
        return torch.cat(parts, -1)

    def loss(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mu_q, lv_q = self.encoder(torch.cat([x, cond], -1)).chunk(2, -1)
        mu_p, lv_p = self.prior(cond).chunk(2, -1)
        z = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp()
        mu_x, lv_x = self._decode(z, cond)
        rec = (0.5 * ((x - mu_x) ** 2 / lv_x.exp() + lv_x)).sum(-1).mean()
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(-1).mean()
        return rec + self.config.beta * kl, {"rec": float(rec), "kl": float(kl)}

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n: int, temperature: float = 1.0) -> torch.Tensor:
        """(B, cond) -> (B, n, K) versions: z from the prior, x from the decoder's Gaussian."""
        mu_p, lv_p = self.prior(cond).chunk(2, -1)
        z = mu_p[:, None] + temperature * torch.randn(cond.shape[0], n, mu_p.shape[-1]) * (0.5 * lv_p).exp()[:, None]
        mu_x, lv_x = self._decode(z, cond[:, None].expand(-1, n, -1))
        return mu_x + temperature * torch.randn_like(mu_x) * (0.5 * lv_x).exp()

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "VariationModel":
        model = cls(VariationConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state"])
        return model
