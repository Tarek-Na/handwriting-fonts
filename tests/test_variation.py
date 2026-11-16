"""The variation model: versions of a form in a writer's style (step A6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.models.variation import VariationConfig, VariationModel  # noqa: E402

K, ISO = 4, 3


def _toy(n: int, rng: np.random.Generator):
    """Writers with a hidden trait t: their isolated letters (forms 0-2) sit at t,
    their joined form (3) at -2 t; each copy adds noise 0.3."""
    t = rng.choice([-1.0, 1.0], size=n)
    iso = t[:, None, None] + 0.3 * rng.standard_normal((n, ISO, K))
    x = -2 * t[:, None] + 0.3 * rng.standard_normal((n, K))
    to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (to(x), torch.full((n,), 3), to(iso), torch.arange(ISO).repeat(n, 1),
            torch.ones(n, ISO, dtype=torch.bool), t)


def _trained(style: bool) -> VariationModel:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = VariationModel(VariationConfig(forms=4, k=K, latent=2, style_dim=8, hidden=32, form_dim=4,
                                           style=style))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(600):
        x, form, ix, iform, imask, _ = _toy(128, rng)
        loss, _ = model.loss(x, model.condition(form, ix, iform, imask))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def test_the_style_encoder_carries_a_writers_trait_into_their_versions():
    rng = np.random.default_rng(1)
    x, form, ix, iform, imask, t = _toy(200, rng)
    for style, expect in ((True, 0.5), (False, None)):
        model = _trained(style)
        with torch.no_grad():
            drawn = model.sample(model.condition(form, ix, iform, imask), 8).mean(1).mean(-1).numpy()
        if expect:
            assert np.corrcoef(drawn, t)[0, 1] < -0.9, "versions move with the writer's trait"
            assert np.abs(drawn - (-2 * t)).mean() < expect
        else:
            assert abs(np.corrcoef(drawn, t)[0, 1]) < 0.3, "without style, every writer is alike"


def test_versions_vary_and_the_checkpoint_round_trips():
    model = _trained(True)
    rng = np.random.default_rng(2)
    _, form, ix, iform, imask, _ = _toy(4, rng)
    cond = model.condition(form, ix, iform, imask)
    torch.manual_seed(0)
    drawn = model.sample(cond, 16)
    assert drawn.shape == (4, 16, K)
    assert drawn.std(1).mean() > 0.05, "versions of one form differ"
    again = VariationModel.from_checkpoint(model.checkpoint()).eval()
    torch.manual_seed(0)
    assert torch.allclose(again.sample(again.condition(form, ix, iform, imask), 16), drawn)


def test_letter_dropout_hides_letters_but_never_all_and_never_adds_one():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import drop_letters

    rng = np.random.default_rng(0)
    mask = np.ones((2000, 20), bool)
    mask[:, 15:] = False                                   # five letters this writer did not write
    mask[0] = False
    mask[0, 3] = True                                      # a writer with a single letter
    kept = drop_letters(mask, 0.3, rng)
    assert not (kept & ~mask).any(), "a missing letter stays missing"
    assert kept[0, 3], "a writer's only letter is kept"
    assert (kept.sum(1) >= 1).all()
    assert abs(kept[1:, :15].mean() - 0.7) < 0.02
