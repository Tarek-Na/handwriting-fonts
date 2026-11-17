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


def test_kmedoids_finds_clear_groups_and_drops_tiny_ones():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation_mix import kmedoids

    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 1, (20, 2)), rng.normal(20, 1, (15, 2)), rng.normal(-40, 1, (2, 2))])
    dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    medoids = kmedoids(dist, 3, min_size=5)
    assert len(medoids) == 2, "the two-member group is too small to stand for a way of writing"
    assert sorted(m < 20 for m in medoids) == [False, True], "one medoid in each real group"


def test_the_choice_head_picks_the_template_a_writer_builds_with():
    """A writer's hidden trait decides both their isolated letters and which of two templates
    their joined form uses; with style the head learns it, without style it cannot."""
    from hfont.models.variation import VariationConfig, VariationModel
    import torch.nn.functional as F

    def toy(n, rng):
        t = rng.integers(0, 2, n)
        iso = (2 * t - 1)[:, None, None] + 0.3 * rng.standard_normal((n, ISO, K))
        to = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        return to(iso), torch.arange(ISO).repeat(n, 1), torch.ones(n, ISO, dtype=torch.bool), torch.tensor(t)

    results = {}
    for style in (True, False):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        model = VariationModel(VariationConfig(forms=ISO + 2, k=K, latent=2, style_dim=8, hidden=32, form_dim=4,
                                               style=style, templates=2, base_forms=1))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        base, valid = torch.zeros(128, dtype=torch.long), torch.ones(128, 2, dtype=torch.bool)
        for _ in range(300):
            ix, ip, im, t = toy(128, rng)
            s = model.style_vector(ix, ip, im) if style else None
            loss = F.cross_entropy(model.choice_logits(base, s, valid), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ix, ip, im, t = toy(400, np.random.default_rng(5))
        with torch.no_grad():
            s = model.style_vector(ix, ip, im) if style else None
            picked = model.choice_logits(torch.zeros(400, dtype=torch.long), s, torch.ones(400, 2, dtype=torch.bool))
        results[style] = float((picked.argmax(-1) == t).float().mean())
    assert results[True] > 0.95 and results[False] < 0.65, results


def _toy_pca(rng):
    """(Uses the pipeline's component count, not this file's toy K.) Two forms with arcs as templates and full-rank components (so changes are representable)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import K as VK

    temps, pca = {}, {}
    for name, m in (("a", 12), ("b", 14)):
        x = np.linspace(-20, 20, m)
        temps[name] = (np.stack([x, 0.02 * x ** 2], 1).astype(np.float32), np.zeros(m, int))
        full = np.linalg.qr(rng.standard_normal((2 * m, 2 * m)))[0].T          # orthonormal rows
        pca[name] = (np.zeros(2 * m, np.float32), full[:VK].astype(np.float32), np.ones(VK, np.float32))
    return pca, temps


def test_writer_augment_is_the_identity_without_a_change_and_widens_with_one():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import K as VK, WriterAugment

    rng = np.random.default_rng(0)
    pca, temps = _toy_pca(rng)
    names = ["a", "b"]
    changer = WriterAugment(pca, names, temps)
    x = torch.zeros(2, VK)
    form = torch.tensor([0, 1])
    same = changer(x, form, torch.eye(2).repeat(2, 1, 1))
    assert torch.allclose(same, x, atol=1e-5)
    wide = torch.tensor([[1.5, 0.0], [0.0, 1.0]]).repeat(2, 1, 1)
    out = changer(x, form, wide)
    mean, comps, scale = pca["a"]
    moves = (out[0].numpy() * scale) @ comps                          # back to moves of form a
    want = (temps["a"][0] @ wide[0].numpy().T - temps["a"][0]).ravel()
    target = comps.T @ (comps @ want)                                 # what K components can hold
    assert np.allclose(moves, target, atol=1e-4)
    assert np.abs(target).max() > 1, "the change is not lost in the projection"


def test_writer_contrast_rewards_matching_views_and_spares_the_same_writer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import writer_contrast

    torch.manual_seed(0)
    s = torch.randn(8, 16)
    ids = torch.arange(8)
    assert writer_contrast(s, s + 0.01 * torch.randn(8, 16), ids) < 0.05
    assert writer_contrast(s, torch.randn(8, 16), ids) > 1.0
    dup = s.clone()
    dup[1] = s[0]                                                      # writer 0 twice, as rows 0 and 1
    loss_same = writer_contrast(dup, dup, torch.tensor([0, 0, 2, 3, 4, 5, 6, 7]))
    loss_diff = writer_contrast(dup, dup, torch.arange(8))
    assert loss_same < 0.05 < loss_diff, "an identical row of the same writer is not a negative"


def test_changing_a_writers_letters_together_matches_changing_each():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import WriterAugment

    rng = np.random.default_rng(0)
    pca, temps = _toy_pca(rng)
    changer = WriterAugment(pca, ["a", "b"], temps)
    x = torch.randn(3, 2, changer.h.shape[1])
    a = torch.from_numpy(WriterAugment.sample(3, rng, 0.2))
    forms = torch.tensor([0, 1])
    together = changer.letters(x, forms, a)
    each = changer(x, forms.expand(3, -1), a[:, None].expand(-1, 2, -1, -1))
    assert torch.allclose(together, each, atol=1e-4)


def test_the_conditional_gaussian_reads_the_writer_from_their_isolated_letters():
    """The fair no-network baseline: a joined form that follows a trait seen in the isolated
    letters is predicted from them, and its versions spread like one writer's copies."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from variation import K as VK, ConditionalGaussian

    rng = np.random.default_rng(0)
    pca = {n: None for n in ("a", "b", "a.fina")}

    def writer(t):
        return {"a": {0: t + 0.3 * rng.standard_normal(VK)}, "b": {0: -t + 0.3 * rng.standard_normal(VK)},
                "a.fina": {c: 2 * t + 0.2 * rng.standard_normal(VK) for c in range(10)}}

    basis = rng.standard_normal((2, VK))                              # style lives in a few directions
    traits = {f"w{i}": rng.standard_normal(2) @ basis for i in range(60)}
    xs = {w: writer(t) for w, t in traits.items()}
    cg = ConditionalGaussian(xs, pca, list(xs))
    assert set(cg.fits) == {"a.fina"}, "only joined forms get a regression"
    t = rng.standard_normal(2) @ basis
    drawn = cg.sample(writer(t), "a.fina", 400, 1.0, rng)
    assert np.abs(drawn.mean(0) - 2 * t).mean() < 0.35, "the versions sit where this writer writes"
    assert abs(drawn.std(0).mean() - 0.2) < 0.08, "and spread like one writer's copies"
