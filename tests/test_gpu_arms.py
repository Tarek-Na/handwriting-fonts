"""Code for the GPU experiments, checked on CPU before any GPU time is spent.

Each arm must start from the trained model without changing it (identity at
initialisation), and must do what it is for once it trains.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _pair(levels=(0, 1)):
    """A model as trained today, and the same weights with finer attention added."""
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    torch.manual_seed(0)
    base = GlyphGenerator(GeneratorConfig(style_attention=True))
    torch.nn.init.normal_(base.ref_attention.proj.weight, std=0.05)  # a trained, live block
    fine = GlyphGenerator(GeneratorConfig(style_attention=True, attention_levels=levels))
    missing, unexpected = fine.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("decoder.attend.") for k in missing)
    return base, fine


def _inputs():
    import torch

    torch.manual_seed(1)
    return (torch.randn(2, 1, 128, 128), torch.randn(2, 5, 1, 128, 128),
            torch.ones(2, 5, dtype=torch.bool), torch.tensor([3, 40]))


def test_finer_attention_starts_as_the_identity():
    """Warm-starting into 16x16 and 32x32 attention must not change a single pixel."""
    import torch

    base, fine = _pair()
    content, refs, mask, ids = _inputs()
    with torch.no_grad():
        before = base(content, refs, mask, ids)["image"]
        after = fine(content, refs, mask, ids)["image"]
    assert torch.equal(before, after)


def test_finer_attention_reads_the_references_once_trained():
    """Change one reference: a live 16x16/32x32 block must move the output further."""
    import torch

    _, fine = _pair()
    content, refs, mask, ids = _inputs()
    other = refs.clone()
    other[:, 2] = torch.randn(2, 1, 128, 128)

    def change():
        with torch.no_grad():
            return (fine(content, refs, mask, ids)["image"]
                    - fine(content, other, mask, ids)["image"]).abs().mean()

    idle = change()
    for block in fine.decoder.attend.values():
        torch.nn.init.normal_(block.proj.weight, std=0.05)
    assert change() > idle * 1.2


def test_generation_batches_reference_features_of_any_shape():
    import torch

    from hfont.models.generator import expand_ref_feats

    _, fine = _pair()
    _, refs, mask, _ = _inputs()
    with torch.no_grad():
        _, feats = fine.encode_style_full(refs[:1], mask[:1])
    assert set(feats) == {"coarse", "fine"} and feats["fine"].shape[-1] == 8
    wide = expand_ref_feats(feats, 6)
    assert wide["fine"].shape[0] == 6 and wide["coarse"].shape[0] == 6


def test_warm_start_accepts_new_identity_modules_and_nothing_else(tmp_path):
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator
    from hfont.train import Trainer
    from hfont.utils import EMA

    base, _ = _pair()
    path = tmp_path / "base.pt"
    torch.save({"generator": base.state_dict(), "ema": EMA(base).state_dict(), "step": 7}, path)

    def trainer(gen):
        return SimpleNamespace(generator=gen, ema=EMA(gen), device="cpu",
                               cfg=SimpleNamespace(ema_decay=0.999))

    fine = GlyphGenerator(GeneratorConfig(style_attention=True, attention_levels=(0, 1)))
    t = trainer(fine)
    Trainer.warm_start(t, str(path))
    assert torch.equal(fine.ref_attention.proj.weight, base.ref_attention.proj.weight)
    assert all(name in t.ema.shadow for name, _ in fine.named_parameters()), \
        "new modules must be averaged too, or the EMA model never learns them"

    wider = GlyphGenerator(GeneratorConfig(style_attention=True, base_channels=32))
    with pytest.raises(RuntimeError):
        Trainer.warm_start(trainer(wider), str(path))


def test_warm_start_adds_the_reference_weighting_table(tmp_path):
    """G4 from a checkpoint without the table: its name has no dot to log by."""
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator
    from hfont.train import Trainer
    from hfont.utils import EMA

    base, _ = _pair()
    path = tmp_path / "base.pt"
    torch.save({"generator": base.state_dict(), "ema": EMA(base).state_dict(), "step": 7}, path)
    g4 = GlyphGenerator(GeneratorConfig(style_attention=True, reference_weighting=True))
    t = SimpleNamespace(generator=g4, ema=EMA(g4), device="cpu",
                        cfg=SimpleNamespace(ema_decay=0.999))
    Trainer.warm_start(t, str(path))
    assert "ref_affinity" in t.ema.shadow


def test_the_attention_arm_is_one_command_line_switch():
    from hfont.cli import train_config

    args = SimpleNamespace(recipe="recommended", cache="c", out="o", config=None,
                           set=["generator.attention_levels=[0,1]", "init_from=runs/hw40k/best.pt"])
    cfg = train_config(args)
    assert cfg.generator.attention_levels == (0, 1) and cfg.generator.style_attention
    assert cfg.init_from == "runs/hw40k/best.pt"


def test_self_reconstruction_gives_the_target_as_its_only_reference():
    """G2: the branch must show the model the answer, and only the answer."""
    import torch

    from hfont.models.losses import GlyphLoss
    from hfont.train import self_reconstruction

    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, content, refs, mask, char_id, ref_char_ids=None):
            seen.update(refs=refs, mask=mask, ref_ids=ref_char_ids)
            return {"image": refs[:, 0]}            # copies its reference perfectly

    target = torch.rand(3, 1, 32, 32) * 2 - 1
    batch = {"target": target, "content": torch.zeros_like(target),
             "char_id": torch.tensor([0, 1, 2])}
    loss = self_reconstruction(Spy(), GlyphLoss(), batch)
    assert seen["refs"].shape == (3, 1, 1, 32, 32) and torch.equal(seen["refs"][:, 0], target)
    assert seen["mask"].all() and seen["mask"].shape == (3, 1)
    assert seen["ref_ids"].tolist() == [[0], [1], [2]], "its reference is the target's own letter"
    wrong = self_reconstruction(lambda c, r, m, i, j: {"image": -r[:, 0]}, GlyphLoss(), batch)
    assert loss < wrong, "copying the reference must be what the branch rewards"


def test_self_reconstruction_trains_the_attention():
    """Its gradient must reach the reference-attention blocks, or it teaches them nothing."""
    import torch

    from hfont.models.losses import GlyphLoss
    from hfont.train import self_reconstruction

    _, fine = _pair()
    for block in fine.decoder.attend.values():
        torch.nn.init.normal_(block.proj.weight, std=0.05)
    content, refs, _, ids = _inputs()
    batch = {"target": refs[:, 0].tanh(), "content": content, "char_id": ids}
    self_reconstruction(fine, GlyphLoss(), batch).backward()
    for name, block in fine.decoder.attend.items():
        assert block.to_kv.weight.grad.abs().sum() > 0, f"no gradient into attention {name}"


CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_local"


@pytest.mark.skipif(not (CACHE / "meta.json").is_file(), reason="no local glyph cache")
def test_one_real_training_step_with_the_new_arms(tmp_path):
    """Smoke run: G1 and G2 together through Trainer.train_step on real cached fonts."""
    import math
    from dataclasses import replace

    from hfont.data.dataset import DatasetConfig
    from hfont.models.generator import GeneratorConfig
    from hfont.train import Trainer, TrainConfig

    cfg = TrainConfig(
        cache_dir=str(CACHE), out_dir=str(tmp_path), steps=1, batch_size=2, num_workers=0,
        device="cpu", amp=False, warmup_steps=1,
        generator=GeneratorConfig(base_channels=16, max_channels=64, style_dim=32,
                                  style_attention=True, attention_levels=(0, 1)),
        data=DatasetConfig(n_refs=(2, 4), epoch_size=4),
    )
    cfg = replace(cfg, loss=replace(cfg.loss, self_recon=1.0))
    trainer = Trainer(cfg)
    parts = trainer.train_step(next(iter(trainer.train_loader)))
    assert "self_recon" in parts and math.isfinite(parts["self_recon"])
    assert math.isfinite(parts["g_loss"])


def test_projection_loss_grows_with_distance_where_l1_does_not():
    """G5: a stroke 12 px off must cost more than one 6 px off; L1 sees them as equal."""
    import torch

    from hfont.models.losses import projection_loss, weighted_l1

    def stroke(col):
        im = -torch.ones(1, 1, 64, 64)
        im[..., 10:54, col:col + 4] = 1.0
        return im

    target = stroke(20)
    near, far = stroke(26), stroke(32)
    assert projection_loss(target, target) < 1e-6
    assert abs(projection_loss(near, target) - 6) < 0.1, "measured in pixels of displacement"
    assert projection_loss(far, target) > projection_loss(near, target) * 1.8
    assert torch.isclose(weighted_l1(near, target, 4.0), weighted_l1(far, target, 4.0))

    pred = near.clone().requires_grad_(True)
    projection_loss(pred, target).backward()
    assert pred.grad.abs().sum() > 0


def test_projection_term_is_off_unless_asked():
    import torch

    from hfont.models.losses import GlyphLoss, LossConfig

    a, b = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    _, off = GlyphLoss(LossConfig()).reconstruction(a, b)
    total, on = GlyphLoss(LossConfig(projection=1.0)).reconstruction(a, b)
    assert "projection" not in off and on["projection"] > 0 and torch.isfinite(total)


def _weighted():
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    base, _ = _pair()
    gen = GlyphGenerator(GeneratorConfig(style_attention=True, reference_weighting=True))
    missing, _ = gen.load_state_dict(base.state_dict(), strict=False)
    assert missing == ["ref_affinity"]
    return base, gen


def test_reference_weighting_starts_with_no_preference():
    """G4: a zero table must leave the trained model exactly as it was."""
    import torch

    base, gen = _weighted()
    content, refs, mask, ids = _inputs()
    ref_ids = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    with torch.no_grad():
        assert torch.equal(base(content, refs, mask, ids)["image"],
                           gen(content, refs, mask, ids, ref_ids)["image"])


def test_reference_weighting_prefers_related_letters_and_learns():
    import torch

    _, gen = _weighted()
    content, refs, mask, ids = _inputs()
    ref_ids = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    with torch.no_grad():
        plain = gen(content, refs, mask, ids, ref_ids)["image"]
        gen.ref_affinity[ids[0], 12] = 8.0          # target 0 should read reference 12
        favoured = gen(content, refs, mask, ids, ref_ids)["image"]
    assert (favoured[0] - plain[0]).abs().mean() > 1e-4, "the preference changed nothing"
    assert torch.equal(favoured[1], plain[1]), "a preference for one target leaked to another"

    gen(content, refs, mask, ids, ref_ids)["image"].mean().backward()
    assert gen.ref_affinity.grad[ids[0], 10:15].abs().sum() > 0


@pytest.mark.skipif(not (CACHE / "meta.json").is_file(), reason="no local glyph cache")
def test_samples_say_which_letter_each_reference_is():
    from hfont.data.dataset import DatasetConfig, GlyphPairDataset, GlyphStore

    store = GlyphStore(CACHE)
    ds = GlyphPairDataset(store, DatasetConfig(split="train", epoch_size=20))
    keys = [g.key for g in store.charset]
    for i in range(20):
        s = ds[i]
        font = ds.font_ids[int(s["font_id"])]
        n = int(s["ref_mask"].sum())
        for j in range(n):
            ref = store.image(font, keys[int(s["ref_char_ids"][j])])
            assert abs(float(s["refs"][j, 0].mean()) - float(ref.mean() * 2 - 1)) < 1e-5


def test_font_assembly_passes_the_reference_letters():
    """Without the letters at inference, a learned table would be ignored silently."""
    import numpy as np
    import torch

    from hfont.charset import LATIN_CORE
    from hfont.generate import LoadedModel, generate_rasters

    _, gen = _weighted()
    gen.eval()
    keys = [g.key for g in LATIN_CORE]
    rng = np.random.default_rng(0)
    refs = {k: (rng.random((128, 128)) > 0.9).astype(np.float32) for k in keys[:4]}
    content = {k: (rng.random((128, 128)) > 0.9).astype(np.float32) for k in keys[4:6]}
    model = LoadedModel(gen, LATIN_CORE, 128)
    model.device = torch.device("cpu")
    before, _ = generate_rasters(model, refs, content)
    with torch.no_grad():
        gen.ref_affinity[4, 1] = 8.0
    after, _ = generate_rasters(model, refs, content)
    assert not np.allclose(before[keys[4]], after[keys[4]])
    assert np.allclose(before[keys[5]], after[keys[5]])
