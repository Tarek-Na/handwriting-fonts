"""The stroke model learns pen sequences and writes them back (step A3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.models.strokegen import END, StrokeConfig, StrokeGenerator, rows_for_training  # noqa: E402


def _tiny() -> StrokeGenerator:
    torch.manual_seed(0)
    return StrokeGenerator(StrokeConfig(d_model=32, heads=4, layers=2, style_layers=1, mixtures=3,
                                        max_len=32, image_size=16, ref_tokens=2, dropout=0.0))


#: An L: a jump in from the corner, down 8 steps, lift, a jump, a bar right 4 steps.
_SEQ = ([(30.0, 15.0, 0.0)] + [(0.0, 3.0, 0.0)] * 7 + [(0.0, 3.0, 1.0)]
        + [(15.0, -6.0, 0.0)] + [(3.0, 0.0, 0.0)] * 3 + [(3.0, 0.0, 1.0)])


def _letter():
    return rows_for_training(np.array(_SEQ, np.float32), step=3.0, max_len=32)


def test_training_rows_end_with_an_end_row_in_step_units():
    rows = _letter()
    assert rows[-1, 2] == END and (rows[-1, :2] == 0).all()
    assert rows[1, 1] == 1.0, "3 px is one step"
    assert tuple(rows[0, :2]) == (1.0, 0.5) and tuple(rows[9, :2]) == (0.5, -0.2), "jumps in tens"
    assert rows_for_training(np.zeros((40, 3), np.float32), 3.0, 32) is None, "too long"


def test_the_letter_encoder_is_cheap_enough_for_30_references():
    """Each training step encodes batch x up to 30 reference letters: the
    encoder must not spend full resolution on wide layers (it once cost 0.68
    GMAC a letter, 0.86 s a step on an L4)."""
    from hfont.models.strokegen import LetterEncoder

    enc = LetterEncoder(256)
    macs = []

    def count(module, inputs, output):
        k = module.kernel_size[0] * module.kernel_size[1]
        macs.append(output.numel() * module.in_channels * k // module.groups)

    for m in enc.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(count)
    out = enc(torch.zeros(1, 1, 64, 64))
    assert out.shape == (1, 256, 8, 8), "still an 8 x 8 grid of tokens"
    assert sum(macs) < 0.3e9, f"{sum(macs) / 1e9:.2f} GMAC a letter"


def test_a_small_model_memorises_a_letter_and_writes_it_back():
    model = _tiny()
    rows = torch.from_numpy(_letter())[None]
    mask = torch.ones(rows.shape[:2], dtype=torch.bool)
    content = torch.zeros(1, 1, 16, 16)
    content[..., 4:12, 6] = 1
    refs = torch.rand(1, 2, 1, 16, 16)
    ref_mask = torch.ones(1, 2, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    # Annealed to zero: at a fixed rate the tiny model circles the letter and
    # where it stands at the last step depends on its initial weights.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400)
    first = None
    for _ in range(400):
        model.train()
        loss = model.loss(model(content, refs, ref_mask, rows), rows, mask)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    assert loss.item() < first - 5, (first, loss.item())
    written = model.write(content, refs, ref_mask)[0]           # back in steps
    want = np.array(_SEQ, np.float32) / 3.0
    assert len(written) == len(want), "the letter ends where it should"
    assert written[:, 2].sum() == 2, "two strokes"
    assert np.abs(written[:, :2] - want[:, :2]).max() < 0.5, written[:, :2] - want[:, :2]


def test_a_hand_with_few_letters_left_gets_the_references_it_has():
    """Validation at step 2000 crashed on a hand whose pool, without the
    target's look-alikes, held fewer than the minimum 8 references."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_strokes import ref_count

    rng = np.random.default_rng(0)
    assert all(ref_count(rng, (8, 30), 5) == 5 for _ in range(20))
    drawn = {ref_count(rng, (8, 30), 12) for _ in range(200)}
    assert min(drawn) >= 8 and max(drawn) <= 12
    assert {ref_count(rng, (8, 30), 100) for _ in range(500)} <= set(range(8, 31))


def test_the_decoder_is_told_where_the_pen_is():
    """Without it, placing an F's bar means adding up every move so far."""
    model = StrokeGenerator(StrokeConfig(d_model=32, heads=4, layers=1, style_layers=1, mixtures=3,
                                         max_len=32, image_size=16, ref_tokens=2, position=True))
    rows = torch.from_numpy(_letter())[None]
    feats = model.inputs(rows)
    where = feats[0, :, 6:8].numpy() * 128 / 3.0            # back to steps
    pts = np.cumsum(np.array(_SEQ, np.float32)[:, :2] / 3.0, axis=0)
    assert np.allclose(where[1:len(pts) + 1], pts, atol=1e-4), "the pen's place after each move"
    assert (where[0] == 0).all(), "nothing drawn before the start"
    assert StrokeGenerator(StrokeConfig(**{**model.config.__dict__, "position": False})).move_in[0].in_features == 6


def test_validation_scores_the_same_letters_every_time():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_strokes import fixed_batches

    class Draws:                                   # samples come from the dataset's own rng
        seed = 1
        rng = np.random.default_rng(1)

        def __len__(self):
            return 4

        def __getitem__(self, _):
            return self.rng.random()

    data = Draws()
    plain = [[data[i] for i in range(4)] for _ in range(2)]
    assert plain[0] != plain[1], "without a reset each pass draws new samples"
    passes = []
    for _ in range(2):
        fixed_batches(data, 2)
        passes.append([data[i] for i in range(4)])
    assert passes[0] == passes[1]
