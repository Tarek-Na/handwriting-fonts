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


def test_a_small_model_memorises_a_letter_and_writes_it_back():
    model = _tiny()
    rows = torch.from_numpy(_letter())[None]
    mask = torch.ones(rows.shape[:2], dtype=torch.bool)
    content = torch.zeros(1, 1, 16, 16)
    content[..., 4:12, 6] = 1
    refs = torch.rand(1, 2, 1, 16, 16)
    ref_mask = torch.ones(1, 2, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        model.train()
        loss = model.loss(model(content, refs, ref_mask, rows), rows, mask)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 5, (first, loss.item())
    written = model.write(content, refs, ref_mask)[0]           # back in steps
    want = np.array(_SEQ, np.float32) / 3.0
    assert len(written) == len(want), "the letter ends where it should"
    assert written[:, 2].sum() == 2, "two strokes"
    assert np.abs(written[:, :2] - want[:, :2]).max() < 0.5, written[:, :2] - want[:, :2]
