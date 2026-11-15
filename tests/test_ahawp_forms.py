"""Positional-form completion: the score and the writer split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.evaluate.metrics import aligned_centreline_f1, centreline_f1  # noqa: E402


def _hook(h=60, w=90, x0=10, y0=10, pen=3):
    im = np.zeros((h, w), np.float32)
    im[y0:y0 + 30, x0:x0 + pen] = 1                     # an upright
    im[y0 + 30 - pen:y0 + 30, x0:x0 + 40] = 1           # and a stroke along the line
    return im


def test_where_a_letter_sits_in_its_image_does_not_count():
    a = _hook()
    b = _hook(h=80, w=140, x0=70, y0=35)               # same letter, other crop
    assert centreline_f1(a, b[:60, :90]) < 0.5, "unaligned, the plain score misses it"
    assert aligned_centreline_f1(a, b) == 1.0


def test_shape_and_size_still_count():
    a = _hook()
    longer = np.zeros((60, 120), np.float32)
    longer[10:40, 10:13] = 1
    longer[37:40, 10:90] = 1                            # the stroke twice as long
    assert aligned_centreline_f1(a, longer) < 0.85
    other = np.zeros((60, 90), np.float32)
    other[30:33, 10:80] = 1                             # no upright at all
    assert aligned_centreline_f1(a, other) < 0.8
    assert aligned_centreline_f1(a, np.zeros_like(a)) == 0.0


def test_every_writer_is_on_exactly_one_side():
    from hfont.data.ahawp import split_of

    writers = [f"user{n:03d}" for n in range(1, 83)]
    sides = {s: [w for w in writers if split_of(w) == s] for s in ("train", "val", "test")}
    assert [len(sides[s]) for s in ("train", "val", "test")] == [49, 17, 16]
    assert sum(len(v) for v in sides.values()) == 82


def test_a_letter_survives_the_trip_to_the_model_canvas_and_back():
    from hfont.models.formgen import from_canvas, to_canvas

    scan = np.zeros((128, 300), np.float32)
    scan[40:100, 200:206] = 1                          # an upright, far right in its crop
    scan[94:100, 120:206] = 1                          # its stroke along the line
    canvas = to_canvas(scan)
    assert canvas.shape == (64, 256)
    cols = np.nonzero(canvas.any(axis=0))[0]
    assert abs((cols[0] + cols[-1]) / 2 - 128) <= 1, "ink centred across"
    assert np.nonzero(canvas.any(axis=1))[0][0] == 20, "rows kept: the cell's own height"
    assert aligned_centreline_f1(from_canvas(canvas) > 0.5, scan) > 0.9


def test_the_form_model_draws_one_canvas_per_letter():
    import torch

    from hfont.models.formgen import FormConfig, FormGenerator

    model = FormGenerator(FormConfig(width=16, d=32, mix_layers=1))
    b, k = 2, 3
    logits = model(torch.rand(b, 1, 64, 256), torch.rand(b, k, 1, 64, 256),
                   torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
                   torch.tensor([0, 5]), torch.tensor([2, 3]))
    assert logits.shape == (b, 1, 64, 256)
    target = (torch.rand(b, 1, 64, 256) > 0.9).float()
    assert model.loss(logits, target)["loss"].item() > 0


def test_the_style_letters_are_encoded_small():
    """Up to 20 style letters a sample: at full size they were the model's cost."""
    import torch

    from hfont.models.formgen import FormConfig, FormGenerator

    model = FormGenerator(FormConfig(width=16, d=32, mix_layers=1))
    seen = []
    model.style_encoder.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[-2:]))
    model.style(torch.rand(1, 4, 1, 64, 256), torch.ones(1, 4, dtype=torch.bool))
    assert tuple(seen[0]) == (32, 128)


def test_a_refining_model_reads_the_donor_form_beside_the_letter():
    import torch

    from hfont.models.formgen import FormConfig, FormGenerator

    model = FormGenerator(FormConfig(width=16, d=32, mix_layers=1, donor=True))
    logits = model(torch.rand(2, 2, 64, 256), torch.rand(2, 3, 1, 64, 256), torch.ones(2, 3, dtype=torch.bool),
                   torch.tensor([0, 5]), torch.tensor([2, 3]))
    assert logits.shape == (2, 1, 64, 256)
    donor_only = torch.zeros(1, 2, 64, 256)
    donor_only[0, 1, 30:34, 40:200] = 1               # a stroke in the donor channel
    a = model(donor_only, torch.zeros(1, 1, 1, 64, 256), torch.ones(1, 1, dtype=torch.bool),
              torch.tensor([0]), torch.tensor([2]))
    b = model(torch.zeros(1, 2, 64, 256), torch.zeros(1, 1, 1, 64, 256), torch.ones(1, 1, dtype=torch.bool),
              torch.tensor([0]), torch.tensor([2]))
    assert not torch.allclose(a, b), "the donor reaches the output"


def test_neighbours_never_list_a_writer_as_its_own():
    import json

    import pytest

    path = Path(__file__).resolve().parents[1] / "data/ahawp/neighbours.json"
    if not path.exists():
        pytest.skip("needs scripts/ahawp_forms.py neighbours")
    from hfont.data.ahawp import split_of

    ranked = json.loads(path.read_text())
    assert all(w not in r and all(split_of(n) == "train" for n in r) for w, r in ranked.items())
