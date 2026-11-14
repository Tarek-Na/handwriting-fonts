"""Generated letters redrawn with the writer's pen (generate.restroke, match_pen)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.evaluate.metrics import cl_dice  # noqa: E402
from hfont.generate import _stroke_weight, match_pen, restroke  # noqa: E402


def _ring(thickness: float, size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size] - size / 2
    r = np.hypot(yy, xx)
    return (np.abs(r - 30) < thickness).astype(np.float32)


def _ell(thickness: int, size: int = 128) -> np.ndarray:
    im = np.zeros((size, size), np.float32)
    im[20:100, 40 - thickness:40 + thickness] = 1
    im[100 - thickness:100 + thickness, 40 - thickness:95] = 1
    return im


def test_a_heavy_letter_is_redrawn_thin_with_its_strokes_where_they_were():
    from scipy.ndimage import label

    heavy, thin = _ring(9), _ring(2)
    out = restroke(heavy, 1.5)
    assert _stroke_weight([out]) < 0.4 * _stroke_weight([heavy])
    assert cl_dice(out, thin) > 0.9
    assert label(out > 0.5)[1] == 1, "the stroke must not break (erosion did)"
    assert label(out < 0.5)[1] == 2, "the counter stays open"


def test_corners_of_a_heavy_letter_do_not_grow_spurs():
    out = restroke(_ell(8), 1.5)
    assert cl_dice(out, _ell(2)) > 0.9


def test_generated_letters_take_the_weight_of_the_written_ones():
    written = [_ring(2), _ell(2)]
    made = match_pen({"o": _ring(9), "L": _ell(8)}, written)
    assert abs(_stroke_weight(made.values()) - _stroke_weight(written)) < 0.25   # weight moves in whole-pixel steps
    assert match_pen({}, written) == {}


def test_a_counter_closed_by_heavy_ink_is_opened_again():
    """The model's `e` and `A` close their counters at 0.5; lighter ink there gives them away."""
    from scipy.ndimage import label

    heavy = _ring(9)
    yy, xx = np.mgrid[:128, :128] - 64
    heavy[np.hypot(yy, xx) <= 21] = 0.65            # the bowl, filled with lighter ink
    assert label(heavy < 0.5)[1] == 1, "closed at the usual cut"
    out = restroke(heavy, 1.5)
    assert label(out < 0.5)[1] == 2 and cl_dice(out, _ring(2)) > 0.8


def test_centreline_score_does_not_reward_heavy_ink():
    """clDice forgives a misplaced stroke if it is drawn heavy; this score must not."""
    from hfont.evaluate.metrics import centreline_f1

    yy, xx = np.mgrid[:128, :128] - 64

    def ring(radius, thickness):
        return (np.abs(np.hypot(yy, xx) - radius) < thickness).astype(np.float32)

    real = ring(30, 2)
    thin, heavy = ring(32, 2), ring(32, 8)
    assert cl_dice(heavy, real) > cl_dice(thin, real) + 0.3          # the bias
    assert abs(centreline_f1(heavy, real) - centreline_f1(thin, real)) < 0.05
    assert centreline_f1(real, real) == 1.0
    assert centreline_f1(ring(45, 2), real) == 0.0                   # far off: nothing matches


def test_a_cache_is_redrawn_in_one_pen_and_says_so(tmp_path):
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pen_cache import convert

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    glyphs = np.stack([_ring(9), _ell(8), _ring(3)])
    np.save(src / "packed_images.npy", np.round(glyphs * 255).astype(np.uint8))
    np.savez(src / "packed_index.npz", font_ids=np.asarray(["f"]))
    (src / "meta.json").write_text(json.dumps({"charset": "latin_core", "fonts": []}))
    assert convert(src, dst, radius=2.0) == 3
    out = np.load(dst / "packed_images.npy").astype(np.float32) / 255
    weights = [_stroke_weight([g]) for g in out]
    assert max(weights) - min(weights) < 0.3, "heavy and light glyphs come out in one pen"
    assert cl_dice(out[0], _ring(2)) > 0.9
    assert json.loads((dst / "meta.json").read_text())["pen_radius"] == 2.0


def test_a_pen_model_is_shown_letters_in_its_training_pen():
    from types import SimpleNamespace

    from hfont.generate import LoadedModel

    model = LoadedModel.__new__(LoadedModel)
    model.pen_radius = 0.0
    heavy = _ring(9)
    assert model.seen(heavy) is heavy
    model.pen_radius = 2.0
    assert _stroke_weight([model.seen(heavy)]) < 0.5 * _stroke_weight([heavy])


CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_local"


def test_training_refuses_a_cache_drawn_in_another_pen():
    import pytest

    if not (CACHE / "meta.json").is_file():
        pytest.skip("no local glyph cache")
    from hfont.data.dataset import DatasetConfig, GlyphPairDataset, GlyphStore

    with pytest.raises(ValueError, match="pen radius"):
        GlyphPairDataset(GlyphStore(CACHE), DatasetConfig(pen_radius=2.5))
