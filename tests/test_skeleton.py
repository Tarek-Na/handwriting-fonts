"""Template skeletons and a writer's moves (step A5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont import skeleton  # noqa: E402
from hfont.evaluate.metrics import aligned_centreline_f1  # noqa: E402


def _arc(width: float = 60, depth: float = 20, n: int = 60) -> np.ndarray:
    x = np.linspace(-width / 2, width / 2, n)
    return np.stack([x, depth * (x / (width / 2)) ** 2], axis=1)


def test_register_moves_a_template_onto_a_stretched_copy():
    template = _arc()
    target = _arc(width=80, depth=30, n=90)                # wider, deeper, more points
    moved = skeleton.register(template, target)
    to_target = np.sqrt(((moved[:, None] - target[None]) ** 2).sum(-1)).min(1)
    assert to_target.max() < 2.0, "every template point lands on the target"
    # The 1% outlier allowance (what keeps a stray stroke from pulling the
    # template, below) leaves the last few px of the ends unmatched.
    assert moved[:, 0].min() < -34 and moved[:, 0].max() > 34, "and covers most of its width"
    assert template[:, 0].max() == 30, "unmoved, it is 20 px narrower"


def test_register_ignores_a_stroke_the_template_lacks():
    template = _arc()
    extra = np.stack([np.full(20, 0.0), np.linspace(-40, -20, 20)], axis=1)   # a separate upright
    moved = skeleton.register(template, np.concatenate([_arc(width=70, n=70), extra]))
    assert (moved[:, 1] > -10).all(), "no template point is dragged to the extra stroke"


def test_points_round_trip_through_the_pen():
    im = np.zeros((60, 120), np.float32)
    im[40:44, 10:110] = 1
    im[10:44, 100:104] = 1
    pts, ids = skeleton.skeleton_points(im)
    assert abs(pts[:, 0].mean()) < 25 and len(np.unique(ids)) >= 1
    drawn = skeleton.render_points(pts, ids, radius=2.0)
    assert aligned_centreline_f1(drawn, im) > 0.9


def test_local_moves_are_what_the_affine_does_not_explain():
    template = _arc()
    bump = np.zeros_like(template)
    bump[25:35, 1] = 3.0                                   # a local dent
    fit = np.array([1.3, 0.2, 0.0, 0.9, 5.0, -2.0])        # width, slant, height, shift
    b, t = fit[:4].reshape(2, 2), fit[4:]
    moved = (template + bump) @ b.T + t
    assert np.allclose(skeleton.local_moves(moved, fit, template), bump)
    assert np.allclose(skeleton.local_moves(template @ b.T + t, fit, template), 0)
