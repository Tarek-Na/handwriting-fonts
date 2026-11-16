"""A letter as a template skeleton plus a writer's moves (step A5).

A writer's form is described by where each point of a shared template
skeleton goes: the template is one training writer's pen path for that form,
resampled to points; a writer's version is the template with every point
moved. The moves are found by non-rigid point registration (coherent point
drift, Myronenko & Song 2010), which needs no point-to-point match given in
advance and survives an extra or missing stroke (it lets target points go
unmatched). The moves are then what a model predicts: from a writer's moves
on letters they wrote, the moves on forms they did not.

* ``skeleton_points``: an ink image -> (N, 2) float (x, y) points every
  ``step`` px along its pen path, plus each point's stroke number, centred on
  the ink's bounding box (the box the score centres on).
* ``register``: template points -> where they go on a target point set.
* ``render_points``: points with their stroke numbers drawn with a round pen.
"""

from __future__ import annotations

import numpy as np


def skeleton_points(image: np.ndarray, step: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
    """(points (N, 2) x, y centred on the ink box, stroke index (N,)), or None for no pen path."""
    from .generate import centreline
    from .strokes import _resample, trace_strokes

    line = centreline(image)
    if line is None:
        return None
    pts, ids = [], []
    for k, stroke in enumerate(trace_strokes(line)):
        p = _resample(stroke, step) if len(stroke) > 1 else stroke
        pts.append(p)
        ids.append(np.full(len(p), k))
    pts = np.concatenate(pts)
    rows, cols = np.nonzero(np.asarray(image) > 0.5)
    centre = np.array([(cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2])
    return pts - centre, np.concatenate(ids)


def _posterior(x: np.ndarray, t: np.ndarray, sigma2: float, outliers: float) -> np.ndarray:
    m, n = len(t), len(x)
    p = np.exp(-((x[None] - t[:, None]) ** 2).sum(-1) / (2 * sigma2))
    c = 2 * np.pi * sigma2 * outliers / (1 - outliers) * m / n
    return p / (p.sum(0, keepdims=True) + c)


def affine(template: np.ndarray, target: np.ndarray, outliers: float = 0.1,
           iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Affine coherent point drift: (B (2, 2), t (2,)) with template @ B.T + t on ``target``.

    The part of a writer's difference that one linear map explains: size,
    width, slant, shear.
    """
    y = np.asarray(template, float)
    x = np.asarray(target, float)
    m, n = len(y), len(x)
    b, t = np.eye(2), np.zeros(2)
    sigma2 = ((x[None] - y[:, None]) ** 2).sum() / (2 * m * n)
    for _ in range(iterations):
        p = _posterior(x, y @ b.T + t, sigma2, outliers)
        p1, pt1 = p.sum(1), p.sum(0)
        np_ = p1.sum()
        mx, my = x.T @ pt1 / np_, y.T @ p1 / np_
        xh, yh = x - mx, y - my
        a = xh.T @ p.T @ yh
        b = a @ np.linalg.inv(yh.T @ (p1[:, None] * yh) + 1e-9 * np.eye(2))
        t = mx - b @ my
        sigma2 = max(((pt1 * (xh ** 2).sum(1)).sum() - np.trace(a @ b.T)) / (2 * np_), 1e-3)
        if sigma2 < 0.05:
            break
    return b, t


def register(template: np.ndarray, target: np.ndarray, beta: float = 12.0, smooth: float = 2.0,
             outliers: float = 0.01, iterations: int = 60, pre_affine: bool = True,
             affine_outliers: float = 0.01) -> np.ndarray:
    """Coherent point drift: the template's points moved onto ``target`` (both (·, 2)).

    An affine fit first (``affine``), then smooth local moves: ``beta`` (px)
    is how far a move spreads to neighbouring points, ``smooth`` how strongly
    moves are kept smooth; ``outliers`` is the share of target points allowed
    to match nothing (a stroke the template lacks).
    """
    y = np.asarray(template, float)
    x = np.asarray(target, float)
    if pre_affine:
        b, t0 = affine(y, x, affine_outliers)
        y = y @ b.T + t0
    m, n = len(y), len(x)
    g = np.exp(-((y[:, None] - y[None]) ** 2).sum(-1) / (2 * beta ** 2))
    w = np.zeros_like(y)
    t = y.copy()
    sigma2 = ((x[None] - y[:, None]) ** 2).sum() / (2 * m * n)
    for _ in range(iterations):
        p = _posterior(x, t, sigma2, outliers)
        p1, pt1 = p.sum(1), p.sum(0)
        px = p @ x
        a = p1[:, None] * g + smooth * sigma2 * np.eye(m)
        w = np.linalg.solve(a, px - p1[:, None] * y)
        t = y + g @ w
        np_ = p1.sum()
        sigma2 = max(((pt1 * (x ** 2).sum(1)).sum() - 2 * (px * t).sum()
                      + (p1 * (t ** 2).sum(1)).sum()) / (2 * np_), 1e-3)
        if sigma2 < 0.05:
            break
    return t


def render_points(points: np.ndarray, ids: np.ndarray, radius: float, margin: int = 8) -> np.ndarray:
    """Points joined stroke by stroke and drawn with a round pen; image cut to the ink plus ``margin``."""
    from .strokes import render

    pts = np.asarray(points, float)
    origin = pts.min(0) - margin - radius
    pts = pts - origin
    size = (int(np.ceil(pts[:, 1].max() + margin + radius)) + 1,
            int(np.ceil(pts[:, 0].max() + margin + radius)) + 1)
    strokes = [pts[ids == k] for k in np.unique(ids)]
    return render(strokes, size, radius)
