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


def _arc_resample(path: np.ndarray, n: int) -> np.ndarray:
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if seg[-1] == 0:
        return np.repeat(path[:1], n, 0)
    at = np.linspace(0.0, seg[-1], n)
    return np.stack([np.interp(at, seg, path[:, 0]), np.interp(at, seg, path[:, 1])], 1)


def extend_ends(moved: np.ndarray, ids: np.ndarray, target: np.ndarray, target_ids: np.ndarray,
                cover: float = 3.0, min_tail: float = 4.0, share: float = 0.25) -> np.ndarray:
    """A fit's stroke ends carried out to where the writer's strokes end.

    ``register`` keeps moves smooth, so a template stroke stops short of a
    long tail or connecting stroke the writer drew (on AHAWP the fitted forms
    are 0.89 of the real copies' width). For each end of each fitted stroke:
    the nearest point of the writer's stroke (``target``, its points in stroke
    order, ``target_ids``), and the part of that stroke beyond it, away from
    the fitted stroke, that no fitted point lies within ``cover`` px of. When
    that tail is at least ``min_tail`` px long, the fitted stroke's last
    ``share`` of points is spread evenly along its own path and the tail.
    Point count and order are kept, so the result is still moves of the
    template's points.
    """
    out = np.asarray(moved, float).copy()
    target = np.asarray(target, float)
    for k in np.unique(ids):
        idx = np.flatnonzero(ids == k)
        if len(idx) < 3:
            continue
        for at_end in (True, False):
            order = idx if at_end else idx[::-1]                    # the end in question is order[-1]
            stroke = out[order]
            end, before = stroke[-1], stroke[-2]
            j = int(np.argmin(((target - end) ** 2).sum(1)))
            tid = target_ids[j]
            tpts = np.flatnonzero(target_ids == tid)
            pos = int(np.searchsorted(tpts, j))
            best = None
            for step in (1, -1):                                     # the target stroke's two directions
                run = tpts[pos::step] if step == 1 else tpts[pos::-1]
                if len(run) < 2:
                    continue
                path = target[run]
                if np.dot(path[min(2, len(path) - 1)] - path[0], end - before) <= 0:
                    continue                                         # runs back along the fitted stroke
                others = np.delete(out, order[-max(2, int(share * len(order))):], 0)
                tail = [path[0]]
                for p in path[1:]:
                    if len(others) and np.sqrt(((others - p) ** 2).sum(1)).min() < cover:
                        break                                        # reaches ink a fitted point already covers
                    tail.append(p)
                tail = np.array(tail)
                length = np.linalg.norm(np.diff(tail, axis=0), axis=1).sum() if len(tail) > 1 else 0.0
                if length >= min_tail and (best is None or length > best[0]):
                    best = (length, tail)
            if best is None:
                continue
            gap = float(np.median(np.linalg.norm(np.diff(stroke, axis=0), axis=1))) or 1.0
            m = max(2, int(round(share * len(order))))
            while m < len(order):                                    # enough points to keep spacing even
                head = stroke[-m:]
                total = np.linalg.norm(np.diff(head, axis=0), axis=1).sum() + best[0]
                if total / (m - 1) <= 2.5 * gap:
                    break
                m += 1
            head = stroke[-m:]
            out[order[-m:]] = _arc_resample(np.concatenate([head, best[1][1:]]), m)
    return out


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


def local_moves(moved: np.ndarray, affine_fit: np.ndarray, template: np.ndarray) -> np.ndarray:
    """What a fit moves beyond its affine part, in the template's frame (M, 2).

    ``affine_fit`` is the 6 numbers (B row by row, then t) of ``affine``;
    ``moved`` is the fitted points. Zero when the fit is purely affine.
    """
    b, t = affine_fit[:4].reshape(2, 2), affine_fit[4:]
    return (moved - t) @ np.linalg.inv(b).T - template
