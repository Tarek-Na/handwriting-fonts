"""Pen strokes from a glyph image, and glyph images from pen strokes (step A3).

A stroke model (``hfont.models.strokegen``) learns letters as sequences of
pen moves, not pixels. No public Latin or Arabic set with pen recordings is
free to use, so the sequences are recovered from images: each glyph's pen
path (``hfont.generate.centreline``) is cut into strokes and put in one fixed
pen order. The order is a convention, not the writer's real one; a model only
needs it to be the same for every glyph.

* ``trace_strokes``: pen path -> strokes, each an (N, 2) array of (x, y).
  The path is a graph whose nodes are stroke ends and crossings; at a
  crossing a stroke goes on along the straightest continuation, so the bar of
  a ``t`` and its stem come out as two strokes, as a hand writes them.
* ``order_strokes``: each stroke starts at its upper (then left) end, and
  strokes go top to bottom, left to right by their start.
* ``to_sequence`` / ``from_sequence``: strokes <-> (T, 3) rows of
  ``(dx, dy, lift)``, resampled every ``step`` pixels; ``lift`` is 1 on the
  last point of a stroke.
* ``render``: strokes drawn with a round pen of a given half-width.
"""

from __future__ import annotations

import numpy as np

_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(mask: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(y + dy, x + dx) for dy, dx in _STEPS
            if 0 <= y + dy < h and 0 <= x + dx < w and mask[y + dy, x + dx]]


def _edges(line: np.ndarray) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]]:
    """The path's edges: pixel runs between nodes (ends, crossings), and the nodes."""
    pixels = set(zip(*np.nonzero(line)))
    degree = {p: len(_neighbours(line, *p)) for p in pixels}
    nodes = {p for p, d in degree.items() if d != 2}
    seen_steps: set[frozenset] = set()
    edges = []
    for start in sorted(nodes):
        for nxt in _neighbours(line, *start):
            if frozenset((start, nxt)) in seen_steps:
                continue
            path, prev, cur = [start], start, nxt
            seen_steps.add(frozenset((start, nxt)))
            while cur not in nodes:
                path.append(cur)
                ahead = [n for n in _neighbours(line, *cur) if n != prev and frozenset((cur, n)) not in seen_steps]
                if not ahead:
                    break
                prev, cur = cur, ahead[0]
                seen_steps.add(frozenset((prev, cur)))
            path.append(cur)
            if len(path) > 1:
                edges.append(path)
    # Closed loops with no node at all (an ``o``): start at the top.
    covered = {p for e in edges for p in e} | nodes
    rest = pixels - covered
    while rest:
        start = min(rest)
        path, prev, cur = [start], None, start
        while True:
            ahead = [n for n in _neighbours(line, *cur) if n != prev and n in rest and n not in path]
            if not ahead:
                break
            prev, cur = cur, ahead[0]
            path.append(cur)
        path.append(start)
        edges.append(path)
        rest -= set(path)
    return edges, nodes


def _direction(path: list[tuple[int, int]], at_start: bool, reach: int = 8) -> np.ndarray:
    """Unit direction of travel leaving ``path`` from one end, inwards."""
    pts = np.array(path if at_start else path[::-1], float)
    v = pts[min(reach, len(pts) - 1)] - pts[0]
    n = np.linalg.norm(v)
    return v / n if n else v


def trace_strokes(line: np.ndarray, min_length: int = 3) -> list[np.ndarray]:
    """Strokes of a pen path (bool image) as (N, 2) arrays of (x, y) pixel centres."""
    edges, nodes = _edges(line)
    degree = {p: len(_neighbours(line, *p)) for p in nodes}
    # A short edge is a spur only when it hangs off a crossing; a short piece
    # standing alone (the dot of an i) is a stroke.
    edges = [e for e in edges if len(e) >= min_length
             or (degree.get(e[0], 2) <= 1 and degree.get(e[-1], 2) <= 1)]
    edges += [[p] for p, d in degree.items() if d == 0]
    # A crossing in a skeleton is a small cluster of junction pixels, joined by
    # edges a pixel or two long. Each cluster is one node.
    junctions = sorted(p for p, d in degree.items() if d >= 3)
    cluster: dict[tuple[int, int], int] = {}
    for p in junctions:
        if p in cluster:
            continue
        cluster[p], todo = len(cluster), [p]
        while todo:
            q = todo.pop()
            for r in junctions:
                if r not in cluster and max(abs(r[0] - q[0]), abs(r[1] - q[1])) <= 3:
                    cluster[r] = cluster[p]
                    todo.append(r)
    edges = [e for e in edges
             if not (len(e) < 2 * min_length and e[0] in cluster and e[-1] in cluster
                     and cluster[e[0]] == cluster[e[-1]])]
    used = [False] * len(edges)
    ends: dict[object, list[tuple[int, bool]]] = {}
    for i, e in enumerate(edges):
        ends.setdefault(cluster.get(e[0], e[0]), []).append((i, True))
        ends.setdefault(cluster.get(e[-1], e[-1]), []).append((i, False))

    def near(p):
        """Edge ends at the same node as ``p``."""
        return ends.get(cluster.get(p, p), [])

    strokes = []
    for i in sorted(range(len(edges)), key=lambda i: -len(edges[i])):
        if used[i]:
            continue
        used[i] = True
        path = list(edges[i])
        for forward in (True, False):               # extend through crossings both ways
            while True:
                tip = path[-1] if forward else path[0]
                heading = -_direction(path, at_start=not forward)
                best, best_cos = None, 0.5          # at most 60 degrees off straight
                for j, at_start in near(tip):
                    if used[j]:
                        continue
                    cos = float(np.dot(heading, _direction(edges[j], at_start)))
                    if cos > best_cos:
                        best, best_cos = (j, at_start), cos
                if best is None:
                    break
                j, at_start = best
                used[j] = True
                nxt = edges[j] if at_start else edges[j][::-1]
                path = path + nxt[1:] if forward else nxt[::-1][:-1] + path
        strokes.append(np.array([(x, y) for y, x in path], float))
    return order_strokes(strokes)


def order_strokes(strokes: list[np.ndarray]) -> list[np.ndarray]:
    """Each stroke from its upper (then left) end; strokes top to bottom, left to right."""
    def key(p):
        return (round(p[1] / 4), p[0])
    out = []
    for s in strokes:
        closed = len(s) > 2 and np.abs(s[0] - s[-1]).max() <= 1
        if closed:
            k = min(range(len(s)), key=lambda i: key(s[i]))
            s = np.concatenate([s[k:], s[1:k + 1]])
        elif key(s[-1]) < key(s[0]):
            s = s[::-1]
        out.append(s)
    return sorted(out, key=lambda s: key(s[0]))


def _resample(stroke: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] == 0:
        return stroke[:1]
    n = max(int(np.ceil(dist[-1] / step)), 1)
    at = np.linspace(0.0, dist[-1], n + 1)
    return np.stack([np.interp(at, dist, stroke[:, 0]), np.interp(at, dist, stroke[:, 1])], axis=1)


def to_sequence(strokes: list[np.ndarray], step: float = 3.0,
                origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Strokes -> (T, 3) float32 rows ``(dx, dy, lift)`` in pixels, from ``origin``."""
    rows, here = [], np.array(origin, float)
    for s in strokes:
        pts = _resample(s, step)
        for k, p in enumerate(pts):
            rows.append((p[0] - here[0], p[1] - here[1], 1.0 if k == len(pts) - 1 else 0.0))
            here = p
    return np.array(rows, np.float32).reshape(-1, 3)


def from_sequence(seq: np.ndarray, origin: tuple[float, float] = (0.0, 0.0)) -> list[np.ndarray]:
    """(T, 3) ``(dx, dy, lift)`` rows -> strokes; the move onto a stroke's first point is a jump."""
    pts = np.cumsum(seq[:, :2], axis=0) + np.asarray(origin, float)
    strokes, start = [], 0
    for i, lift in enumerate(seq[:, 2]):
        if lift > 0.5:
            strokes.append(pts[start:i + 1])
            start = i + 1
    if start < len(pts):
        strokes.append(pts[start:])
    return [s for s in strokes if len(s)]


def render(strokes: list[np.ndarray], size: int | tuple[int, int], radius: float) -> np.ndarray:
    """Strokes drawn with a round pen of half-width ``radius``; ink 1, float32."""
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import line as draw_line

    from .generate import _draw

    h, w = (size, size) if isinstance(size, int) else size
    mask = np.zeros((h, w), bool)
    for s in strokes:
        pts = np.round(s).astype(int)
        if len(pts) == 1:
            pts = np.concatenate([pts, pts])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            rr, cc = draw_line(y0, x0, y1, x1)
            keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            mask[rr[keep], cc[keep]] = True
    if not mask.any():
        return mask.astype(np.float32)
    return _draw(distance_transform_edt(~mask), radius)


def glyph_sequence(image: np.ndarray, step: float = 3.0) -> np.ndarray | None:
    """A glyph image straight to its pen sequence (None when it has no pen path)."""
    from .generate import centreline

    line = centreline(image)
    if line is None:
        return None
    return to_sequence(trace_strokes(line), step)
