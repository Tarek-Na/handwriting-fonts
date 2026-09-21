"""Raster glyph -> cubic Bezier outlines.

Classical, not learned. The brief is right that this is a solved problem, but
"solved" does not mean "any implementation will do" — the quality of this stage
sets a hard ceiling on the exported font, and the usual failure is an outline
that looks fine at 500px and is visibly lumpy at text size.

The pipeline is the standard one:

1. **Sub-pixel contour extraction.** Marching squares on the *grayscale* field
   at the 0.5 level, not on a thresholded bitmap. The generator's output is
   antialiased, and that antialiasing is real sub-pixel position information —
   thresholding first throws it away and leaves staircase edges that no amount
   of later smoothing recovers.
2. **Corner detection.** Letterforms are not smooth curves; they are smooth
   curves joined at deliberate corners. Stem ends, serifs and the junction of a
   bowl and a stem must stay sharp, so corners are found first and fitting
   never crosses one.
3. **Curve fitting.** Schneider's algorithm between corners: fit one cubic,
   measure the worst deviation, reparameterize, and split only where it will
   not converge. Fewer, longer curves are better here — every extra on-curve
   point is somewhere the outline can kink.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

Point = np.ndarray  # shape (2,)


@dataclass
class BezierPath:
    """One closed contour as a start point plus cubic segments.

    Each segment is ``(control1, control2, end)``, matching how both the
    TrueType/CFF pens and SVG expect to receive curves.
    """

    start: tuple[float, float]
    segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )

    @property
    def points(self) -> list[tuple[float, float]]:
        out = [self.start]
        for _, _, end in self.segments:
            out.append(end)
        return out

    def signed_area(self) -> float:
        """Shoelace area of the on-curve polygon; sign gives orientation."""
        pts = self.points
        total = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            total += x0 * y1 - x1 * y0
        return total / 2.0

    def reverse(self) -> "BezierPath":
        """Flip winding direction, preserving the curve geometry."""
        pts = self.points
        if len(pts) < 2:
            return self
        rev_segments = []
        n = len(self.segments)
        for i in range(n - 1, -1, -1):
            c1, c2, _ = self.segments[i]
            prev = self.start if i == 0 else self.segments[i - 1][2]
            rev_segments.append((c2, c1, prev))
        return BezierPath(start=self.segments[-1][2] if n else self.start,
                          segments=rev_segments)

    def transform(self, fn) -> "BezierPath":
        return BezierPath(
            start=fn(self.start),
            segments=[(fn(a), fn(b), fn(c)) for a, b, c in self.segments],
        )


# --------------------------------------------------------------------------- #
# 1. contour extraction
# --------------------------------------------------------------------------- #

def extract_contours(image: np.ndarray, level: float = 0.5) -> list[np.ndarray]:
    """Sub-pixel closed contours of the ink region, as (N, 2) x/y arrays."""
    from skimage import measure

    # Pad by one pixel so that ink touching the border still yields a closed
    # loop rather than an open curve marching squares cannot terminate.
    padded = np.pad(image.astype(np.float64), 1, mode="constant", constant_values=0.0)
    raw = measure.find_contours(padded, level)

    contours: list[np.ndarray] = []
    for contour in raw:
        # find_contours yields (row, col); the rest of the pipeline wants (x, y).
        #
        # The 0.5 is a convention change, not a fudge. find_contours places
        # pixel *centres* at integer indices, while the rasterizer treats pixel
        # i as covering [i, i+1] — centre at i+0.5. Without the shift every
        # traced outline sits half a pixel up and to the left of the ink it came
        # from, which is invisible glyph-by-glyph and costs about 15 points of
        # round-trip IoU across a font. The -1.0 removes the one-pixel pad.
        pts = np.column_stack([contour[:, 1] + 0.5 - 1.0, contour[:, 0] + 0.5 - 1.0])
        # It closes loops by repeating the first point; drop the duplicate.
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) >= 4:
            contours.append(pts)
    return contours


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def drop_specks(
    contours: list[np.ndarray], min_area: float = 2.0, min_points: int = 6
) -> list[np.ndarray]:
    """Remove contours too small to be intentional.

    A generated raster usually carries a few isolated grey blobs. Left in, each
    becomes a tiny closed path in the font — invisible at text size, but it
    inflates the file, upsets outline validators, and can flip winding in the
    middle of a glyph.
    """
    return [
        c for c in contours
        if len(c) >= min_points and abs(polygon_area(c)) >= min_area
    ]


# --------------------------------------------------------------------------- #
# 2. corner detection + resampling
# --------------------------------------------------------------------------- #

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _end_tangent(pts: np.ndarray, span: float = 2.5, from_start: bool = True) -> np.ndarray:
    """Direction the run leaves an endpoint in, measured over an arc window.

    Taking the tangent from the immediately adjacent point is the obvious thing
    and it is badly wrong here. A marching-squares contour turns a corner with a
    sub-pixel diagonal jog, so the first step out of a run that is in truth
    perfectly horizontal can read as a 37-degree diaponal. That error is then
    multiplied by the control-point distance — of order the whole chord — and
    bows the curve right out of the glyph.

    Measuring to the first point at least ``span`` pixels away averages the jog
    out while staying local enough to track real curvature.
    """
    ordered = pts if from_start else pts[::-1]
    origin = ordered[0]
    deltas = ordered[1:] - origin
    if len(deltas) == 0:
        return np.zeros(2)
    dists = np.linalg.norm(deltas, axis=1)
    far = np.nonzero(dists >= span)[0]
    idx = int(far[0]) if far.size else int(np.argmax(dists))

    # Average the directions up to that point rather than using the single
    # chord, so a slightly curved run is not biased by where the cut landed.
    window = deltas[: idx + 1]
    weights = np.linalg.norm(window, axis=1)
    if weights.sum() <= 1e-12:
        return _unit(deltas[-1])
    return _unit((window * weights[:, None]).sum(axis=0))


def _tangent_at(pts: np.ndarray, index: int, span: float = 2.5) -> np.ndarray:
    """Forward tangent at an interior point, averaged over both sides."""
    forward = _end_tangent(pts[index:], span, from_start=True)
    backward = _end_tangent(pts[: index + 1], span, from_start=False)
    return _unit(forward - backward)


def find_corners(pts: np.ndarray, window: int = 4, threshold_deg: float = 48.0) -> list[int]:
    """Indices where the contour turns sharply enough to be a real corner.

    The turn is measured across a window rather than between adjacent points:
    adjacent-point angles on a marching-squares contour are dominated by
    quantization noise and would mark half the outline as corners.
    """
    n = len(pts)
    if n < 2 * window + 1:
        return []

    idx = np.arange(n)
    before = pts[(idx - window) % n]
    after = pts[(idx + window) % n]
    incoming = pts - before
    outgoing = after - pts

    in_n = incoming / np.clip(np.linalg.norm(incoming, axis=1, keepdims=True), 1e-12, None)
    out_n = outgoing / np.clip(np.linalg.norm(outgoing, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip((in_n * out_n).sum(axis=1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))

    candidates = np.nonzero(angle > threshold_deg)[0]
    if candidates.size == 0:
        return []

    # Sharp turns produce a run of adjacent candidates; keep the sharpest of
    # each run so one corner does not become five.
    corners: list[int] = []
    run = [candidates[0]]
    for prev, cur in zip(candidates, candidates[1:]):
        if cur - prev <= window:
            run.append(cur)
        else:
            corners.append(int(max(run, key=lambda i: angle[i])))
            run = [cur]
    corners.append(int(max(run, key=lambda i: angle[i])))

    # The first and last runs may wrap around index 0.
    if len(corners) > 1 and (corners[0] + n - corners[-1]) <= window:
        drop = corners[0] if angle[corners[0]] < angle[corners[-1]] else corners[-1]
        corners.remove(drop)
    return sorted(corners)


def rdp_indices(pts: np.ndarray, epsilon: float) -> list[int]:
    """Ramer-Douglas-Peucker, returning kept indices of an open polyline."""
    n = len(pts)
    if n < 3:
        return list(range(n))

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        seg = pts[hi] - pts[lo]
        seg_len = np.linalg.norm(seg)
        chunk = pts[lo + 1 : hi]
        if seg_len < 1e-12:
            dist = np.linalg.norm(chunk - pts[lo], axis=1)
        else:
            # Explicit 2-D cross product; np.cross on 2-vectors is deprecated
            # in NumPy 2 and this runs often enough to matter.
            delta = chunk - pts[lo]
            dist = np.abs(seg[0] * delta[:, 1] - seg[1] * delta[:, 0]) / seg_len
        if dist.size == 0:
            continue
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            split = lo + 1 + k
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return list(np.nonzero(keep)[0])


# --------------------------------------------------------------------------- #
# 3. cubic fitting (Schneider)
# --------------------------------------------------------------------------- #

def _bezier_point(ctrl: np.ndarray, t: float) -> np.ndarray:
    mt = 1.0 - t
    return (
        mt**3 * ctrl[0]
        + 3 * mt**2 * t * ctrl[1]
        + 3 * mt * t**2 * ctrl[2]
        + t**3 * ctrl[3]
    )


def _bezier_points(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a cubic at many parameters at once -> (N, 2).

    The per-point Python version of this was the pipeline's worst bottleneck.
    Curve fitting evaluates the whole sample set on every fit attempt, every
    reparameterization step and every split, so a Python loop over points gets
    multiplied by the recursion depth and the number of runs. On clean glyphs
    that was merely slow; on an undertrained model's noisy output, which yields
    far more contours and corners, export stopped finishing at all.
    """
    mt = 1.0 - t
    return (
        (mt**3)[:, None] * ctrl[0]
        + (3 * mt**2 * t)[:, None] * ctrl[1]
        + (3 * mt * t**2)[:, None] * ctrl[2]
        + (t**3)[:, None] * ctrl[3]
    )


def _chord_parameterize(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 1e-12 else np.linspace(0, 1, len(pts))


def _fit_one_cubic(
    pts: np.ndarray, u: np.ndarray, t0: np.ndarray, t1: np.ndarray,
    box_slack: float = 0.35,
) -> np.ndarray:
    """Least-squares control points for fixed endpoints and end tangents."""
    p0, p3 = pts[0], pts[-1]
    mt = 1.0 - u
    b0 = mt**3
    b1 = 3 * mt**2 * u
    b2 = 3 * mt * u**2
    b3 = u**3

    a0 = b1[:, None] * t0[None, :]
    a1 = b2[:, None] * t1[None, :]
    rhs = pts - (b0[:, None] * p0[None, :] + b3[:, None] * p3[None, :])

    c00 = float((a0 * a0).sum())
    c01 = float((a0 * a1).sum())
    c11 = float((a1 * a1).sum())
    x0 = float((a0 * rhs).sum())
    x1 = float((a1 * rhs).sum())

    det = c00 * c11 - c01 * c01
    chord = float(np.linalg.norm(p3 - p0))
    if abs(det) < 1e-12:
        # Degenerate system (collinear samples): fall back to the Wu/Barsky
        # heuristic of one third of the chord in each tangent direction.
        alpha0 = alpha1 = chord / 3.0
    else:
        alpha0 = (x0 * c11 - x1 * c01) / det
        alpha1 = (c00 * x1 - c01 * x0) / det

    # Control points are held inside the chord length. The unconstrained
    # least-squares solution is free to place them several chords away, which
    # fits the sampled points acceptably while bowing the curve far outside
    # them in between. On a long straight run — a serif foot, the flat of an E
    # — the normal equations are near-singular and it does exactly that, so the
    # outline overshoots the baseline by tens of units while the per-sample
    # error still looks small. A cubic whose controls lie within the chord
    # cannot stray far from it.
    limit = max(chord, 1e-6)
    if not (1e-6 < alpha0 <= limit) or not (1e-6 < alpha1 <= limit):
        alpha0 = min(max(alpha0, chord / 3.0), limit) if alpha0 > 1e-6 else chord / 3.0
        alpha1 = min(max(alpha1, chord / 3.0), limit) if alpha1 > 1e-6 else chord / 3.0

    ctrl = np.stack([p0, p0 + alpha0 * t0, p3 + alpha1 * t1, p3])

    # Hard bound on how far the curve may stray from the points it is fitting.
    #
    # A cubic lies inside the convex hull of its four control points, so
    # confining the two off-curve points to the data's bounding box (inflated by
    # the fitting tolerance) *guarantees* the curve stays there too. Clamping
    # changes the curve's shape, but if that costs accuracy the splitter sees
    # the error and subdivides, which is the correct response — whereas an
    # unclamped fit answers a slightly-off endpoint tangent by bowing the
    # outline clean out of the glyph, and nothing downstream catches it.
    lo = pts.min(axis=0) - box_slack
    hi = pts.max(axis=0) + box_slack
    ctrl[1] = np.clip(ctrl[1], lo, hi)
    ctrl[2] = np.clip(ctrl[2], lo, hi)
    return ctrl


def _max_error(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> tuple[float, int]:
    dist = np.linalg.norm(_bezier_points(ctrl, u) - pts, axis=1)
    k = int(np.argmax(dist))
    return float(dist[k]), k


def _reparameterize(pts: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step pulling each sample toward its closest point."""
    d1 = 3.0 * (ctrl[1:] - ctrl[:-1])
    d2 = 2.0 * (d1[1:] - d1[:-1])

    mt = 1.0 - u
    point = _bezier_points(ctrl, u)
    deriv = (mt**2)[:, None] * d1[0] + (2 * mt * u)[:, None] * d1[1] + (u**2)[:, None] * d1[2]
    deriv2 = mt[:, None] * d2[0] + u[:, None] * d2[1]

    diff = point - pts
    numerator = (diff * deriv).sum(axis=1)
    denominator = (deriv * deriv).sum(axis=1) + (diff * deriv2).sum(axis=1)

    # Divide only where the denominator is safe. Computing the quotient first
    # and selecting afterwards still evaluates it everywhere, which warns and
    # produces NaN at the degenerate samples.
    usable = np.abs(denominator) > 1e-12
    safe = np.where(usable, denominator, 1.0)
    stepped = np.where(usable, u - numerator / safe, u)
    return np.clip(stepped, 0.0, 1.0)


def fit_cubics(
    pts: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    tolerance: float,
    straight_tolerance: float = 0.40,
    max_depth: int = 6,
    _depth: int = 0,
) -> list[np.ndarray]:
    """Fit an open point run with as few cubics as the tolerance allows."""
    if len(pts) < 2:
        return []

    # Straight runs are emitted as straight segments and never fitted.
    #
    # This deliberately also covers the two-point case, which used to be handled
    # separately by honouring the endpoint tangents. Two points carry no
    # curvature information, so there is nothing to honour — and because fitting
    # runs on RDP output, any adjacent pair of points is straight within the
    # simplification epsilon by construction. Curving between them invents
    # detail that is not in the contour: it was placing a control point 19px out
    # along a 28-degree tangent and bowing E's top arm 9px above the glyph.
    #
    # Latin is full of long flat edges — the arms of E, the stem of H, all of L,
    # T, I, z — and they are the worst thing to hand to a tangent-based fit. The
    # endpoint tangent is measured near a corner and comes out a few degrees off;
    # harmless on a short run, but the control-point distance scales with the
    # chord, so on E's 58px arm a 14-degree error bows the outline 14px clear of
    # the glyph. Checking straightness first removes the whole failure mode, and
    # a straight edge is what these runs should produce anyway.
    chord_vec = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len > 1e-9:
        direction = chord_vec / chord_len
        offsets = pts - pts[0]
        perpendicular = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
        if float(perpendicular.max()) <= straight_tolerance:
            step = chord_vec / 3.0
            return [np.stack([pts[0], pts[0] + step, pts[-1] - step, pts[-1]])]

    u = _chord_parameterize(pts)
    ctrl = _fit_one_cubic(pts, u, t0, t1, tolerance)
    error, split = _max_error(pts, u, ctrl)

    if error < tolerance:
        return [ctrl]

    if _depth < max_depth:
        for _ in range(4):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _fit_one_cubic(pts, u, t0, t1, tolerance)
            error, split = _max_error(pts, u, ctrl)
            if error < tolerance:
                return [ctrl]

    if _depth >= max_depth or split <= 0 or split >= len(pts) - 1:
        return [ctrl]

    # Split at the worst point, with a tangent from its neighbourhood so the
    # two halves meet smoothly.
    mid_tangent = _tangent_at(pts, split)
    left = fit_cubics(pts[: split + 1], t0, -mid_tangent, tolerance,
                      straight_tolerance, max_depth, _depth + 1)
    right = fit_cubics(pts[split:], mid_tangent, t1, tolerance,
                       straight_tolerance, max_depth, _depth + 1)
    return left + right


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

@dataclass
class TraceConfig:
    #: Contour level in the antialiased image. 0.5 is the visual edge.
    level: float = 0.5
    #: Max deviation of the fitted curve from the simplified polygon, in pixels.
    tolerance: float = 0.20
    #: Polygon-approximation tolerance, in pixels. Must sit above the
    #: rasterizer's grey-level quantization (1/supersample) or the fitter will
    #: try to reproduce quantization steps as curvature.
    simplify: float = 0.04
    #: Arc length over which endpoint tangents are estimated, in pixels.
    tangent_span: float = 2.5
    #: A run whose points all lie within this distance of its chord is emitted
    #: as a straight segment. Set above the contour's noise floor, not below it.
    straight_tolerance: float = 0.40
    corner_window: int = 4
    corner_threshold_deg: float = 48.0
    min_area: float = 2.0
    min_points: int = 6
    #: Hard cap on contours fitted per glyph, largest by area first.
    #:
    #: A converged model produces two or three contours for a letter. An
    #: undertrained one produces a field of grey speckle, and every speck is a
    #: closed contour to fit — enough of them to stall export completely. No
    #: real glyph in either phase needs anything like this many, so the cap
    #: costs nothing and makes export time bounded regardless of model quality.
    max_contours: int = 24


def trace_contour(pts: np.ndarray, cfg: TraceConfig) -> BezierPath | None:
    """Fit one closed contour into a Bezier path."""
    corners = find_corners(pts, cfg.corner_window, cfg.corner_threshold_deg)

    if corners:
        # Rotate so the contour starts on a corner, then cut at every corner.
        start = corners[0]
        rolled = np.roll(pts, -start, axis=0)
        cuts = [c - start for c in corners[1:]] if len(corners) > 1 else []
        runs = []
        prev = 0
        for cut in cuts:
            runs.append(rolled[prev : cut + 1])
            prev = cut
        runs.append(np.vstack([rolled[prev:], rolled[:1]]))
        smooth_join = False
    else:
        # No corners: one closed smooth loop, cut arbitrarily but joined
        # smoothly so the seam is invisible.
        rolled = pts
        runs = [np.vstack([rolled, rolled[:1]])]
        smooth_join = True

    segments = []
    for run in runs:
        if len(run) < 2:
            continue
        # Polygon approximation first, curve fitting second — the order potrace
        # uses, and for the same reason. A marching-squares contour carries
        # quantization noise at the level of the rasterizer's grey steps, and
        # fitting directly to it makes the splitter chase that noise: an H came
        # out as 333 curves, all of them tracking jitter no one can see. RDP
        # within a known epsilon removes it and bounds the error it introduces,
        # so the total stays under epsilon + tolerance.
        keep = rdp_indices(run, cfg.simplify)
        reduced = run[keep] if len(keep) >= 2 else run

        # Tangents are still measured on the *unreduced* run, which has far more
        # points to average the jitter out of.
        if smooth_join:
            # Tangents come from across the seam, not from inside the run.
            t0 = _end_tangent(np.vstack([rolled, rolled[:1]]), cfg.tangent_span, True)
            t1 = _end_tangent(np.vstack([rolled[-1:], rolled]), cfg.tangent_span, False)
        else:
            t0 = _end_tangent(run, cfg.tangent_span, from_start=True)
            t1 = _end_tangent(run, cfg.tangent_span, from_start=False)

        for ctrl in fit_cubics(reduced, t0, t1, cfg.tolerance, cfg.straight_tolerance):
            segments.append(
                (
                    (float(ctrl[1][0]), float(ctrl[1][1])),
                    (float(ctrl[2][0]), float(ctrl[2][1])),
                    (float(ctrl[3][0]), float(ctrl[3][1])),
                )
            )

    if not segments:
        return None
    first = runs[0][0]
    return BezierPath(start=(float(first[0]), float(first[1])), segments=segments)


def trace_image(image: np.ndarray, cfg: TraceConfig | None = None) -> list[BezierPath]:
    """Full raster -> outline conversion for one glyph.

    ``image`` is (H, W) float in [0, 1] with 1 = ink. Returned paths are in
    image pixel coordinates (x right, y *down*); converting to font coordinates
    is the caller's job, since only it knows the baseline.
    """
    cfg = cfg or TraceConfig()
    contours = drop_specks(
        extract_contours(image, cfg.level), cfg.min_area, cfg.min_points
    )
    if len(contours) > cfg.max_contours:
        log.debug("glyph produced %d contours; keeping the %d largest",
                  len(contours), cfg.max_contours)
        contours = sorted(contours, key=lambda c: -abs(polygon_area(c)))[: cfg.max_contours]

    paths: list[BezierPath] = []
    for contour in contours:
        path = trace_contour(contour, cfg)
        if path is not None:
            paths.append(path)
    return paths
