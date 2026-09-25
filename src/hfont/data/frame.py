"""The one definition of where a glyph sits on the canvas.

Both the corpus renderer and handwriting intake place glyphs through this
module, and that is the whole point of it.

They used to compute placement independently. The renderer trusted the font's
designed baseline (y = 0) and sized each font by the extent of all 75 glyphs;
intake, which has no designed baseline to trust, estimated one from the bottoms
of letters like ``a e o n`` and sized by the 24 letters a person writes. For a
typeset font the two agree. For a handwriting font they do not: Segoe Print came
through intake 17% larger than the model had ever seen it, because handwriting
letters do not sit on the designer's line. The model was trained on one framing
and would have been fed another, and nothing would have errored.

The fix is to define the frame only in terms of what handwriting also has:
ink bounds of the reference letters, and a baseline estimated from them.
Rendering a font now asks the same question a photograph can answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Lowercase letters that sit on the baseline with no descender. Their bottom
#: edges are the most reliable evidence of where a writer's baseline is.
BASELINE_CHARS = "acemnorsuvwxz"


@dataclass(frozen=True)
class InkFrame:
    """Vertical frame in the source's own units, with y pointing *down*."""

    baseline: float
    ascent: float
    descent: float


#: Percentile of ascent and descent the canvas is sized to fit, instead of the
#: maximum. See frame_from_boxes for why this is 90 and not 100.
EXTENT_PERCENTILE = 90.0


def frame_from_boxes(
    boxes: dict[str, tuple[float, float, float, float]],
    percentile: float = EXTENT_PERCENTILE,
) -> InkFrame:
    """Estimate baseline, ascent and descent from per-character ink boxes.

    ``boxes`` maps a character to ``(x0, y0, x1, y1)`` with y increasing
    downward — image rows, or font units negated.

    **Why a percentile and not the maximum.** Sizing the canvas to fit the single
    most extreme letter means one sprawling capital or one deep descender shrinks
    the entire alphabet. Designed typefaces are regular enough that this costs
    nothing — their ascenders all reach the same line — but real handwriting is
    not, and the result was that photographed hands arrived at **x-height ~21px
    where the corpus sits at ~39px**, roughly half scale. That size difference is
    the axis every photographed hand collapsed onto, with
    corr(x-height, position on it) = −0.80 (output/review/STYLE_COLLAPSE.md).

    Framing on the 90th percentile instead is, measurably, a **no-op on
    typefaces**: across twelve Windows fonts the rendering is byte-identical
    (IoU 1.000, 0.0% x-height change), because for a font the 90th percentile of
    ascent *is* the maximum. On the three photographed hands it nearly doubles
    the glyph, 21.0px to 41.6px, for a mean of 0.20% of a letter's ink pushed to
    the canvas border (worst letter 5.7%).

    That asymmetry is the point, and it is why this needs no re-render and no
    retrain: the corpus the model learned from is unchanged, while real
    handwriting now arrives at the scale that corpus is drawn at.

    Pass ``percentile=100`` for the old behaviour.
    """
    if not boxes:
        raise ValueError("no ink boxes to frame")
    bottoms = np.asarray([b[3] for b in boxes.values()], dtype=np.float64)
    anchors = [b[3] for c, b in boxes.items() if c in BASELINE_CHARS]
    # Median, not mean: one letter written with a flourish below the line
    # should not drag the whole alphabet down with it.
    baseline = float(np.median(anchors)) if anchors else float(np.percentile(bottoms, 75))
    ascent = float(np.percentile([baseline - b[1] for b in boxes.values()], percentile))
    descent = float(np.percentile([b[3] - baseline for b in boxes.values()], percentile))
    return InkFrame(baseline=baseline, ascent=max(ascent, 1e-6), descent=max(descent, 0.0))


def canvas_scale(frame: InkFrame, size: int, baseline: float, margin: float) -> float:
    """Largest scale at which the frame's ascent and descent both fit.

    ``baseline`` and ``margin`` are fractions of the canvas height.
    """
    above = size * baseline - size * margin
    below = size - size * margin - size * baseline
    limits = [above / frame.ascent]
    if frame.descent > 0:
        limits.append(below / frame.descent)
    return min(limits)
