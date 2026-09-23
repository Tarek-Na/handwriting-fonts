"""Make a designed font look like it was written by a hand.

The style encoder puts every photographed hand on one shared axis and barely
distinguishes the writers along it (see output/review/STYLE_COLLAPSE.md: font
style codes deviate from the centroid in unrelated directions, mean cosine
-0.103, while the three real hands deviate in near-parallel directions at
+0.928). Stroke weight and edge softness were both eliminated as the cause, in
both directions. What is left is that a designed font is *regular* in every way
a real hand is not:

* the same letter is drawn slightly differently every time it appears;
* the stroke thickens and thins along its length as pen pressure varies;
* letters do not sit exactly on the baseline, and lean by a degree or two.

A corpus made entirely of designed fonts therefore never asks the encoder to
tell two irregular hands apart, so it has not learned to. These transforms add
that irregularity to training references so that "irregular" becomes an
ordinary style dimension the encoder must look *past*, rather than the single
direction every real hand collapses onto.

The amplitudes are drawn once per sample and shared by that sample's references
and target -- how irregular a hand is, is part of its style -- while the
specific distortion is drawn independently for each glyph, because writing the
same letter twice is what a real hand cannot do identically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class JitterParams:
    """How irregular one synthetic 'hand' is. Drawn once per sample."""

    #: Amplitude of the elastic displacement field, in pixels.
    elastic: float = 0.0
    #: Peak-to-peak stroke weight variation along a stroke, as a fraction.
    weight: float = 0.0
    #: Baseline wobble, in pixels.
    shift: float = 0.0
    #: Lean, in degrees.
    rotate: float = 0.0

    @classmethod
    def sample(cls, rng: random.Random, strength: float) -> "JitterParams":
        """Draw one hand's irregularity. ``strength`` 0 disables everything."""
        if strength <= 0:
            return cls()
        return cls(
            elastic=strength * rng.uniform(0.0, 1.6),
            weight=strength * rng.uniform(0.0, 0.5),
            shift=strength * rng.uniform(0.0, 1.5),
            rotate=strength * rng.uniform(-2.5, 2.5),
        )

    @property
    def active(self) -> bool:
        return any((self.elastic, self.weight, self.shift, self.rotate))


def jitter_glyph(image: np.ndarray, params: JitterParams, rng: np.random.Generator) -> np.ndarray:
    """Draw this glyph once more, by the same unsteady hand.

    ``params`` fixes the hand; ``rng`` supplies this particular drawing of it.
    """
    if not params.active:
        return image

    from scipy.ndimage import gaussian_filter, map_coordinates

    h, w = image.shape
    out = image.astype(np.float32)

    # Pen pressure: a smooth field that thickens and thins the stroke along its
    # length. Applied before the geometry so the weight follows the letterform.
    if params.weight > 0:
        field = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 6.0)
        field /= max(float(np.abs(field).max()), 1e-6)
        out = np.clip(out * (1.0 + params.weight * field), 0.0, 1.0)

    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    dx = np.zeros((h, w), dtype=np.float32)

    # Elastic warp: the same letter, shaped differently this time.
    if params.elastic > 0:
        for target in (dy, dx):
            noise = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 8.0)
            scale = max(float(np.abs(noise).max()), 1e-6)
            target += params.elastic * noise / scale

    # Lean and baseline wobble, about the canvas centre.
    if params.rotate or params.shift:
        theta = np.radians(params.rotate)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry, rx = rows - cy, cols - cx
        dy += (ry * np.cos(theta) + rx * np.sin(theta) - ry) + rng.normal(0.0, params.shift)
        dx += (-ry * np.sin(theta) + rx * np.cos(theta) - rx) + rng.normal(0.0, params.shift)

    warped = map_coordinates(out, [rows + dy, cols + dx], order=1, mode="constant", cval=0.0)
    return np.clip(warped, 0.0, 1.0).astype(np.float32)
