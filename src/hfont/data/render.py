"""Render glyphs from a font file into normalized training images.

Normalization is the part worth reading carefully, because it is what makes the
corpus and the inference-time input comparable.

Glyphs are *not* normalized per glyph. Each glyph is placed using a single
scale and baseline computed once for the whole font, from the ink bounding box
of the glyphs being rendered. Within a font, relative proportions — x-height
against cap height, ascender length, stroke weight — are preserved exactly,
because those *are* the style. Across fonts, overall size is equalized.

The reason to key the scale off the ink box rather than off the em square is
that at inference time the input is a photograph of someone's handwriting,
which has no em square. An ink box over a known set of letters is the one
measurement available in both cases, so both go through the same code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont, TTLibError

from ..charset import SEED_24, Charset, GlyphSpec
from .frame import canvas_scale, frame_from_boxes
from .raster import FlatteningPen, fill_contours

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderConfig:
    """Geometry of the rendered glyph canvas.

    The baseline sits at a *fixed* canvas row for every font, and the per-font
    scale is whatever makes that font's ascenders and descenders both fit.

    Deriving the baseline per font instead (place the ink box, let the baseline
    fall where it may) renders equally well and is wrong for this project: a
    generated glyph has no source font to ask, so at export time there would be
    no way to know which row of the canvas is the baseline — and a font whose
    baseline is off by a few percent has every letter floating above or sinking
    below the line. Fixing it here makes it a known constant everywhere
    downstream, and costs only some unused canvas on fonts with short
    extenders.
    """

    size: int = 128
    #: Baseline row as a fraction of canvas height from the top.
    baseline: float = 0.75
    #: Fraction of the canvas height left clear at the top and bottom.
    margin: float = 0.06
    #: Rasterizer supersampling factor. 4 gives 16 gray levels, 8 gives 64.
    supersample: int = 4
    #: Refuse to scale a font up by more than this, so a font whose sampled
    #: glyphs are all tiny (rare, but it happens with broken subsets) does not
    #: get blown up into a blurry mess.
    max_upscale: float = 4.0
    #: Assumed ascender height as a fraction of the em. With the defaults above
    #: this makes the canvas span ~1.02 em, which is typical for Latin, and it
    #: is what fixes the pixels-per-em constant used when exporting.
    ascender_em: float = 0.8
    #: The letters that define a font's frame: its baseline, and the ascent and
    #: descent the canvas must hold. These are the letters a user writes, so a
    #: typeset font is framed from exactly the evidence a photograph provides.
    #: See data/frame.py for why that matters.
    frame_chars: str = SEED_24
    #: "ink": frame from the reference letters' ink, matching intake.
    #: "font": the font's designed baseline and full-charset extent (legacy).
    frame_mode: str = "ink"

    @property
    def baseline_px(self) -> float:
        return self.size * self.baseline

    @property
    def margin_px(self) -> float:
        return self.size * self.margin

    @property
    def pixels_per_em(self) -> float:
        """Canvas pixels per em. Constant, by construction of the geometry."""
        return (self.baseline_px - self.margin_px) / self.ascender_em


@dataclass
class GlyphRaster:
    """One rendered glyph plus the metrics needed to rebuild a font from it."""

    spec: GlyphSpec
    image: np.ndarray  # (size, size) float32 in [0, 1], 1 = ink
    #: Advance width as a fraction of the canvas width.
    advance: float
    #: Ink bounding box in pixels (x0, y0, x1, y1), or None if the glyph is blank.
    ink_bbox: tuple[float, float, float, float] | None

    @property
    def is_blank(self) -> bool:
        return self.ink_bbox is None


@dataclass
class FontFrame:
    """The per-font normalization transform, in font units -> pixels."""

    scale: float
    #: Canvas row the baseline is drawn on.
    baseline_y: float
    units_per_em: int
    #: Where the baseline is in the font's own units (y up). Zero for the
    #: designed baseline; the ink-estimated baseline of a handwriting font is
    #: usually a little off it.
    baseline_units: float = 0.0

    def to_font_units(self, px: float) -> float:
        return px / self.scale


class FontRenderError(RuntimeError):
    pass


class FontRenderer:
    """Renders a charset from one font file into normalized rasters."""

    def __init__(self, path: str | Path, config: RenderConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config or RenderConfig()
        try:
            self._font = TTFont(str(self.path), fontNumber=0, lazy=True)
            self._glyph_set = self._font.getGlyphSet()
            self._cmap = self._font.getBestCmap()
        except (TTLibError, KeyError, AssertionError, OSError, Exception) as exc:
            raise FontRenderError(f"cannot open {self.path.name}: {exc}") from exc

        try:
            self.units_per_em = int(self._font["head"].unitsPerEm)
        except KeyError as exc:
            raise FontRenderError(f"{self.path.name} has no head table") from exc
        if self.units_per_em <= 0:
            raise FontRenderError(f"{self.path.name} has bad unitsPerEm")

    # -- glyph lookup -------------------------------------------------------
    def glyph_name(self, spec: GlyphSpec) -> str | None:
        """Font-internal glyph name for a spec, or None if the font lacks it."""
        return self._cmap.get(spec.codepoint)

    def available(self, charset: Charset) -> list[GlyphSpec]:
        """Subset of ``charset`` this font can actually draw."""
        return [g for g in charset if self.glyph_name(g) is not None]

    def coverage(self, charset: Charset) -> float:
        return len(self.available(charset)) / max(len(charset), 1)

    # -- measurement --------------------------------------------------------
    def _bounds(self, name: str) -> tuple[float, float, float, float] | None:
        pen = BoundsPen(self._glyph_set)
        try:
            self._glyph_set[name].draw(pen)
        except Exception:  # malformed outline, missing component, recursion
            return None
        return pen.bounds

    def measure(self, specs: list[GlyphSpec]) -> FontFrame:
        """Compute the shared scale and baseline for a set of glyphs."""
        if self.config.frame_mode == "ink":
            return self._measure_ink(specs)
        return self._measure_font(specs)

    def _measure_ink(self, specs: list[GlyphSpec]) -> FontFrame:
        """Frame from the reference letters' ink, exactly as intake does.

        Bounds come from the outlines of ``frame_chars`` only, negated into the
        y-down convention frame.py shares with image rows. The baseline is the
        ink estimate, not the designed y = 0, and it is recorded in font units
        so render_glyph can place it on the canvas baseline row.
        """
        cfg = self.config
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for char in dict.fromkeys(cfg.frame_chars):
            name = self._cmap.get(ord(char))
            bounds = self._bounds(name) if name else None
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                boxes[char] = (x0, -y1, x1, -y0)
        if not boxes:
            raise FontRenderError(f"{self.path.name} draws none of the frame letters")

        frame = frame_from_boxes(boxes)
        scale = canvas_scale(frame, cfg.size, cfg.baseline, cfg.margin)

        # Keep the widest glyph inside the canvas, and refuse absurd upscaling
        # of near-empty fonts.
        max_advance = max(
            (float(getattr(self._glyph_set[n], "width", 0) or 0)
             for n in (self.glyph_name(s) for s in specs) if n),
            default=0.0,
        )
        if max_advance > 0:
            scale = min(scale, cfg.size * 0.96 / max_advance)
        scale = min(scale, cfg.size / self.units_per_em * cfg.max_upscale)

        return FontFrame(
            scale=scale,
            baseline_y=cfg.baseline_px,
            units_per_em=self.units_per_em,
            baseline_units=-frame.baseline,  # back to y-up font units
        )

    def _measure_font(self, specs: list[GlyphSpec]) -> FontFrame:
        """Legacy frame: designed baseline, full-charset extent."""
        cfg = self.config
        y_min, y_max = None, None
        max_advance = 0.0

        for spec in specs:
            name = self.glyph_name(spec)
            if name is None:
                continue
            bounds = self._bounds(name)
            if bounds is not None:
                _, lo, _, hi = bounds
                y_min = lo if y_min is None else min(y_min, lo)
                y_max = hi if y_max is None else max(y_max, hi)
            width = getattr(self._glyph_set[name], "width", 0) or 0
            max_advance = max(max_advance, float(width))

        if y_min is None or y_max is None or y_max <= y_min:
            raise FontRenderError(f"{self.path.name} draws no ink for the charset")

        # The baseline is fixed, so the scale is set by whichever of the
        # ascender or the descender runs out of canvas first.
        above_px = cfg.baseline_px - cfg.margin_px
        below_px = cfg.size - cfg.margin_px - cfg.baseline_px
        limits = []
        if y_max > 0:
            limits.append(above_px / y_max)
        if y_min < 0:
            limits.append(below_px / -y_min)
        if not limits:
            raise FontRenderError(f"{self.path.name} has degenerate vertical extent")
        scale = min(limits)

        # Keep the widest glyph inside the canvas even if that means the ink no
        # longer reaches the margins.
        if max_advance > 0:
            scale = min(scale, cfg.size * 0.96 / max_advance)

        # Guard against absurd upscaling of near-empty fonts.
        em_scale = cfg.size / self.units_per_em
        scale = min(scale, em_scale * cfg.max_upscale)

        return FontFrame(
            scale=scale, baseline_y=cfg.baseline_px, units_per_em=self.units_per_em
        )

    # -- rendering ----------------------------------------------------------
    def render_glyph(self, spec: GlyphSpec, frame: FontFrame) -> GlyphRaster | None:
        cfg = self.config
        name = self.glyph_name(spec)
        if name is None:
            return None

        glyph = self._glyph_set[name]
        advance_units = float(getattr(glyph, "width", 0) or 0)
        advance_px = advance_units * frame.scale

        # Horizontal placement. In ink mode the ink box is centred, because that
        # is all intake can do with a handwritten letter; the font builder then
        # centres the ink inside the predicted advance, so the two agree. Legacy
        # mode centres the advance box and keeps the font's own side bearings.
        bounds = self._bounds(name) if cfg.frame_mode == "ink" else None
        if bounds is not None:
            ink_center = (bounds[0] + bounds[2]) / 2.0
            offset_x = cfg.size / 2.0 - ink_center * frame.scale
        else:
            offset_x = (cfg.size - advance_px) / 2.0

        pen = FlatteningPen(
            self._glyph_set,
            scale=frame.scale,
            offset_x=offset_x,
            # Puts the frame's baseline (font units, y up) on the canvas row.
            offset_y=frame.baseline_y + frame.baseline_units * frame.scale,
        )
        try:
            glyph.draw(pen)
        except Exception as exc:
            log.debug("%s: glyph %s failed to draw: %s", self.path.name, name, exc)
            return None

        image = fill_contours(pen.contours, cfg.size, cfg.size, cfg.supersample)

        ink = np.argwhere(image > 0.02)
        if ink.size:
            bbox = (
                float(ink[:, 1].min()),
                float(ink[:, 0].min()),
                float(ink[:, 1].max() + 1),
                float(ink[:, 0].max() + 1),
            )
        else:
            bbox = None

        return GlyphRaster(
            spec=spec,
            image=image,
            advance=advance_px / cfg.size,
            ink_bbox=bbox,
        )

    def render(
        self, charset: Charset, frame: FontFrame | None = None
    ) -> tuple[dict[str, GlyphRaster], FontFrame]:
        """Render every glyph of ``charset`` this font supports.

        Returns the rasters keyed by :attr:`GlyphSpec.key`, and the frame used —
        pass the frame back in to render more glyphs in the same coordinate
        system.
        """
        specs = self.available(charset)
        if not specs:
            raise FontRenderError(f"{self.path.name} covers none of {charset.name}")
        if frame is None:
            frame = self.measure(specs)

        out: dict[str, GlyphRaster] = {}
        for spec in specs:
            raster = self.render_glyph(spec, frame)
            if raster is not None:
                out[spec.key] = raster
        return out, frame

    # -- identification -----------------------------------------------------
    def name_records(self) -> dict[int, str]:
        names: dict[int, str] = {}
        try:
            table = self._font["name"]
        except KeyError:
            return names
        for record in table.names:
            if record.nameID in names:
                continue
            try:
                names[record.nameID] = record.toUnicode()
            except UnicodeDecodeError:
                continue
        return names

    @property
    def family(self) -> str:
        return self.name_records().get(16) or self.name_records().get(1) or self.path.stem

    @property
    def style(self) -> str:
        return self.name_records().get(17) or self.name_records().get(2) or "Regular"

    def close(self) -> None:
        self._font.close()

    def __enter__(self) -> "FontRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
