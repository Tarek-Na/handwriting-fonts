"""Regression tests.

Most of these exist because the bug they describe actually happened during
development and was not obvious from looking at the output. Several were
invisible glyph-by-glyph and only showed up as a few points of aggregate
fidelity, which is exactly the kind of defect that survives into a shipped font.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.charset import (  # noqa: E402
    LATIN_CORE,
    SEED_24,
    Form,
    GlyphSpec,
    seed_charset,
)
from hfont.data.raster import Contour, fill_contours  # noqa: E402
from hfont.vector.trace import TraceConfig, trace_image  # noqa: E402


# --------------------------------------------------------------------------- #
# charset
# --------------------------------------------------------------------------- #

def test_glyph_specs_sort_when_forms_collide():
    """Sorting must work when two glyphs share a character.

    Latin never exercises this — one form per character means tuple comparison
    short-circuits on the character and never reaches the form. Arabic has up to
    four forms per letter, so an unordered Form enum would break every sorted()
    in the pipeline the moment Phase 2 starts.
    """
    specs = [
        GlyphSpec("a", Form.MEDI),
        GlyphSpec("a", Form.ISOL),
        GlyphSpec("b", Form.INIT),
        GlyphSpec("a", Form.FINA),
    ]
    assert [s.name for s in sorted(specs)] == ["a", "a.medi", "a.fina", "b.init"]
    assert Form.ISOL < Form.INIT < Form.MEDI < Form.FINA


def test_seed_set_is_within_budget_and_charset():
    assert 20 <= len(SEED_24) <= 30
    subset = seed_charset(LATIN_CORE, "seed24")
    assert len(subset) == len(set(SEED_24))
    assert set(subset.chars) <= set(LATIN_CORE.chars)


def test_glyph_names_follow_adobe_conventions():
    assert GlyphSpec(",").name == "comma"
    assert GlyphSpec("A").name == "A"
    assert GlyphSpec("0").name == "zero"


# --------------------------------------------------------------------------- #
# rasterizer
# --------------------------------------------------------------------------- #

def test_nonzero_winding_leaves_holes_empty():
    outer = Contour([(2, 2), (30, 2), (30, 30), (2, 30)])
    inner = Contour([(10, 10), (10, 22), (22, 22), (22, 10)])
    img = fill_contours([outer, inner], 32, 32, supersample=4)
    assert img[5, 5] == pytest.approx(1.0)
    assert img[16, 16] == pytest.approx(0.0)
    # Exact area: 28*28 outer minus 12*12 hole.
    assert float(img.sum()) == pytest.approx(28 * 28 - 12 * 12, abs=0.5)


def test_rasterizer_is_deterministic():
    """Training data must be bit-identical wherever it is generated."""
    square = Contour([(4.3, 4.7), (20.1, 4.7), (20.1, 19.2), (4.3, 19.2)])
    first = fill_contours([square], 32, 32, supersample=4)
    second = fill_contours([square], 32, 32, supersample=4)
    assert np.array_equal(first, second)


def test_horizontal_coverage_is_continuous():
    """Sub-pixel x positions must produce different coverage, not quantized steps."""
    values = []
    for offset in (0.0, 0.25, 0.5, 0.75):
        c = Contour([(4 + offset, 4), (12, 4), (12, 12), (4 + offset, 12)])
        values.append(float(fill_contours([c], 16, 16, supersample=4).sum()))
    assert len(set(np.round(values, 3))) == 4, "x coverage is quantized"


# --------------------------------------------------------------------------- #
# tracer
# --------------------------------------------------------------------------- #

def _rasterize_paths(paths, size, steps=24):
    contours = []
    for path in paths:
        pts = [path.start]
        cur = np.asarray(path.start, dtype=np.float64)
        for c1, c2, end in path.segments:
            c1, c2, end = (np.asarray(p, dtype=np.float64) for p in (c1, c2, end))
            t = np.linspace(0, 1, steps + 1)[1:]
            mt = 1 - t
            xy = (
                (mt**3)[:, None] * cur
                + (3 * mt**2 * t)[:, None] * c1
                + (3 * mt * t**2)[:, None] * c2
                + (t**3)[:, None] * end
            )
            pts.extend(map(tuple, xy))
            cur = end
        contours.append(Contour([tuple(map(float, p)) for p in pts]))
    return fill_contours(contours, size, size, supersample=4)


def test_traced_rectangle_keeps_its_position():
    """Guards the pixel-centre convention mismatch.

    skimage's find_contours indexes pixel centres at integers; the rasterizer
    treats pixel i as covering [i, i+1]. Without the half-pixel correction every
    outline lands up and to the left of its ink — imperceptible per glyph, about
    15 points of round-trip IoU across a font.
    """
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:40, 12:50] = 1.0
    back = _rasterize_paths(trace_image(img), 64)

    ink = np.argwhere(back > 0.5)
    assert ink.size, "traced rectangle vanished"
    assert ink[:, 0].min() == pytest.approx(20, abs=1)
    assert ink[:, 0].max() == pytest.approx(39, abs=1)
    assert ink[:, 1].min() == pytest.approx(12, abs=1)
    assert ink[:, 1].max() == pytest.approx(49, abs=1)


def test_straight_edges_do_not_bulge():
    """Guards the curve-fitting overshoot.

    A long flat run whose endpoint tangent is a few degrees off gets a control
    point placed most of a chord away along that wrong direction, bowing the
    outline far outside the shape. On a real E it reached 14px past the glyph.
    """
    img = np.zeros((64, 64), dtype=np.float32)
    img[28:34, 6:58] = 1.0  # a long, thin horizontal bar
    paths = trace_image(img)

    ys = [p[1] for path in paths for p in path.points]
    ctrl = [c[1] for path in paths for seg in path.segments for c in seg[:2]]
    assert min(ys + ctrl) > 24.0, "outline bulges above the bar"
    assert max(ys + ctrl) < 38.0, "outline bulges below the bar"


def test_tracer_terminates_on_noise():
    """An undertrained model emits speckle; export must still finish.

    Without a contour cap this did not terminate in any practical time, because
    every speck is a closed contour to fit.
    """
    rng = np.random.default_rng(0)
    noise = rng.random((64, 64)).astype(np.float32)
    paths = trace_image(noise, TraceConfig(max_contours=8))
    assert len(paths) <= 8


def test_rectangle_traces_to_few_segments():
    """A rectangle is four lines; fitting it to dozens of curves is a bug."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:44, 14:50] = 1.0
    total = sum(len(p.segments) for p in trace_image(img))
    assert total <= 8, f"rectangle produced {total} segments"


# --------------------------------------------------------------------------- #
# framing: corpus renderer and handwriting intake must agree
# --------------------------------------------------------------------------- #

HANDWRITING_FONT = Path("C:/Windows/Fonts/Inkfree.ttf")


@pytest.mark.skipif(not HANDWRITING_FONT.is_file(), reason="needs a handwriting-style font")
def test_intake_reproduces_corpus_framing():
    """Intake applied to the renderer's own output must give the same images back.

    The corpus renderer and the photo intake used to frame glyphs by different
    rules — designed baseline and full-charset extent on one side, estimated
    baseline and seed-letter extent on the other. Typeset fonts hid it; a
    handwriting font came through intake 17% larger than the model had been
    trained to expect. Both now go through data/frame.py, so round-tripping a
    font's seed letters through intake is an identity.
    """
    from hfont.data.render import FontRenderer, RenderConfig
    from hfont.evaluate.metrics import tolerant_f1
    from hfont.intake import IntakeConfig, normalize_samples

    corpus, _ = FontRenderer(HANDWRITING_FONT, RenderConfig()).render(LATIN_CORE)
    seed = list(seed_charset(LATIN_CORE, "seed24"))
    # Present the rendered glyphs as dark ink on light paper, as a scan would be.
    raw = {g: (255 * (1.0 - corpus[g.key].image)).astype(np.uint8) for g in seed}
    recovered = normalize_samples(raw, IntakeConfig()).images

    scores = [tolerant_f1(recovered[g.key], corpus[g.key].image) for g in seed]
    assert min(scores) > 0.97, f"framing drift: worst tolF1 {min(scores):.3f}"


@pytest.mark.skipif(not HANDWRITING_FONT.is_file(), reason="needs a handwriting-style font")
def test_frame_does_not_depend_on_the_seed_set():
    """A bigger seed set must not change the scale glyphs arrive at.

    Intake framed on every sample it was given, while the renderer frames on
    ``SEED_24`` alone. Moving from ``seed24`` to ``seed30`` therefore added `f`
    and `j` — the tallest ascender and the deepest descender — to the frame and
    silently shrank every letter to fit them. Real handwriting, which has more
    extreme proportions than any typeset font, lost 40% of its size that way,
    putting it well outside anything the model was trained on.
    """
    from hfont.data.render import FontRenderer, RenderConfig
    from hfont.intake import IntakeConfig, normalize_samples

    corpus, _ = FontRenderer(HANDWRITING_FONT, RenderConfig()).render(LATIN_CORE)

    def scale_for(seed: str) -> float:
        raw = {
            g: (255 * (1.0 - corpus[g.key].image)).astype(np.uint8)
            for g in seed_charset(LATIN_CORE, seed)
        }
        return normalize_samples(raw, IntakeConfig()).scale

    assert scale_for("seed30") == pytest.approx(scale_for("seed24"), rel=1e-6)


def test_freehand_intake_finds_every_letter():
    """Letters written on blank paper, with no grid, must still come back.

    Rows are clustered, the writing line is fitted per row and dots are
    attached to their stems. The dot of an `i` sits nearer the centre of the
    row above than its own, so clustering on raw vertical position put it on
    the previous line — leaving a dotless `i` and a spurious dot under some
    letter overhead.
    """
    from hfont.charset import LATIN_SEED_SETS
    from hfont.intake import FreehandConfig, freehand_cells

    rows = (10, 10, 10)
    chars = LATIN_SEED_SETS["seed30"]
    page = np.full((260, 620, 3), 240, dtype=np.uint8)
    index = 0
    for r, count in enumerate(rows):
        baseline = 70 + 70 * r
        for c in range(count):
            x = 24 + 58 * c
            char = chars[index]
            index += 1
            page[baseline - 24 : baseline, x : x + 6] = 30         # an n-shaped
            page[baseline - 24 : baseline, x + 10 : x + 16] = 30   # letter, drawn
            page[baseline - 24 : baseline - 18, x : x + 16] = 30   # in pen strokes
            if char in "ij":                                       # ... and its dot
                page[baseline - 38 : baseline - 32, x + 9 : x + 15] = 30

    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars

    for char in "ij":
        spec = next(g for g in found if g.char == char)
        ink = found[spec] < 0.5
        gaps = np.diff(np.nonzero(ink.any(axis=1))[0])
        assert (gaps > 1).any(), f"{char} lost its dot: it must arrive with its stem"


def test_detached_top_bar_stays_with_its_letter():
    """A stroke written clear of its letter must not be thrown away.

    Dots were restricted to `i` and `j` so that the leftovers of a printed
    ruling could not be attached to letters as dots. That rule also discarded
    the top bar of an `E` written without joining the stem: intake reported
    "dropped a 365 px mark that belongs to no letter", and the writer's own
    `E` was exported missing its bar, since seed letters are passed through
    unchanged. A stroke overlapping exactly one letter's column belongs to it.
    """
    from hfont.charset import LATIN_SEED_SETS
    from hfont.intake import FreehandConfig, freehand_cells

    rows = (10, 10, 10)
    chars = LATIN_SEED_SETS["seed30"]
    page = np.full((260, 620, 3), 240, dtype=np.uint8)
    index = 0
    for r, count in enumerate(rows):
        baseline = 70 + 70 * r
        for c in range(count):
            x = 24 + 58 * c
            char = chars[index]
            index += 1
            page[baseline - 24 : baseline, x : x + 6] = 30
            page[baseline - 24 : baseline, x + 10 : x + 16] = 30
            page[baseline - 24 : baseline - 18, x : x + 16] = 30
            if char == "E":  # a bar of its own, wider than the letter, clear of it
                page[baseline - 34 : baseline - 29, x - 2 : x + 18] = 30

    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars

    spec = next(g for g in found if g.char == "E")
    ink_rows = np.nonzero((found[spec] < 0.5).any(axis=1))[0]
    assert ink_rows.size, "E arrived with no ink at all"
    gaps = np.diff(ink_rows)
    assert (gaps > 1).any(), "E lost its detached top bar"


def test_symmetric_loss_does_not_reward_overinking():
    """Over- and under-inking by the same amount must cost about the same.

    The first loss weighted target ink only, so extra ink was 4x cheaper than
    missing ink and the model learned to draw everything ~1.5x too heavy.
    """
    import torch

    from hfont.models.losses import weighted_l1

    target = -torch.ones(1, 1, 32, 32)
    target[..., 12:20, 8:24] = 1.0          # an 8px-tall bar
    thick = -torch.ones_like(target)
    thick[..., 11:21, 8:24] = 1.0           # 1px too heavy on each side
    thin = -torch.ones_like(target)
    thin[..., 13:19, 8:24] = 1.0            # 1px too light on each side

    over = float(weighted_l1(thick, target, ink_weight=4.0))
    under = float(weighted_l1(thin, target, ink_weight=4.0))
    assert abs(over - under) / max(over, under) < 0.05, (over, under)


# --------------------------------------------------------------------------- #
# font assembly
# --------------------------------------------------------------------------- #

def _tiny_font(tmp_path: Path):
    from hfont.fontbuild.build import BuildConfig, FontMetadata, rasters_to_font, save_font

    rasters, advances = {}, {}
    for spec in LATIN_CORE:
        img = np.zeros((128, 128), dtype=np.float32)
        img[40:96, 50:80] = 1.0  # a plain bar stands in for every letter
        rasters[spec.key] = img
        advances[spec.key] = 0.45

    fb = rasters_to_font(rasters, advances, LATIN_CORE, FontMetadata(family="Test"), BuildConfig())
    return save_font(fb, tmp_path / "test.otf")


def test_exported_font_has_a_space_glyph(tmp_path):
    """Space has no outline, so nothing generates it - the builder must.

    Its absence passes every pixel metric and every font viewer, and makes the
    font useless for text: words run together.
    """
    from fontTools.ttLib import TTFont

    path = _tiny_font(tmp_path)
    tt = TTFont(str(path))
    cmap = tt.getBestCmap()
    assert 0x20 in cmap, "no space in cmap"
    assert tt["hmtx"][cmap[0x20]][0] > 0, "space has zero advance"
    tt.close()


def test_exported_font_shapes_real_text(tmp_path):
    from hfont.evaluate.shaping import check_font

    report = check_font(_tiny_font(tmp_path), charset=LATIN_CORE)
    assert report.ok, report.summary()


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_local"
needs_cache = pytest.mark.skipif(
    not (CACHE / "meta.json").is_file(),
    reason="no local glyph cache; run `hfont index` and `hfont prepare` first",
)


@needs_cache
def test_target_glyph_never_appears_in_style_references():
    """The central correctness property of the sampler.

    If the target can appear among the references the task degenerates into
    copying, training loss collapses, and the model is worthless at inference —
    where the user has by definition not written the letter being generated.
    Excluding by character name alone is not enough: fonts point several
    characters at one outline, so the check is on rendered pixels.
    """
    from hfont.data.dataset import DatasetConfig, GlyphPairDataset, GlyphStore

    ds = GlyphPairDataset(GlyphStore(CACHE), DatasetConfig(split="train"))
    for i in range(200):
        sample = ds[i]
        for j in range(int(sample["ref_mask"].sum())):
            assert not np.array_equal(
                sample["refs"][j].numpy(), sample["target"].numpy()
            ), f"sample {i} leaked the target into reference {j}"


@needs_cache
def test_split_is_by_family_not_by_file():
    """Two weights of one family in different splits would leak style."""
    from hfont.data.dataset import GlyphStore

    store = GlyphStore(CACHE)
    splits: dict[str, set[str]] = {}
    for record in store.records:
        splits.setdefault(record.family, set()).add(record.split)
    offenders = {f: s for f, s in splits.items() if len(s) > 1}
    assert not offenders, f"families split across sets: {list(offenders)[:5]}"


@needs_cache
def test_eval_batches_collate():
    """Fonts missing a seed glyph must still batch together."""
    import torch
    from torch.utils.data import DataLoader

    from hfont.data.dataset import FontEvalDataset, GlyphStore

    ds = FontEvalDataset(GlyphStore(CACHE), split="val", max_fonts=8)
    batch = next(iter(DataLoader(ds, batch_size=16)))
    assert batch["refs"].ndim == 5
    assert batch["ref_mask"].shape[0] == batch["refs"].shape[0]
    assert torch.is_tensor(batch["target"])
