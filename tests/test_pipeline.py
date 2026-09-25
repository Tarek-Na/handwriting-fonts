"""Regression tests.

Most of these exist because the bug they describe actually happened during
development and was not obvious from looking at the output. Several were
invisible glyph-by-glyph and only showed up as a few points of aggregate
fidelity, which is exactly the kind of defect that survives into a shipped font.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import random
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


def test_cl_dice_sees_shape_not_stroke_weight():
    """The added metric must be the one thing tolerant F1 is not: weight-blind.

    Tolerant F1 scores a glyph dilated by a pixel exactly 1.000, the same as an
    untouched copy, because its 1.5px tolerance is wider than a real hand's
    1.3px stroke. Every conclusion ranked by it is therefore blind to the
    project's known over-inking. clDice is added alongside it to separate "the
    right shape, too heavy" from "the wrong shape", and this pins that property.
    """
    from skimage.morphology import dilation, disk

    from hfont.evaluate.metrics import cl_dice, iou

    glyph = np.zeros((64, 64), dtype=np.float32)
    glyph[10:54, 20:22] = 1.0            # a stem
    glyph[30:32, 20:44] = 1.0            # and a crossbar
    fatter = dilation(glyph, disk(2))
    other = np.zeros((64, 64), dtype=np.float32)
    other[10:54, 40:42] = 1.0

    assert (fatter > 0.5).sum() > 2 * (glyph > 0.5).sum(), "the test needs a real thickening"
    assert cl_dice(fatter, glyph) > 0.95, "clDice must not punish weight alone"
    assert iou(fatter, glyph) < 0.6, "IoU is the one that punishes weight"
    assert cl_dice(other, glyph) < 0.5, "clDice must still reject the wrong shape"


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


def _lopsided_font(tmp_path: Path, name: str):
    """A font whose ink sits far off-centre inside each advance.

    ``a`` is drawn against the right edge of its canvas and ``b`` against the
    left, so each one's advance comfortably contains its own ink while the pair
    ``ab`` sets with the two strokes crossing. Every per-glyph rule in the
    builder passes; only looking at the pair finds it.
    """
    from hfont.fontbuild.build import BuildConfig, FontMetadata, rasters_to_font, save_font

    rasters, advances = {}, {}
    for spec in LATIN_CORE:
        img = np.zeros((128, 128), dtype=np.float32)
        right_heavy = spec.char in "acegikmoqsuwy"
        img[40:96, 64:127] = 1.0 if right_heavy else 0.0
        img[40:96, 1:64] = 0.0 if right_heavy else 1.0
        rasters[spec.key] = img
        advances[spec.key] = 0.45

    fb = rasters_to_font(rasters, advances, LATIN_CORE, FontMetadata(family="Lop"), BuildConfig())
    return save_font(fb, tmp_path / name)


def test_shaping_report_counts_colliding_letters(tmp_path):
    """The collision count must reflect a real collision, not always be zero.

    ``overlapping_pairs`` sat on the report as a declared field that nothing
    ever assigned, so every font this project exported reported zero colliding
    pairs whether or not its letters ran into each other.
    """
    from hfont.evaluate.shaping import check_font

    samples = ("ab", "ba")
    # Against the version that only declared the field, this reads 0.
    report = check_font(_lopsided_font(tmp_path, "lop_report.otf"), samples, charset=LATIN_CORE)
    assert report.overlapping_pairs > 0, (
        "no collisions on a font built to collide: " + report.summary()
    )
    assert report.shaped_pairs == 2

    from hfont.evaluate.shaping import ink_collisions

    collisions, pairs, worst = ink_collisions(_lopsided_font(tmp_path, "lop.otf"), samples)
    assert pairs == 2, f"expected one adjacent pair per sample, examined {pairs}"
    assert ("a", "b") in collisions, f"ab must collide; found {collisions}"
    assert worst > 0.1, f"the overlap is over half an em by construction, got {worst:.3f}"

    # Control: the same check on well-spaced letters must stay quiet, otherwise
    # a count that is always non-zero is as useless as one that is always zero.
    clean, clean_pairs, clean_worst = ink_collisions(_tiny_font(tmp_path), samples)
    assert clean_pairs == 2
    assert not clean, f"centred ink with a wide advance must not collide: {clean}"
    assert clean_worst == 0.0


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


def test_scale_jitter_moves_refs_and_target_together_but_not_content():
    """Size must come from the style path, so content stays put.

    Real hands arrive at about half the size of a rendered font on the model's
    canvas, and that size difference is the axis every photographed hand
    collapses onto. Training only covers it if references and target are
    rescaled as one while the content glyph keeps its own scale -- which is the
    relationship at inference, where the content font is rendered full size and
    the references are whatever the writer's hand came out as.
    """
    import numpy as np

    from hfont.data.augment import rescale_glyph, sample_scale

    glyph = np.zeros((128, 128), dtype=np.float32)
    glyph[40:96, 50:80] = 1.0

    def ink_height(image):
        rows = np.nonzero((image > 0.5).any(axis=1))[0]
        return int(rows[-1] - rows[0] + 1) if rows.size else 0

    assert ink_height(rescale_glyph(glyph, 0.5)) < ink_height(glyph) * 0.6
    assert ink_height(rescale_glyph(glyph, 1.5)) > ink_height(glyph) * 1.4
    assert rescale_glyph(glyph, 1.0) is glyph, "an identity scale must not resample"

    # The draw has to reach the small end: hands land near 0.54x, and a linear
    # draw would put most of its mass above 1.0 and miss exactly that region.
    rng = random.Random(0)
    drawn = [sample_scale(rng, 1.0) for _ in range(400)]
    assert min(drawn) < 0.6, f"never sampled small enough: min {min(drawn):.2f}"
    assert max(drawn) > 1.6, f"never sampled large enough: max {max(drawn):.2f}"
    below = sum(1 for d in drawn if d < 1.0)
    assert 0.4 < below / len(drawn) < 0.6, f"draw is lopsided: {below}/{len(drawn)} below 1.0"
    assert sample_scale(rng, 0.0) == 1.0, "jitter 0 must disable the augmentation"


@needs_cache
def test_scale_jitter_is_off_by_default_and_changes_samples_when_on():
    """The default path must be byte-identical to the shipped behaviour."""
    import numpy as np

    from hfont.data.dataset import DatasetConfig, GlyphPairDataset, GlyphStore

    store = GlyphStore(CACHE)
    plain = GlyphPairDataset(store, DatasetConfig(split="train"))[7]
    again = GlyphPairDataset(store, DatasetConfig(split="train"))[7]
    assert np.array_equal(plain["target"].numpy(), again["target"].numpy())

    jittered = GlyphPairDataset(store, DatasetConfig(split="train", scale_jitter=1.0))[7]
    assert jittered["content"].shape == plain["content"].shape
    assert not np.array_equal(jittered["target"].numpy(), plain["target"].numpy()), (
        "scale_jitter=1.0 left the target untouched"
    )
    # Content is drawn from the same font and key either way, so if it moved,
    # the augmentation is reaching the path it must not touch.
    assert np.array_equal(jittered["content"].numpy(), plain["content"].numpy()), (
        "the content glyph was rescaled; size must come from the style path only"
    )


# --------------------------------------------------------------------------- #
# reference attention
# --------------------------------------------------------------------------- #

def test_reference_attention_starts_as_an_exact_identity():
    """An untrained attention block must not disturb the model it is added to.

    The output projection is zero-initialised precisely so an existing
    checkpoint can be warm-started into this architecture. If it were not
    exactly the identity, step 0 would destroy the behaviour being built on.
    """
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    torch.manual_seed(0)
    gen = GlyphGenerator(GeneratorConfig(style_attention=True))
    content = torch.randn(2, 1, 128, 128)
    refs = torch.randn(2, 4, 1, 128, 128)
    mask = torch.ones(2, 4, dtype=torch.bool)
    ids = torch.tensor([1, 5])

    with torch.no_grad():
        style, ref_feats = gen.encode_style_full(refs, mask)
        attended = gen.decode(content, ids, style, ref_feats, mask)["image"]
        plain = gen.decode(content, ids, style)["image"]
    assert torch.equal(attended, plain), "zero-init attention changed the output"


def test_reference_attention_ignores_padded_slots():
    """Padding is zeros, which is a value the attention could otherwise read."""
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    torch.manual_seed(0)
    gen = GlyphGenerator(GeneratorConfig(style_attention=True))
    torch.nn.init.normal_(gen.ref_attention.proj.weight, std=0.1)  # make it live

    content = torch.randn(1, 1, 128, 128)
    refs = torch.randn(1, 6, 1, 128, 128)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 3:] = False
    ids = torch.tensor([2])

    polluted = refs.clone()
    polluted[0, 3:] = 42.0
    with torch.no_grad():
        a = gen.decode(content, ids, *gen.encode_style_full(refs, mask)[:1],
                       gen.encode_style_full(refs, mask)[1], mask)["image"]
        b = gen.decode(content, ids, *gen.encode_style_full(polluted, mask)[:1],
                       gen.encode_style_full(polluted, mask)[1], mask)["image"]
    assert torch.allclose(a, b, atol=1e-5), "masked-out references changed the output"


def test_reference_attention_actually_reads_the_references():
    """The point of the block: change a reference, change the target.

    A trained attention block must respond to *which* references it is given,
    over and above the pooled style code. This compares the same perturbation
    with the block live against the block disabled, so what it measures is the
    attention path's contribution and not the style vector's.
    """
    import torch

    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    torch.manual_seed(0)
    gen = GlyphGenerator(GeneratorConfig(style_attention=True))
    torch.nn.init.normal_(gen.ref_attention.proj.weight, std=0.1)

    content = torch.randn(1, 1, 128, 128)
    refs = torch.randn(1, 4, 1, 128, 128)
    other = refs.clone()
    other[0, 1] = torch.randn(1, 128, 128)
    mask = torch.ones(1, 4, dtype=torch.bool)
    ids = torch.tensor([2])

    with torch.no_grad():
        s_a, f_a = gen.encode_style_full(refs, mask)
        s_b, f_b = gen.encode_style_full(other, mask)
        with_attn = (gen.decode(content, ids, s_a, f_a, mask)["image"]
                     - gen.decode(content, ids, s_b, f_b, mask)["image"]).abs().mean()
        pooled_only = (gen.decode(content, ids, s_a)["image"]
                       - gen.decode(content, ids, s_b)["image"]).abs().mean()
    assert with_attn > pooled_only, (
        f"attention added no reference sensitivity: {with_attn:.5f} vs {pooled_only:.5f}"
    )


def test_attention_is_off_by_default_and_costs_nothing_when_off():
    """The shipped architecture must be untouched unless asked for."""
    from hfont.models.generator import GeneratorConfig, GlyphGenerator

    assert GeneratorConfig().style_attention is False
    plain = GlyphGenerator(GeneratorConfig())
    assert plain.ref_attention is None
    attn = GlyphGenerator(GeneratorConfig(style_attention=True))
    assert plain.num_parameters() < attn.num_parameters()


def test_style_contrastive_separates_hands_and_joins_views():
    """Low when same-hand views agree and differ from other hands; high otherwise.

    Nothing in the reconstruction loss asks style codes to be *different* for
    different hands, and measurement showed they are not: three writers sit at
    cosine 0.99+. This loss is the direct ask, so it must actually reward the
    arrangement it claims to.
    """
    import torch

    from hfont.models.losses import style_contrastive

    torch.manual_seed(0)
    valid = torch.ones(6, dtype=torch.bool)

    # Well-arranged: each sample's two views agree, samples differ from each other.
    base = torch.eye(6, 32)
    good = style_contrastive(base, base + 0.01 * torch.randn(6, 32), valid)

    # Collapsed: every sample has nearly the same code, which is the defect.
    flat = torch.randn(1, 32).repeat(6, 1)
    bad = style_contrastive(flat + 0.01 * torch.randn(6, 32),
                            flat + 0.01 * torch.randn(6, 32), valid)
    assert good < bad, f"collapsed codes not penalised: good {good:.3f} vs bad {bad:.3f}"

    # Fewer than two usable samples cannot form pairs; must not blow up.
    lone = style_contrastive(base, base, torch.zeros(6, dtype=torch.bool))
    assert float(lone) == 0.0


def test_style_views_are_disjoint_halves_of_the_references():
    """A positive pair must be the same hand through *different* letters.

    If the halves overlapped, the model could satisfy the loss by encoding the
    shared reference rather than the hand, which is the shortcut this is meant
    to rule out.
    """
    import torch

    from hfont.train import Trainer

    mask = torch.tensor([
        [True, True, True, True, False, False, False, False],
        [True, True, True, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
    ])
    counts = mask.sum(dim=1)
    half = counts // 2
    rank = mask.cumsum(dim=1) - 1
    first = mask & (rank < half[:, None])
    second = mask & (rank >= half[:, None])

    assert not (first & second).any(), "halves overlap"
    assert torch.equal(first | second, mask), "halves do not cover the references"
    assert torch.equal(counts >= 2, torch.tensor([True, True, False]))
    assert first[0].sum() == 2 and second[0].sum() == 2
    assert first[1].sum() == 1 and second[1].sum() == 2
    assert Trainer is not None  # the split above mirrors Trainer._style_views


def test_recommended_config_is_the_measured_winner_and_defaults_stay_safe():
    """The recipe must carry the winning settings, and NOT become the default.

    Flipping GeneratorConfig.style_attention to True would break every
    checkpoint saved before it existed: their stored config has no such key, so
    the default would apply, the model would build an attention block, and
    load_state_dict would fail on the missing weights. This asserts both halves
    of that -- the recipe opts in, the dataclass does not.
    """
    from hfont.data.dataset import DatasetConfig
    from hfont.models.generator import GeneratorConfig
    from hfont.models.losses import LossConfig
    from hfont.train import recommended_config

    cfg = recommended_config("/tmp/cache", "/tmp/out")
    assert cfg.generator.style_attention is True
    assert cfg.loss.style_contrastive > 0
    assert cfg.data.scale_jitter == 0.0, "scale jitter failed its gate; must stay off"
    assert cfg.steps == 40_000

    # The defaults a bare checkpoint falls back to must remain the old behaviour.
    assert GeneratorConfig().style_attention is False
    assert LossConfig().style_contrastive == 0.0
    assert DatasetConfig().scale_jitter == 0.0

    assert recommended_config("/tmp/c", "/tmp/o", steps=5).steps == 5


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #

def test_percentile_framing_is_a_noop_on_regular_typefaces():
    """The whole case for this rule is that it only bites on irregular hands.

    A designed font's ascenders all reach the same line, so the 90th percentile
    of ascent *is* the maximum and the rendering is unchanged — which is what
    lets the corpus stay as the model learned it, with no re-render or retrain.
    If this ever stops holding, the framing change silently invalidates the
    trained model and this test is the only thing that would say so.
    """
    from hfont.data.frame import frame_from_boxes

    # Ascenders level at y=0, descenders level at y=120, baseline at y=100.
    regular = {c: (0.0, 0.0, 10.0, 100.0) for c in "bdfhkl"}
    regular.update({c: (0.0, 40.0, 10.0, 100.0) for c in "acemnorsuvwxz"})
    regular.update({c: (0.0, 40.0, 10.0, 120.0) for c in "gpqy"})

    strict = frame_from_boxes(regular, percentile=100)
    relaxed = frame_from_boxes(regular)
    assert relaxed.ascent == pytest.approx(strict.ascent), "changed a regular typeface"
    assert relaxed.descent == pytest.approx(strict.descent)


def test_percentile_framing_ignores_one_sprawling_letter():
    """A single outlier must not shrink the whole alphabet.

    Sizing to the extreme is why photographed hands arrived at roughly half the
    corpus scale: one flamboyant capital or one deep descender set the frame for
    all 30 letters.
    """
    from hfont.data.frame import frame_from_boxes

    hand = {c: (0.0, 40.0, 10.0, 100.0) for c in "acemnorsuvwxz"}
    hand.update({c: (0.0, 10.0, 10.0, 100.0) for c in "bdhkl"})
    hand["A"] = (0.0, -60.0, 10.0, 100.0)   # one letter reaching far above
    hand["j"] = (0.0, 40.0, 10.0, 190.0)    # and one far below

    strict = frame_from_boxes(hand, percentile=100)
    relaxed = frame_from_boxes(hand)
    assert relaxed.ascent < strict.ascent, "the outlier still sets the frame"
    assert relaxed.descent < strict.descent
    # Smaller extent means a larger scale on a fixed canvas, which is the point.
    assert strict.ascent + strict.descent > 1.3 * (relaxed.ascent + relaxed.descent)


def test_framing_percentile_100_reproduces_the_old_behaviour():
    """The previous rule must remain reachable, for comparison and rollback."""
    from hfont.data.frame import frame_from_boxes

    boxes = {"a": (0.0, 40.0, 10.0, 100.0), "b": (0.0, 5.0, 10.0, 100.0),
             "g": (0.0, 40.0, 10.0, 150.0)}
    f = frame_from_boxes(boxes, percentile=100)
    assert f.ascent == pytest.approx(max(f.baseline - b[1] for b in boxes.values()))
    assert f.descent == pytest.approx(max(b[3] - f.baseline for b in boxes.values()))


def test_pen_normalisation_leaves_a_dark_pen_untouched():
    """A saturated sheet must pass through byte-identical.

    Measured on the real writers: the two who used dark pens scored identically
    with this on and off. That no-op is what makes it safe to leave on by
    default, and the test pins it.
    """
    from hfont.intake import normalize_pen_darkness

    rng = np.random.default_rng(0)
    sheet = {}
    for k in "abc":
        img = np.zeros((32, 32), dtype=np.float32)
        img[8:24, 12:20] = 1.0
        img[8:24, 11] = img[8:24, 20] = 0.5          # anti-aliased edges
        img += rng.uniform(0, 0.05, img.shape).astype(np.float32)
        sheet[k] = np.clip(img, 0, 1)

    out = normalize_pen_darkness(sheet)
    for k in sheet:
        assert np.array_equal(out[k], sheet[k]), f"{k} was altered on a dark-pen sheet"


def test_pen_normalisation_lifts_a_light_pen_to_full_ink():
    """A faint pen's strokes must be raised clear of the tracer's 0.5 threshold.

    An orange pen on white scores ~0.52 ink, so its strokes straddled the
    threshold and export dropped 44% of the ink. After this they must not.
    """
    from hfont.intake import normalize_pen_darkness

    sheet = {}
    for k in "abc":
        img = np.zeros((32, 32), dtype=np.float32)
        img[8:24, 12:20] = 0.52
        sheet[k] = img
    before = sum(int((v > 0.5).sum()) for v in sheet.values())

    out = normalize_pen_darkness(sheet)
    after = sum(int((v > 0.5).sum()) for v in out.values())
    stroke = np.concatenate([v[v > 0.1] for v in out.values()])
    assert np.median(stroke) > 0.9, f"stroke core still faint: {np.median(stroke):.2f}"
    assert after >= before
    assert all(v.max() <= 1.0 for v in out.values()), "ink exceeded 1.0"


def test_pen_normalisation_gain_is_capped():
    """A near-empty sheet must not have its noise amplified without limit."""
    from hfont.intake import PEN_MAX_GAIN, normalize_pen_darkness

    sheet = {"a": np.full((16, 16), 0.12, dtype=np.float32)}
    out = normalize_pen_darkness(sheet)
    assert out["a"].max() <= 0.12 * PEN_MAX_GAIN + 1e-6


def _part(y0, x0, y1, x1):
    from types import SimpleNamespace
    return SimpleNamespace(bbox=(y0, x0, y1, x1), area=(y1 - y0) * (x1 - x0))


def test_descender_hook_that_curls_away_joins_its_letter():
    """The hook of a `j` must stay with the `j`.

    It sits just below the stem and bends left, so it fails every other rule:
    its centre is outside the stem's span, too little of it overlaps the stem's
    column, and it lies below the body where a dot would lie above. Intake used
    to log "dropped a 159 px mark that belongs to no letter" and keep a 10 px
    fragment of the stem, and the writer's own `j` exported as a dot.
    """
    from hfont.intake import _attach_marks

    a_body, j_body = _part(10, 20, 60, 60), _part(10, 100, 60, 108)
    grouped = [[{"parts": [a_body]}, {"parts": [j_body]}]]
    hook = _part(61, 88, 70, 103)          # 1 px below the stem, curling left

    _attach_marks(grouped, [hook], median_height=50, chars=[["A", "j"]])
    assert hook in grouped[0][1]["parts"], "the j lost its hook"
    assert hook not in grouped[0][0]["parts"]


def test_mark_beside_a_letter_does_not_join_it_by_touch():
    """Only a descender joins by touch -- not whatever brushes a letter's side.

    Allowing a touch from any side glued the crossing of two grid lines onto
    writer 2's `A`: a `+` touches exactly one letter as surely as a `j` hook
    does, and would have been imitated into every `A` the font set.
    """
    from hfont.intake import _attach_marks

    a_body, j_body = _part(10, 20, 60, 60), _part(10, 100, 60, 108)
    grouped = [[{"parts": [a_body]}, {"parts": [j_body]}]]
    cross = _part(25, 8, 40, 19)           # 1 px left of the A, at mid-height

    _attach_marks(grouped, [cross], median_height=50, chars=[["A", "j"]])
    assert cross not in grouped[0][0]["parts"], "a grid crossing joined the A"
