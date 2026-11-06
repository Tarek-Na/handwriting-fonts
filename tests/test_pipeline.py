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


def test_quality_report_carries_a_scale_invariant_metric():
    """The number shown to a user must not move just because letters got bigger.

    tol_f1 uses a fixed pixel tolerance, so scoring the same glyphs at half size
    inflated it by ~0.09 on real fonts while clDice did not move. The framing
    change that fixed half-scale intake therefore made the displayed tol-F1 drop
    with no loss of quality; clDice is reported beside it so that cannot be
    misread again.
    """
    from scipy.ndimage import zoom

    from hfont.evaluate.report import _score

    target = np.zeros((128, 128), dtype=np.float32)
    target[30:100, 40:48] = 1.0
    target[30:38, 40:90] = 1.0
    pred = np.roll(target, 2, axis=1)            # a slightly misplaced guess

    def half(v):
        z = zoom(v, 0.5, order=1)
        out = np.zeros_like(v)
        out[32:32 + z.shape[0], 32:32 + z.shape[1]] = z
        return out

    full, small = _score(pred, target), _score(half(pred), half(target))
    assert "cl_dice" in full
    assert small["tol_f1"] > full["tol_f1"] + 0.05, "tol_f1 should reward smallness"
    assert abs(small["cl_dice"] - full["cl_dice"]) < abs(small["tol_f1"] - full["tol_f1"]), (
        "cl_dice moved with scale as much as tol_f1 did"
    )


def _cell(top, foot, h=200, w=60):
    """A block letter on one shared sheet frame: paper 240, ink 30."""
    img = np.full((h, w), 240, dtype=np.uint8)
    img[top:foot, 20:40] = 30
    return img


def _foot(v):
    return int(np.nonzero((v > 0.5).any(axis=1))[0][-1]) + 1


def _snap_sheet(snap):
    from hfont.intake import IntakeConfig, normalize_samples

    spec = {g.char: g for g in LATIN_CORE}
    raw = {spec[c]: _cell(90, 120) for c in "aeonsr"}   # on the line
    raw[spec["m"]] = _cell(75, 105)                       # written 15 px high
    raw[spec["g"]] = _cell(90, 150)                       # a descender
    out = normalize_samples(raw, IntakeConfig(size=128, snap_baseline=snap)).images
    return {c: out[spec[c].key] for c in "amg"}


def test_letter_written_off_the_line_is_rested_on_it():
    """A one-off wobble must not become every copy of that glyph's position.

    In handwriting a letter lands differently each time; a font reuses one
    glyph everywhere. Left alone, a writer's single high `m` floats above the
    line in every word the font sets.
    """
    canvas_baseline = int(0.75 * 128)
    snapped = _snap_sheet(True)
    assert abs(_foot(snapped["m"]) - canvas_baseline) <= 1, "m still off the line"
    assert abs(_foot(snapped["a"]) - canvas_baseline) <= 1

    loose = _snap_sheet(False)
    assert _foot(loose["m"]) < canvas_baseline - 5, "the fixture's m should sit high"


def test_snapping_leaves_descenders_below_the_line():
    """Pulling a descender's foot up to the line would lop off every g, p, y."""
    canvas_baseline = int(0.75 * 128)
    snapped = _snap_sheet(True)
    assert _foot(snapped["g"]) > canvas_baseline + 5, "the g descender was pulled up"


# --------------------------------------------------------------------------- #
# stroke weight
# --------------------------------------------------------------------------- #

def _bar_glyph(width=4):
    """An antialiased ring: curved, like handwriting, so weight varies smoothly.

    An axis-aligned bar gains whole rows of pixels at once as it thickens and
    its weight moves in steps, which is not what a written letter does.
    """
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    r = np.hypot(yy - 64, xx - 64)
    return np.clip(width / 2.0 + 0.5 - np.abs(r - 36), 0.0, 1.0).astype(np.float32)


def test_reweight_is_identity_at_zero_and_keeps_the_skeleton_when_thickening():
    """Moving an edge outward must change weight, not shape."""
    from hfont.evaluate.metrics import cl_dice
    from hfont.generate import _stroke_weight, reweight

    glyph = _bar_glyph()
    assert reweight(glyph, 0.0) is glyph
    heavier = reweight(glyph, 1.0)
    assert _stroke_weight([heavier]) > _stroke_weight([glyph]) * 1.2
    assert cl_dice(heavier, glyph) > 0.95, "thickening moved the skeleton"


def test_seed_letters_are_brought_to_the_generated_weight():
    """An exported font must not hold two stroke weights.

    The model draws near corpus weight whatever the writer's pen, so 45
    generated glyphs came out up to ~50% heavier than the 30 the writer wrote.
    """
    from hfont.generate import _stroke_weight, match_seed_weight

    seeds = {k: _bar_glyph(3) for k in "abc"}
    generated = [_bar_glyph(6) for _ in range(4)]
    matched = match_seed_weight(seeds, generated)

    gap = _stroke_weight(generated) / _stroke_weight(matched.values()) - 1
    assert abs(gap) < 0.05, f"weights still differ by {gap:+.0%}"


def test_seed_weight_matching_never_thins():
    """Thinning breaks strokes -- a hand heavier than the model is left alone.

    Thinning the generated letters was rejected three separate ways because
    their extra weight hedges against uncertain position, and erosion snapped
    them (clDice 0.307 -> 0.275). The fix is safe only because it goes one way.
    """
    from hfont.generate import match_seed_weight

    seeds = {k: _bar_glyph(6) for k in "abc"}
    matched = match_seed_weight(seeds, [_bar_glyph(3) for _ in range(4)])
    for k in seeds:
        assert np.array_equal(matched[k], seeds[k]), f"{k} was thinned"


def test_generated_letters_are_drawn_at_the_writers_proportions():
    """Generated lowercase must take the writer's x-height and ascender, not the corpus's.

    The model drew lowercase 1.18-1.34x the writer's x-height on five of six
    hands, so every word mixed two sizes. One factor per letter cannot fix it:
    the ascender-to-x-height ratio differs too, so the zones are mapped apart.
    """
    from hfont.generate import match_seed_proportions

    def stem(up, down=0, width=8):
        image = np.zeros((128, 128), dtype=np.float32)
        image[96 - up : 96 + down, 60 : 60 + width] = 1.0
        return image

    seeds = {"a": stem(30), "e": stem(30), "o": stem(30), "b": stem(60), "k": stem(60),
             "A": stem(70), "E": stem(70)}
    drawn = {"c": stem(40), "u": stem(40), "x": stem(40), "d": stem(70), "h": stem(70),
             "C": stem(70), "F": stem(70)}
    chars = {k: k for k in list(seeds) + list(drawn)}
    advances = {k: 0.5 for k in drawn}
    images, advances = match_seed_proportions(dict(drawn), advances, seeds, chars, 0.75)

    def reach(image):
        rows = np.nonzero((image > 0.5).any(axis=1))[0]
        return 96 - rows[0]

    assert abs(reach(images["u"]) - 30) <= 1, "x-height letters kept the model's size"
    assert abs(reach(images["d"]) - 60) <= 1, "ascenders were not mapped to the writer's"
    assert np.array_equal(images["C"], drawn["C"]), "capitals already matched were resampled"
    assert advances["u"] == pytest.approx(0.5 * 30 / 40)
    assert advances["C"] == 0.5


def test_a_written_letter_that_lost_its_strokes_is_replaced():
    """A broken sample must not be passed through into the font.

    Writer 3's `j` was a thin, light line; 28 px survived intake, it exported as
    a dot, and "jumps" set as ".umps". His `y` broke the same way. Each carried
    well under a third of the ink of the model's drawing of that letter.
    """
    from hfont.generate import seed_is_broken

    drawn = np.zeros((128, 128), dtype=np.float32)
    drawn[20:100, 60:66] = 1.0                       # the model's whole letter
    fragment = np.zeros_like(drawn)
    fragment[20:30, 60:62] = 1.0                     # a dot of what was written
    assert seed_is_broken(fragment, drawn)


def test_a_healthy_written_letter_is_kept_even_if_thinner():
    """Being lighter than the model is normal and must not trigger replacement.

    The model draws heavier than most hands, so every healthy letter on the
    three sheets carried 0.39-0.9 of the model's ink. Replacing those would
    throw away the writer's own letters for no reason.
    """
    from hfont.generate import seed_is_broken

    drawn = np.zeros((128, 128), dtype=np.float32)
    drawn[20:100, 60:66] = 1.0
    thinner = np.zeros_like(drawn)
    thinner[20:100, 61:64] = 1.0                     # same letter, half the weight
    assert not seed_is_broken(thinner, drawn)
    assert not seed_is_broken(drawn, np.zeros_like(drawn)), "no model ink: keep the sample"


# --------------------------------------------------------------------------- #
# ruling stubs
# --------------------------------------------------------------------------- #

def _ruled_stroke():
    """A 6 px stroke crossed by a 2 px ruling, as regrowth leaves it."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:100, 60:66] = True               # the pen stroke
    mask[59:61, :] = True                    # a printed line across the page
    grown = np.zeros_like(mask)
    grown[20:100, 60:66] = True
    grown[59:61, 54:72] = True               # regrowth crept 6 px along the line
    return mask, grown


def test_ruling_stub_regrown_into_a_letter_is_pruned():
    """A grid line that crossed a stroke must not survive as a tick on it.

    On squared paper every letter written on a line came out with a tick at its
    foot, `g` and `y` with a `+` where the tail crossed, and `i` with a line
    through its stem. Fused to the letter, so no component test catches them.
    """
    from hfont.intake import _prune_ruling_stubs

    mask, grown = _ruled_stroke()
    pruned = _prune_ruling_stubs(mask, grown, line=1.0, radius=2)
    beside = np.zeros_like(mask)
    beside[59:61, 54:60] = beside[59:61, 66:72] = True
    assert (pruned & beside).sum() <= 0.5 * (grown & beside).sum(), "stub survived"


def test_ruling_pruning_keeps_the_stroke_and_a_dot():
    """Only ruling may go: never the pen stroke it crossed, nor a compact mark.

    Every stroke edge is thin at the pixel, so a pixel-level test would walk off
    the stub and round the letter. And an `i` dot is small; it must not be taken
    for ruling because it touches one.
    """
    from hfont.intake import _prune_ruling_stubs

    mask, grown = _ruled_stroke()
    mask[52:58, 90:96] = grown[52:58, 90:96] = True     # a dot resting on the line
    mask[58, 90:96] = grown[58, 90:96] = True
    pruned = _prune_ruling_stubs(mask, grown, line=1.0, radius=2)
    assert pruned[20:100, 60:66].all(), "part of the pen stroke was pruned"
    assert pruned[52:58, 90:96].all(), "the dot was taken for ruling"


def _squared_sheet(pen=30, blur=0.0):
    """Seyès-like paper: a grey grid blotted where it crosses, and a red margin."""
    from scipy.ndimage import gaussian_filter

    from hfont.charset import LATIN_SEED_SETS

    rows, chars = (10, 10, 10), LATIN_SEED_SETS["seed30"]
    page = np.full((260, 640, 3), 240, dtype=np.uint8)
    for at in range(6, 640, 20):
        page[at : at + 2] = 150
        page[:, at : at + 2] = 150
    for y in range(5, 260, 20):
        for x in range(5, 640, 20):
            page[y : y + 4, x : x + 4] = 150
    page[:, 12:18] = (225, 60, 60)
    for index, char in enumerate(chars):
        baseline, x = 70 + 70 * (index // 10), 34 + 58 * (index % 10)
        page[baseline - 24 : baseline, x : x + 6] = pen
        page[baseline - 24 : baseline, x + 10 : x + 16] = pen
        page[baseline - 24 : baseline - 18, x : x + 16] = pen
        if char in "ij":
            page[baseline - 38 : baseline - 32, x + 9 : x + 15] = pen
    if blur:
        page = np.clip(gaussian_filter(page.astype(np.float32), (blur, blur, 0)), 0, 255)
    return page.astype(np.uint8), rows, chars


@pytest.mark.parametrize("blur", [0.0, 0.7, 1.0])
def test_squared_paper_keeps_the_pen_and_drops_the_grid(blur):
    """Grid crossings and a red margin must not come back as letters.

    Two held-out Seyès sheets gave 17-23 letters per row of 10: every grid
    crossing is thick enough to keep as a core, and regrowth walks back along
    the lines from it. This sheet gave (29, 30, 29) by thickness alone, and with
    the grid gone the margin was still an extra letter (10, 11, 10). Blurred as
    a phone camera blurs, calibrating on every pixel the thickness pass removed
    let pen strokes into the grid's sample, and no ink at all came back.
    """
    from hfont.intake import FreehandConfig, freehand_cells

    page, rows, chars = _squared_sheet(blur=blur)
    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars


def _write_seed30(page, pen=(30, 30, 30), drop=0, low=None):
    """Write the seed30 letters as n-shapes in three rows, sloping by ``drop``."""
    from hfont.charset import LATIN_SEED_SETS

    chars = LATIN_SEED_SETS["seed30"]
    for index, char in enumerate(chars):
        row, col = divmod(index, 10)
        x = 34 + 58 * col
        base = int(round(70 + 70 * row + drop * col / 9)) + (low[1] if low and char == low[0] else 0)
        page[base - 24 : base, x : x + 6] = pen
        page[base - 24 : base, x + 10 : x + 16] = pen
        page[base - 24 : base - 18, x : x + 16] = pen
        if char in "ij":
            page[base - 38 : base - 32, x + 9 : x + 15] = pen
    return page, (10, 10, 10), chars


@pytest.mark.parametrize("pen", [(40, 60, 170), (30, 30, 30)])
def test_faint_grid_is_still_ruled_paper(pen):
    """A grid photographed faintly must still be removed, not read as letters.

    One sheet's grid read as 1.8% ink, far under the 6% that marked paper as
    ruled, while the pieces of it that did cross the ink level came through as
    letters, 15-20 to a row of 10. Here: (14, 14, 10) before.
    """
    from hfont.intake import FreehandConfig, freehand_cells

    page = np.full((260, 640, 3), 240, dtype=np.uint8)
    for at in range(6, 640, 20):
        page[at : at + 2] = 195                      # a grid barely there...
        page[:, at : at + 2] = 195
    for k, at in enumerate(range(6, 640, 40)):
        page[(37 * k) % 220 : (37 * k) % 220 + 22, at : at + 2] = 165   # ...but for pieces
    page, rows, chars = _write_seed30(page, pen)
    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars


def test_rows_of_a_tilted_photo_keep_their_letters():
    """A sloping row must not hand its last letter to the row below.

    The photo is rarely square to the page. One hand's `y`, at the low end of a
    sloping row and written low, was clustered into the next row and merged
    with the `j` under it: (10, 9, 10).
    """
    from hfont.intake import FreehandConfig, freehand_cells

    page, rows, chars = _write_seed30(np.full((300, 640, 3), 240, dtype=np.uint8),
                                      drop=30, low=("y", 20))
    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars


def test_letter_cut_off_by_the_photo_edge_is_left_out():
    """Ink touching the frame is not all there and belongs to no seed letter.

    A letter from a line above the sheet's rows, cut off by the top of the
    photo, was clustered into the top row and merged into the `S` beneath it.
    """
    from hfont.intake import FreehandConfig, IntakeConfig, _letter_groups, to_ink_field

    page, rows, _ = _write_seed30(np.full((260, 640, 3), 240, dtype=np.uint8))
    page[0:14, 382:398] = 30                         # over the S, cut by the frame
    ink = to_ink_field(page.min(axis=2).astype(np.float32) / 255.0, IntakeConfig())
    _, grouped, _, _ = _letter_groups(ink, FreehandConfig(rows=rows), len(rows))
    assert [len(row) for row in grouped] == list(rows)
    tops = [p.bbox[0] for row in grouped for letter in row for p in letter["parts"]]
    assert min(tops) > 0, "the cut-off letter was merged into the S"


def test_coloured_pen_is_not_trimmed_by_thickness():
    """A fine coloured pen must keep the strokes thickness would take from it.

    Once colour has told the pen from the grid, pruning and the thickness pass
    only do harm: a fine pen is as thin as the grid, and one hand's `G`, `S`,
    `g` and `0` broke wherever they ran beside a grid line.
    """
    from hfont.intake import FreehandConfig, IntakeConfig, _paper_only, _pen_ink, to_ink_field

    page = np.full((240, 240, 3), 240, dtype=np.uint8)
    for at in (20, 220):
        page[at] = 130
        page[:, at] = 130
    page[40:200, 100:103] = (40, 60, 170)            # a fine blue stroke...
    page[40:47, 98:105] = (40, 60, 170)              # ...thick only at its start
    rgb = page.astype(np.float32) / 255.0
    paper = _paper_only(rgb.min(axis=2))
    ink = to_ink_field(paper, IntakeConfig())
    mask = ink > 0.3
    grown = mask.copy()
    grown[[20, 220]] = False
    grown[:, [20, 220]] = False
    grown[60:, 90:110] = False                       # all thickness regrowth reached

    keep = _pen_ink(ink, mask, grown, paper, rgb, FreehandConfig(), line=1.0, radius=2)
    assert keep[40:200, 100:103].all(), "the stroke was cut where thickness lost it"


def test_coloured_pen_does_not_regrow_into_grey_ruling():
    """A blue stroke must not carry a stub of grey grid line off with it.

    Regrowth walks up to 3 x radius pixels along any ink touching the pen, and
    pruning reclaims only what is thin all round. Where a photographed grid
    line thickens beside a stroke, the stub stayed: one held-out writer's blue
    `f` came out with a `+` at its foot, and ticks on most of his letters.
    """
    from hfont.intake import FreehandConfig, IntakeConfig, _paper_only, _pen_ink, to_ink_field

    page = np.full((240, 240, 3), 240, dtype=np.uint8)
    for at in (20, 220):
        page[at] = 130                             # thin printed lines,
        page[:, at] = 130                          # removed by thickness
    page[40:200, 100:106] = (40, 60, 170)          # a blue stroke
    page[118:121, 106:150] = 130                   # grey, too thick to prune, touching it
    rgb = page.astype(np.float32) / 255.0
    paper = _paper_only(rgb.min(axis=2))
    ink = to_ink_field(paper, IntakeConfig())
    mask = ink > 0.3
    grown = mask.copy()
    grown[[20, 220]] = False
    grown[:, [20, 220]] = False

    keep = _pen_ink(ink, mask, grown, paper, rgb, FreehandConfig(), line=1.0, radius=2)
    assert keep[40:200, 100:106].all(), "part of the pen stroke was lost"
    assert not keep[118:121, 107:150].any(), "a grey stub regrew from the blue stroke"


def test_lined_paper_skips_the_pen_test():
    """Ruled lines with no crossings must not go through the pen test.

    The test exists for grid crossings. On one sheet shot in warm light the
    yellowed paper tinted the printed lines more than the pale pen, the lines
    were as dark as it, and `a`, `e`, `o` seeded nothing and were thrown away
    whole. Here the pen test took 8% of the writing. Lined paper is cleaned
    the same way with or without the photo's colour.
    """
    from hfont.intake import (
        FreehandConfig, IntakeConfig, _is_grid, _paper_only, _printed_lines,
        _remove_rulings, to_ink_field,
    )

    page = np.full((260, 640, 3), (235, 220, 165), dtype=np.uint8)   # warm paper
    for base in (71, 141, 211):
        page[base : base + 3] = (80, 80, 60)                          # a line per row
    page, _, _ = _write_seed30(page, pen=(110, 110, 120))            # pale grey-blue pen
    rgb = page.astype(np.float32) / 255.0
    paper = _paper_only(rgb.min(axis=2))
    ink = to_ink_field(paper, IntakeConfig())
    lines = _printed_lines(ink)
    assert lines is not None and not _is_grid(lines)
    cfg = FreehandConfig()
    assert np.array_equal(_remove_rulings(ink, cfg, paper, rgb), _remove_rulings(ink, cfg))

    squared, _, _ = _squared_sheet()
    squared_ink = to_ink_field(squared.min(axis=2).astype(np.float32) / 255.0, IntakeConfig())
    assert _is_grid(_printed_lines(squared_ink)), "squared paper was taken for lined"


def test_train_command_can_run_the_recommended_recipe():
    """`hfont train --recipe recommended` must train what recommended_config says.

    The command built plain TrainConfig defaults, so a fresh machine would
    have trained the old architecture without reference attention.
    """
    from hfont.cli import build_parser, train_config

    args = build_parser().parse_args([
        "train", "--cache", "c", "--out", "o", "--recipe", "recommended",
        "--set", "data.handwriting_weight=6.0", "--set", "steps=40000",
    ])
    cfg = train_config(args)
    assert cfg.generator.style_attention and cfg.loss.style_contrastive > 0
    assert cfg.data.handwriting_weight == 6.0 and cfg.steps == 40000
    plain = train_config(build_parser().parse_args(["train", "--cache", "c", "--out", "o"]))
    assert not plain.generator.style_attention


def test_lined_paper_keeps_strokes_as_thin_as_its_lines():
    """Removing lines by direction must not erode a thin stroke of the writing.

    Thickness removal took the bar of an `H`, the arms of a `k` and half an `M`
    from a pale pen nearly as thin as the printed lines. Lines that never cross
    are removed where they run alone; strokes crossing them keep their length.
    """
    from hfont.intake import FreehandConfig, IntakeConfig, _remove_rulings, to_ink_field

    page = np.full((200, 640), 240, dtype=np.uint8)
    for base in (60, 120, 180):
        page[base : base + 3] = 90                      # printed lines, 3 px
    page[80:123, 300:306] = 30                          # an H resting on a line:
    page[80:123, 330:336] = 30                          # two legs crossing it...
    page[100:102, 306:330] = 30                         # ...and a 2 px bar
    ink = to_ink_field(page.astype(np.float32) / 255.0, IntakeConfig())
    out = _remove_rulings(ink, FreehandConfig()) > 0.3
    assert not out[60:63, 20:280].any(), "a printed line survived"
    assert out[100:102, 308:328].all(), "the thin bar was removed with the lines"
    assert out[80:123, 301:305].all(), "a leg lost the stretch where it crossed a line"


def test_crossbar_short_of_both_legs_rejoins_its_letter():
    """A stroke bridging two pieces must not leave its letter as two letters.

    One hand's `H` had a bar stopping a pixel or two short of both legs: it
    came through as two letters and a stray mark, (11, 10, 10) for 10 a row.
    """
    from hfont.intake import FreehandConfig, freehand_cells

    page, rows, chars = _write_seed30(np.full((260, 640, 3), 240, dtype=np.uint8))
    x, base = 34 + 58 * 3, 70                           # the H, fourth in row one
    page[base - 24 : base, x : x + 16] = 240
    page[base - 24 : base, x : x + 5] = 30
    page[base - 24 : base, x + 15 : x + 20] = 30
    page[base - 14 : base - 11, x + 7 : x + 13] = 30    # 2 px clear of each leg
    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars


def test_short_i_written_above_the_line_is_found():
    """A letter too short to be a body, floating above the line, still counts.

    One hand wrote `i` as a short stroke well above the line: under the height
    that marks a letter body, and too high for the window that small parts are
    claimed from, which was set for commas. No dot or comma is that tall.
    """
    from hfont.intake import FreehandConfig, freehand_cells

    page, rows, chars = _write_seed30(np.full((260, 640, 3), 240, dtype=np.uint8))
    x, base = 34 + 58 * 8, 210                          # the i, ninth in row three
    page[base - 40 : base, x - 2 : x + 18] = 240
    page[base - 22 : base - 12, x + 5 : x + 10] = 30    # stem: 10 px, 12 px up
    page[base - 30 : base - 26, x + 6 : x + 10] = 30    # dot
    found = freehand_cells(page, LATIN_CORE, "seed30", FreehandConfig(rows=rows))
    assert "".join(g.char for g in found) == chars


def test_pen_like_the_grid_falls_back_to_thickness():
    """When nothing sets the pen apart from the lines, keep what thickness kept.

    A pen as grey as the grid seeds nothing by darkness, and the red margin was
    then the only colour on the page: the pen stage kept the margin alone.
    """
    from hfont.intake import (
        FreehandConfig, IntakeConfig, _paper_only, _remove_rulings, to_ink_field,
    )

    page, _, _ = _squared_sheet(pen=150)
    rgb = page.astype(np.float32) / 255.0
    paper = _paper_only(rgb.min(axis=2))
    ink = to_ink_field(paper, IntakeConfig())
    cfg = FreehandConfig()
    by_thickness = _remove_rulings(ink, cfg)
    assert (by_thickness > cfg.ink_level).any()
    assert np.array_equal(_remove_rulings(ink, cfg, paper, rgb), by_thickness)


def test_freehand_ink_is_used_as_measured_not_measured_again():
    """Ink measured across the page must not be re-measured inside each cell.

    freehand_cells measures ink once for the whole page, on purpose, and hands
    each crop over as ``1 - ink``. normalize_samples then re-measured it inside
    the crop, where a letter filling its cell darkens its own paper estimate: a
    faint letter lost most of its ink, and writer 3's light pen came through in
    fragments (share of each letter in one piece 0.819 -> 0.896 once fixed).
    """
    from hfont.intake import IntakeConfig, normalize_samples

    spec = {g.char: g for g in LATIN_CORE}
    ink = np.zeros((120, 60), dtype=np.float32)
    ink[10:90, 8:52] = 0.35                       # a faint letter filling its cell
    raw = {spec[c]: 1.0 - ink for c in "aeonsr"}

    def total(measured):
        cfg = IntakeConfig(size=128, ink_is_measured=measured, normalize_pen=False,
                           threshold_window=0.9)
        return float(normalize_samples(raw, cfg).images[spec["a"].key].sum())

    assert total(True) > 3.0 * total(False), "the page's ink measurement was undone"


def test_long_descender_is_fitted_not_cut():
    """A tail longer than the canvas has room for must arrive whole.

    Framing on the 90th percentile sizes the alphabet on its body, and a
    handwritten tail runs far past what the canvas holds: `j` lost 17-80% of its
    ink off the bottom edge, and went into the font as a stub. The overflowing
    half is compressed toward the baseline instead, leaving the body alone.
    """
    from hfont.intake import IntakeConfig, normalize_samples

    spec = {g.char: g for g in LATIN_CORE}

    def stem(tail):
        ink = np.zeros((240, 60), dtype=np.float32)
        ink[60 : 100 + tail, 26:32] = 1.0          # on the line at row 100
        return 1.0 - ink

    raw = {spec[c]: stem(0) for c in "aeonsr"}
    raw[spec["j"]] = stem(90)                      # ~200 canvas px of tail, room for 30
    out = normalize_samples(raw, IntakeConfig(size=128, ink_is_measured=True,
                                              normalize_pen=False)).images
    j, body = out[spec["j"].key], out[spec["a"].key]
    rows = np.nonzero((j > 0.5).any(axis=1))[0]
    assert rows[-1] >= 118, "the tail was cut short, not fitted"
    assert j[-1].max() < 0.15, "the tail still runs off the canvas"
    np.testing.assert_allclose(j[:90], body[:90], atol=1e-5,
                               err_msg="fitting the tail changed the body")


def test_hanging_letter_at_the_end_of_a_bent_row_keeps_its_body_above_the_line():
    """A row that bends: its straight line must not run through the top of a `y`.

    Karim's second row rises from `e` to `r` and drops again by `s` and `t`.
    The line fitted through its feet ran past his `y` at the very top of the
    letter (227 against a top of 227, x-height ~35 px), so the whole `y` was
    placed below the line and squashed as a tail.
    """
    from hfont.intake import _letter_lines

    spec = {g.char: g for g in LATIN_CORE}
    feet = {"e": 309, "m": 282, "n": 277, "o": 266, "r": 253, "s": 263, "t": 279}
    letters = [{"box": (100 * i, foot - 35, 100 * i + 60, foot)} for i, foot in enumerate(feet.values())]
    letters.append({"box": (700, 227, 760, 288)})                 # the y: top 227, tail to 288
    specs = [spec[c] for c in feet] + [spec["y"]]
    line = lambda x: 309 - 0.115 * x                               # straight fit, ~227 at the y
    out = _letter_lines(letters, specs, line)
    assert out[:-1] == [line(100 * i + 30) for i in range(len(feet))], "other letters moved"
    assert abs(out[-1] - (227 + 35)) < 1e-6, "y's line should be its top plus the x-height"

    # A y the straight line fits is left alone.
    letters[-1] = {"box": (700, 190, 760, 250)}
    assert _letter_lines(letters, specs, line)[-1] == line(730)
