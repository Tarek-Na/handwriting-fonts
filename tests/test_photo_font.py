"""A font from the short sheet: isolated letters in, every joined form from the model.

Needs the local derived data (templates, components) and the final model
checkpoint, none of which is in git; skipped without them. The letters are
drawn from the templates, not from anyone's handwriting.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

MODEL = ROOT / "runs/v1_drop/best.pt"
NEEDED = [ROOT / "data/ahawp/templates.npz", ROOT / "data/ahawp/variation_pca.npz", MODEL]
pytestmark = [pytest.mark.skipif(not all(p.is_file() for p in NEEDED), reason="no local model or data"),
              pytest.mark.filterwarnings("ignore::FutureWarning")]
pytest.importorskip("uharfbuzz")

LEFT_OUT = ["uni0639", "uni0647", "uni0637", "uni0642", "uni0648"]   # ain, heh, tah, qaf, waw


def _template_letters(skip=()):
    from hfont import skeleton
    from hfont.data import ahawp
    from ahawp_forms import specs_by_name
    from skeleton_moves import load_templates
    import variation

    temps = load_templates()
    specs = specs_by_name() | {s.name: s for s in ahawp.EXTRA.values()}
    return {specs[n]: skeleton.render_points(temps[n][0], temps[n][1], 2.5).astype(np.float32)
            for n in variation.isolated_names(variation.load_pca()) if n in specs and n not in skip}


def test_a_sheet_with_empty_boxes_still_gives_every_joined_form_in_versions_that_join():
    import ahawp_font
    import variation

    font, glyphs = variation.photo_font(_template_letters(LEFT_OUT), str(MODEL), versions=3)
    names = set(font.getGlyphOrder())
    joined = [n for n in variation.form_names(variation.load_pca()) if "." in n]
    assert all(n in names for n in joined), "the model draws the joined forms of letters left empty too"
    assert sum(n.endswith(".alt1") for n in names) >= len(joined) // 2, "with alternates"
    buffer = io.BytesIO()
    font.save(buffer)
    report = ahawp_font.check_joins(buffer.getvalue(), glyphs, ahawp_font.WORDS + ahawp_font.COMPOSED_WORDS)
    assert report["joins"] >= 20 and report["misses"] == 0       # 32 joins with every box filled


def test_the_same_letters_and_seed_give_the_same_font_and_another_seed_another():
    import variation
    from fontTools.pens.recordingPen import RecordingPen

    letters = _template_letters()

    def outlines(seed):
        font, _ = variation.photo_font(letters, str(MODEL), versions=2, seed=seed)
        gs = font.getGlyphSet()
        out = {}
        for n in font.getGlyphOrder():
            pen = RecordingPen()
            gs[n].draw(pen)
            out[n] = pen.value
        return out

    a, b, c = outlines(0), outlines(0), outlines(1)
    assert a == b
    assert a != c


def test_too_few_letters_are_refused_rather_than_misread():
    """One letter gives a worse style than none (val writers), so the font is not built from it."""
    import variation

    letters = _template_letters()
    two = dict(list(letters.items())[:variation.MIN_LETTERS - 1])
    with pytest.raises(ValueError, match="at least"):
        variation.photo_font(two, str(MODEL))
    three = dict(list(letters.items())[:variation.MIN_LETTERS])
    font, _ = variation.photo_font(three, str(MODEL), versions=1)
    assert "uni0628.medi" in font.getGlyphOrder()


MIX = ROOT / "runs/m1/best.pt"


@pytest.mark.skipif(not MIX.is_file(), reason="no local multi-template model")
def test_the_demo_draws_medial_ain_and_kaf_from_the_chosen_constructions():
    """The demo's medial ain and kaf come from the multi-template model's alternative templates
    (variation.CONSTRUCTIONS), in the writer's style; every other joined form from the final model."""
    import variation
    import variation_mix as vm

    letters = _template_letters()
    built = variation.construction_drawings(letters, n=3)
    assert set(built) == set(variation.CONSTRUCTIONS) and all(len(d) == 3 for d in built.values())
    tp = vm.templates_once()
    for form, k in variation.CONSTRUCTIONS.items():
        ids = tp.temps[f"{form}#{k}"][1]
        assert all(np.array_equal(i, ids) for _, i in built[form]), "drawn from the chosen template"

    from fontTools.pens.recordingPen import RecordingPen

    def outline(font, name):
        pen = RecordingPen()
        font.getGlyphSet()[name].draw(pen)
        return pen.value

    plain, _ = variation.photo_font(letters, str(MODEL), versions=3, seed=0)
    chosen, _ = variation.photo_font(letters, str(MODEL), versions=3, seed=0, constructions=built)
    assert outline(plain, "uni0639.medi") != outline(chosen, "uni0639.medi")
    assert outline(plain, "uni0643.medi") != outline(chosen, "uni0643.medi")
    assert outline(plain, "uni0628.medi") == outline(chosen, "uni0628.medi"), "other forms unchanged"


def test_without_the_multi_template_model_the_demo_draws_every_form_with_the_final_model(tmp_path):
    import variation

    assert variation.construction_drawings(_template_letters(), n=3, model=tmp_path / "missing.pt") == {}
