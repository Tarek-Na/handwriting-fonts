"""A1: letters written at different heights still join, by cursive attachment."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

hb = pytest.importorskip("uharfbuzz")

from hfont.charset import Form, GlyphSpec  # noqa: E402
from hfont.fontbuild.joined import build_joined_font, join_points  # noqa: E402

BEH, ALEF = "ب", "ا"


def _letter(left: int, right: int, height: int, stem: int | None = None) -> np.ndarray:
    """A pen line along the writing line from ``left`` to ``right`` at row ``height``,
    with an optional upright stem: a crude letter with a connecting stroke."""
    im = np.zeros((128, 128), np.float32)
    im[height - 2:height + 3, left:right] = 1.0
    if stem is not None:
        im[height - 40:height, stem - 2:stem + 3] = 1.0
    return im


@pytest.fixture(scope="module")
def font():
    # Each form is written at its own height, as a hand does: nothing lines up.
    images = {
        GlyphSpec(BEH, Form.ISOL): _letter(40, 90, 96, stem=64),
        GlyphSpec(BEH, Form.INIT): _letter(30, 80, 90, stem=70),
        GlyphSpec(BEH, Form.MEDI): _letter(30, 100, 99, stem=64),
        GlyphSpec(BEH, Form.FINA): _letter(40, 100, 93, stem=60),
        GlyphSpec(ALEF, Form.ISOL): _letter(60, 68, 96, stem=64),
        GlyphSpec(ALEF, Form.FINA): _letter(60, 100, 101, stem=62),
    }
    ttfont, glyphs = build_joined_font(images, radius=2.0)
    buffer = io.BytesIO()
    ttfont.save(buffer)
    return hb.Font(hb.Face(buffer.getvalue())), ttfont, {g.outline.name: g for g in glyphs}


def _shape(font, text):
    hbfont, ttfont, glyphs = font
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbfont, buf, {"curs": True})
    order = ttfont.getGlyphOrder()
    placed, cursor = [], 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):   # visual order, left to right
        placed.append((order[info.codepoint], cursor + pos.x_offset, pos.y_offset))
        cursor += pos.x_advance
    return placed


def test_join_points_are_the_ends_of_the_pen_path_on_the_joining_sides():
    line = np.zeros((128, 128), bool)
    line[90, 30:100] = True
    line[60:90, 99] = True             # something taller also reaches the right edge
    assert join_points(line, Form.ISOL) == (None, None)
    assert join_points(line, Form.INIT) == (None, (30, 90))
    assert join_points(line, Form.FINA) == ((99, 90), None), "the lowest point: the connecting stroke"
    assert join_points(line, Form.MEDI) == ((99, 90), (30, 90))


def test_the_shaper_picks_positional_forms(font):
    names = [name for name, _, _ in _shape(font, BEH * 3)]
    assert [n.split(".")[-1] for n in names] == ["fina", "medi", "init"]   # visual order


@pytest.mark.parametrize("word", [BEH * 2, BEH * 3, BEH * 4, BEH + BEH + ALEF])
def test_every_join_meets_although_the_letters_were_written_at_different_heights(font, word):
    glyphs = font[2]
    placed = _shape(font, word)
    assert len(placed) == len(word)
    for (left, lx, ly), (right, rx, ry) in zip(placed, placed[1:]):
        arrive = glyphs[left].entry        # the later letter: the pen arrives on its right
        leave = glyphs[right].exit         # the earlier letter: the pen leaves on its left
        assert arrive and leave, f"{right} -> {left} has no join points"
        assert (lx + arrive[0], ly + arrive[1]) == (rx + leave[0], ry + leave[1]), \
            f"{right} does not meet {left}"
    assert placed[0][2] == 0, "the last letter of the run sits on the baseline"


def test_without_cursive_attachment_the_same_letters_miss(font):
    """What the join points buy: by advance alone the heights do not meet."""
    hbfont, ttfont, glyphs = font
    buf = hb.Buffer()
    buf.add_str(BEH * 3)
    buf.guess_segment_properties()
    hb.shape(hbfont, buf, {"curs": False})
    assert all(pos.y_offset == 0 for pos in buf.glyph_positions)
    heights = {glyphs[n].entry[1] for n in glyphs if glyphs[n].entry} | \
              {glyphs[n].exit[1] for n in glyphs if glyphs[n].exit}
    assert len(heights) > 1


def _tailed(tail_row: int, body: tuple[int, int] = (50, 80)) -> np.ndarray:
    """A pen path: a bump (the letter) with long tails to both edges at ``tail_row``."""
    line = np.zeros((128, 200), bool)
    line[tail_row, 10:190] = True
    line[tail_row - 30:tail_row, body[0]] = True
    line[tail_row - 30, body[0]:body[1]] = True
    line[tail_row - 30:tail_row, body[1]] = True
    return line


def test_long_tails_become_short_stubs_that_end_on_the_writing_line():
    from hfont.fontbuild.joined import retarget

    line = _tailed(tail_row=93)                  # written 3 rows under the line at 90
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    assert (entry, leave) == join_points(out, Form.MEDI)
    assert entry[1] == 90 and leave[1] == 90, "both joins on the writing line"
    assert entry[0] - leave[0] < 60, "180 px of tail cut to the letter plus two stubs"
    assert out[60:64, 50:81].any(), "the letter itself is untouched"
    kept, _ = retarget(line, Form.FINA, baseline_row=90, stub=8, tolerance=6)
    assert kept[93, 10:40].all(), "a side that does not join keeps its stroke"


def test_a_short_steep_stub_in_a_tight_crop_still_joins_on_the_writing_line():
    from hfont.fontbuild.joined import joined_glyph

    image = np.zeros((128, 9), np.float32)       # cropped tight: short tails touch both edges
    image[95:99, 0:9] = 1                        # the stroke, 16 rows under the line
    image[60:99, 3:6] = 1                        # a stem, so there is a letter body
    glyph = joined_glyph(GlyphSpec("ب", Form.MEDI), image, radius=2.0,
                         baseline_px=80, scale=10.0, stub=0.06)
    assert abs(glyph.entry[1]) <= 5 and abs(glyph.exit[1]) <= 5, (glyph.entry, glyph.exit)


def test_the_forms_cell_border_is_not_taken_for_ink():
    import io as _io

    from PIL import Image

    from hfont.data import ahawp

    grey = np.full((128, 200), 255, np.uint8)
    grey[30:60, 60:66] = 0                       # the letter, well above the bottom
    grey[60:64, 60:120] = 0
    grey[125:127, :] = 0                         # the cell border along the bottom edge
    png = _io.BytesIO()
    Image.fromarray(grey).save(png, format="PNG")
    im = ahawp.ink(png.getvalue())
    assert not im[120:].any(), "the border is gone"
    assert im[30:64, 60:120].sum() == grey[30:64, 60:120].size - (grey[30:64, 60:120] > 0).sum()

    joined = grey.copy()
    joined[64:127, 60:62] = 0                    # a stroke that reaches the border joins it
    png = _io.BytesIO()
    Image.fromarray(joined).save(png, format="PNG")
    assert ahawp.ink(png.getvalue())[125:127].any(), "ink touching the letter is kept"


def test_cutting_a_tail_does_not_move_the_join_to_another_stroke():
    from hfont.fontbuild.joined import retarget

    line = _tailed(tail_row=93)                  # bump plus 180 px of tail
    line[110, 20:170] = True                     # a separate stroke under it, 150 px:
    #                                              smaller than the tailed one, larger once cut
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    assert entry[1] == 90 and leave[1] == 90
    assert out[entry[1], entry[0]] and out[leave[1], leave[0]]


def test_a_dot_under_a_cut_tail_moves_inside_the_join():
    from hfont.fontbuild.joined import retarget

    line = _tailed(tail_row=90)                  # letter at columns 50-80, tails to 10 and 190
    line[100:102, 25:27] = True                  # a dot under the left tail, column 25
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    ys, xs = np.nonzero(out[95:])
    assert xs.size and leave[0] < xs.min() and xs.max() < entry[0], (leave, entry, xs)


def test_a_dot_is_never_taken_for_the_join():
    line = np.zeros((128, 128), bool)
    line[90, 40:100] = True
    line[110:113, 20:23] = True                  # a dot, further left than the letter
    assert join_points(line, Form.INIT) == (None, (40, 90))


def test_ahawp_names_map_to_letters_and_forms():
    from hfont.data import ahawp

    assert ahawp.spec_for("beh", "middle") == GlyphSpec(BEH, Form.MEDI)
    assert ahawp.spec_for("ain", "regular") == GlyphSpec("ع", Form.ISOL)
    assert ahawp.spec_for("lam", "alif") == ahawp.LAM_ALEF
    assert ahawp.spec_for("alif", "hamza") == GlyphSpec("أ")
    assert ahawp.spec_for("beh", "sideways") is None


def test_the_writing_line_is_where_the_connecting_stroke_runs():
    from hfont.data import ahawp

    def letter(row, left, right, drop=0):
        im = np.zeros((128, 160), np.float32)
        im[row - 2:row + 3, left:right] = 1.0             # the connecting stroke
        im[row - 30:row + drop, 70:75] = 1.0              # a stem, hanging `drop` below it
        return im

    fina, isol = GlyphSpec("ر", Form.FINA), GlyphSpec("ر", Form.ISOL)
    letters = {fina: letter(80, 70, 150, drop=20), isol: letter(60, 60, 90, drop=20)}
    lines = ahawp.writing_lines(letters)
    assert abs(lines[fina] - 80) <= 2
    assert abs(lines[isol] - 60) <= 4, "the isolated form hangs below the line as its final form does"


def test_scribbled_copies_are_passed_over():
    from hfont.data import ahawp

    good = np.zeros((64, 64), np.float32)
    good[30:34, 10:50] = 1
    scribble = np.ones((64, 64), np.float32)
    assert ahawp.typical([good, scribble, good.copy(), good.copy()]) != 1


def test_two_dots_under_a_cut_tail_stay_two_dots():
    from scipy.ndimage import label

    from hfont.fontbuild.joined import retarget

    line = _tailed(tail_row=90)                  # letter at columns 50-80, tails to 10 and 190
    line[100:102, 20:22] = True                  # two dots under the left tail, 8 px apart
    line[100:102, 30:32] = True
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    below = out[95:]
    assert label(below, np.ones((3, 3), int))[1] == 2, "still two dots"
    xs = np.nonzero(below)[1]
    assert leave[0] < xs.min() and xs.max() < entry[0]
    assert xs.max() - xs.min() == 11, "their spacing kept"


def test_pulled_in_dots_do_not_land_on_the_letter():
    from scipy.ndimage import label

    from hfont.fontbuild.joined import retarget

    line = np.zeros((128, 200), bool)
    line[90, 10:190] = True                      # the connecting stroke, cut to stubs
    line[60:90, 70] = True                       # a tooth at column 70
    line[56:58, 20:22] = True                    # a dot at the tooth's height, under the left tail
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    assert label(out, np.ones((3, 3), int))[1] == 2, "the dot is still its own piece"
    ys, xs = np.nonzero(out[50:60])
    assert leave[0] < xs.min() and xs.max() < entry[0], "and inside the joins"


def test_a_letter_drawn_over_its_base_as_a_separate_stroke_is_not_cut_away():
    from hfont.fontbuild.joined import retarget

    line = np.zeros((128, 200), bool)
    line[90, 10:190] = True                      # a medial kaf's flat base, one stroke
    line[30:80, 100] = True                      # and its upper stroke, drawn separately
    line[30, 100:130] = True
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    assert entry[0] > leave[0], "entry right of exit"
    ys, xs = np.nonzero(out[25:85])
    assert leave[0] < xs.min() and xs.max() < entry[0], "the letter sits between its joins"
    assert entry[0] - leave[0] < 80, "and the long base ends were cut to stubs"


def test_a_broken_off_tail_is_removed_but_dots_stay():
    from hfont.fontbuild.joined import retarget

    line = _tailed(tail_row=90)                  # letter at columns 50-80, tails to 10 and 190
    line[90, 120:135] = False                    # a break in the right tail: 135-190 is its own piece
    line[100:102, 160:162] = True                # and a dot under it, off the line
    out, (entry, leave) = retarget(line, Form.MEDI, baseline_row=90, stub=8, tolerance=6)
    assert not out[85:95, entry[0] + 1:].any(), "nothing left on the line past the join"
    ys, xs = np.nonzero(out[98:104])
    assert xs.size and leave[0] < xs.min() and xs.max() < entry[0], "the dot is kept, inside the joins"
    assert entry[0] - leave[0] < 70
