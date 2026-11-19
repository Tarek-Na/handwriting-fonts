"""The live demo's sheet check: it tells the writer what would make their font better."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hfont.charset import GlyphSpec  # noqa: E402


def _sheet(height: int, touch: str | None = None) -> dict:
    out = {}
    for ch in "ابجدرسصطعف":
        im = np.zeros((138, 548), np.float32)
        top = 20 if ch != touch else 0                                  # a letter running off its box
        im[top:20 + height, 270:275] = 1
        out[GlyphSpec(ch)] = im
    return out


def test_a_well_written_sheet_gets_no_advice():
    import demo_app

    letters = _sheet(80)
    letters.update({GlyphSpec(c): letters[GlyphSpec("ا")] for c in "قكلمن"})
    assert demo_app.check_sheet(letters) == []


def test_a_letter_cut_off_at_the_box_edge_is_named():
    import demo_app

    notes = demo_app.check_sheet(_sheet(80, touch="ع"))
    assert any("edge" in n and "ﻉ" in n for n in notes), notes


def test_small_writing_is_said_to_be_enlarged_and_few_letters_are_counted():
    import demo_app

    notes = demo_app.check_sheet(_sheet(30))
    assert any("small" in n for n in notes) and any("Only 10 of 20" in n for n in notes), notes


def test_warm_up_runs_a_sheet_at_launch_only_when_the_sample_is_there(tmp_path, monkeypatch):
    """The first real sheet must not pay for loading the model code (cold: 27 s to the page)."""
    import demo_app

    calls = []
    monkeypatch.setattr(demo_app, "OUT", tmp_path)
    monkeypatch.setattr(demo_app, "make", lambda job, seed, family, background=False: calls.append(
        (job.name, (job / "photo.png").is_file(), background)))
    monkeypatch.setattr(demo_app, "SAMPLE", tmp_path / "missing.png")
    assert demo_app.warm_up() == 0.0 and calls == []
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"png")
    monkeypatch.setattr(demo_app, "SAMPLE", sample)
    demo_app.warm_up()
    assert calls == [("warmup", True, False)], "one whole sheet, font included, before the first visitor"
