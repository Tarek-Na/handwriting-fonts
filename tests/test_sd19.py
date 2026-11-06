"""SD19 writers become training fonts with the right labels and letters on a line."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.data.sd19 import index_writers, parse_cls, writer_glyphs  # noqa: E402


def _png(ink: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _centred(h: int, w: int = 30) -> np.ndarray:
    """A solid block of ink centred in a 128x128 box, as SD19 stores every letter."""
    ink = np.zeros((128, 128), bool)
    top, left = 64 - h // 2, 64 - w // 2
    ink[top : top + h, left : left + w] = True
    return ink


def _form(letters: dict[str, np.ndarray], form: str = "0001_01", hsf: str = "hsf_0"):
    """A by_write zip holding one lowercase field, plus its .cls text."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, ink in enumerate(letters.values()):
            z.writestr(f"by_write/{hsf}/f{form}/l{form}/l{form}_{i:05d}.png", _png(ink))
    cls = f"{len(letters)}\n" + "".join(f"{ord(c):02x}\n" for c in letters)
    buf.seek(0)
    return zipfile.ZipFile(buf), {f"{hsf}/f{form}/l{form}.cls": cls}


def test_cls_lists_characters_in_image_order():
    assert parse_cls("3\n61\n4c\n30\n") == ["a", "L", "0"]
    with pytest.raises(ValueError):
        parse_cls("4\n61\n62\n")


def test_each_png_gets_the_label_at_its_index():
    letters = {"o": _centred(40), "b": _centred(70), "p": _centred(70)}
    z, labels = _form(letters)
    (writer,) = index_writers(z, labels)
    assert writer.writer_id == "hsf_0/f0001_01"
    for i, ch in enumerate(letters):
        assert writer.members[ch] == [f"by_write/hsf_0/f0001_01/l0001_01/l0001_01_{i:05d}.png"]


def test_a_field_whose_counts_disagree_is_dropped_whole():
    z, labels = _form({"o": _centred(40), "b": _centred(70)})
    key = next(iter(labels))
    labels[key] = "3\n6f\n62\n63\n"   # one label too many: every pairing is suspect
    assert index_writers(z, labels) == []


def test_descender_is_put_back_below_the_line():
    """SD19 centres every letter; a p must still hang below the baseline after conversion."""
    letters = {c: _centred(40) for c in "aceonrsuvwxz"}   # x-height letters
    letters["b"] = _centred(70)                            # ascender
    letters["p"] = _centred(70)                            # descender, same size as b
    z, labels = _form(letters)
    (writer,) = index_writers(z, labels)
    out = writer_glyphs(z, writer, LATIN_CORE)
    key = {g.char: g.key for g in LATIN_CORE}

    def extent(ch):
        rows = np.nonzero((out[key[ch]] > 0.5).any(axis=1))[0]
        return rows[0], rows[-1]

    line = 0.75 * 128
    o_top, o_bottom = extent("o")
    b_top, b_bottom = extent("b")
    p_top, p_bottom = extent("p")
    assert abs(o_bottom + 1 - line) <= 2 and abs(b_bottom + 1 - line) <= 2, "o and b sit on the line"
    assert p_bottom > line + 10, "p was left sitting on the line"
    assert abs(p_top - o_top) <= 3, "p's bowl should start at the x-height, like o"
    assert b_top < o_top - 10, "b keeps its ascender: sizes survive conversion"


def test_unknown_widths_are_skipped_not_learned():
    """SD19 has no spacing; its NaN advances must not turn the whole loss into NaN."""
    import torch

    from hfont.models.losses import GlyphLoss, LossConfig

    loss = GlyphLoss(LossConfig(advance=1.0))
    image = torch.zeros(4, 1, 16, 16)
    pred = torch.tensor([0.5, 0.5, 0.5, 0.5])
    known = torch.tensor([0.4, 0.6, 0.4, 0.6])
    mixed = torch.tensor([0.4, 0.6, float("nan"), float("nan")])
    total_known, parts_known = loss.reconstruction(image, image, pred[:2], known[:2])
    total_mixed, parts_mixed = loss.reconstruction(image, image, pred, mixed)
    assert torch.isfinite(total_mixed)
    assert parts_mixed["advance"] == pytest.approx(parts_known["advance"])
    _, parts_none = loss.reconstruction(image, image, pred, torch.full((4,), float("nan")))
    assert "advance" not in parts_none


def test_writers_join_the_cache_and_are_sampled_by_their_own_weight(tmp_path):
    """Converted writers load like fonts, and sd19_weight decides how often they are drawn."""
    import json

    from hfont.data.dataset import DatasetConfig, GlyphPairDataset, GlyphStore
    from hfont.data.sd19 import add_to_cache

    letters = {c: _centred(40, 20 + i) for i, c in enumerate("aceonrsuvwxz")}
    letters |= {"b": _centred(70), "p": _centred(70)}
    z, labels = _form(letters)
    writers = index_writers(z, labels)
    glyphs = writer_glyphs(z, writers[0], LATIN_CORE)
    (tmp_path / "meta.json").write_text(json.dumps({
        "charset": LATIN_CORE.name, "glyph_keys": [g.key for g in LATIN_CORE],
        "render": {"size": 128}, "fonts": []}))
    assert add_to_cache(tmp_path, [(writers[0], glyphs), (writers[0], glyphs)]) == 2
    assert add_to_cache(tmp_path, [(writers[0], glyphs)]) == 1, "re-running replaces, not appends"

    # A stand-in typeset font beside the writer, to see who gets drawn.
    meta = json.loads((tmp_path / "meta.json").read_text())
    (writer_record,) = meta["fonts"]
    writer_record["split"] = "train"
    meta["fonts"].append(writer_record | {"font_id": "typeset", "category": "SANS_SERIF"})
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    store = GlyphStore(tmp_path)
    record = store.records[0]
    assert record.category == "SD19" and record.n_glyphs == len(letters)
    assert np.isnan(store.advance(record.font_id, store.keys(record.font_id)[0]))

    def drawn(weight):
        ds = GlyphPairDataset(store, DatasetConfig(split="train", sd19_weight=weight, epoch_size=60))
        return {ds.font_ids[int(ds[i]["font_id"])] for i in range(60)}

    assert drawn(1.0) == {record.font_id, "typeset"}
    assert drawn(0.0) == {"typeset"}
