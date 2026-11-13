"""Retrieve-and-refine: the content letter comes from the nearest training hand."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.data import neighbors  # noqa: E402


def _bar(angle: float) -> np.ndarray:
    """A thick stroke at an angle: one hand's way of writing every letter."""
    yy, xx = np.mgrid[:64, :64].astype(np.float32) - 32
    t = np.deg2rad(angle)
    return (np.abs(-np.sin(t) * xx + np.cos(t) * yy) < 4).astype(np.float32)


def _store(hands: dict[str, tuple[float, str]], keys="ab"):
    """Fake store: each hand writes every key as a bar at its own angle."""
    records = [SimpleNamespace(font_id=f, split=split, category="SD19")
               for f, (_, split) in hands.items()]
    return SimpleNamespace(records=records, has=lambda f, k: k in keys,
                           image=lambda f, k: _bar(hands[f][0]))


def test_neighbours_are_the_most_alike_training_hands_never_itself_or_held_out():
    store = _store({"me": (10, "train"), "close": (14, "train"), "far": (80, "train"),
                    "held": (11, "val")})
    table = neighbors.build(store, ["a", "b"], k=2)
    assert table["me"] == ["close", "far"]          # not itself, not the held-out writer
    assert table["held"][0] == "me"                 # held-out writers get neighbours too
    assert "held" not in sum(table.values(), [])


def test_bank_gives_each_letter_from_the_nearest_hand_that_has_it():
    hands = {"close": (14, "train"), "far": (80, "train")}
    store = _store(hands, keys="ab")
    bank = neighbors.Bank(store, ["a"])
    fallback = {"a": np.zeros((64, 64)), "z": np.ones((64, 64))}
    got = bank.content({"a": _bar(10)}, fallback)
    assert np.array_equal(got["a"], _bar(14))
    assert np.array_equal(got["z"], fallback["z"]), "a letter no hand has keeps the fallback"


CACHE = Path(__file__).resolve().parents[1] / "data" / "cache_local"


@pytest.mark.skipif(not (CACHE / "meta.json").is_file(), reason="no local glyph cache")
def test_training_content_comes_from_a_neighbour(monkeypatch):
    from hfont.data.dataset import DatasetConfig, FontEvalDataset, GlyphPairDataset, GlyphStore

    store = GlyphStore(CACHE)
    train = [r.font_id for r in store.split("train")]
    near = train[-1]
    table = {f: [near] for f in train}
    with pytest.raises(FileNotFoundError):
        GlyphPairDataset(store, DatasetConfig(content_source="nearest"))
    monkeypatch.setattr(neighbors, "load", lambda cache_dir: table)
    ds = GlyphPairDataset(store, DatasetConfig(content_source="nearest", epoch_size=8))
    for i in range(8):
        sample = ds[i]
        key = {i: k for k, i in ds.glyph_index.items()}[int(sample["char_id"])]
        if store.has(near, key):
            expected = store.image(near, key) * 2.0 - 1.0
            assert np.allclose(sample["content"][0].numpy(), expected, atol=1e-6)

    val = FontEvalDataset(store, split="val", max_fonts=1, neighbors={
        r.font_id: [near] for r in store.split("val")})
    font_id, key = val._items[0]
    if store.has(near, key):
        assert np.allclose(val[0]["content"][0].numpy(), store.image(near, key) * 2.0 - 1.0, atol=1e-6)
