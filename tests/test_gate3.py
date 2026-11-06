"""The human test is balanced, never shows the answer, and anchors its score."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfont.evaluate.gate3 import plan_trials, score  # noqa: E402

WRITTEN = {f"w{i}": set("AEGHMORSabegkmnorsty036BDNfij") - ({"k"} if i == 0 else set())
           for i in range(4)}


def test_trials_are_balanced_and_the_probe_is_never_a_reference():
    trials = plan_trials(WRITTEN, per_writer=4)
    assert Counter(t.condition for t in trials) == {"generated": 16, "ceiling": 16, "floor": 16}
    for t in trials:
        assert t.char not in t.references, "the probe's character was shown as a reference"
        assert t.char in WRITTEN[t.writer]
        if t.condition == "floor":
            assert t.source != t.writer and t.char in WRITTEN[t.source]
        if t.condition == "ceiling":
            assert t.source == t.writer
    # The three conditions of one probe share character and references: only the hand differs.
    by_probe = {}
    for t in trials:
        by_probe.setdefault((t.writer, t.char), set()).add(t.references)
    assert all(len(refs) == 1 for refs in by_probe.values())


def test_plan_is_the_same_for_every_rater():
    assert plan_trials(WRITTEN, seed=3) == plan_trials(WRITTEN, seed=3)


def test_score_reads_generated_between_floor_and_ceiling():
    rows = []
    for rater in ("r1", "r2"):
        for i in range(10):
            rows.append({"rater": rater, "trial_id": i, "condition": "ceiling", "same": i < 9})
            rows.append({"rater": rater, "trial_id": 100 + i, "condition": "floor", "same": i < 1})
            rows.append({"rater": rater, "trial_id": 200 + i, "condition": "generated", "same": i < 5})
    s = score(rows)
    assert abs(s["ceiling"]["yes"] - 0.9) < 1e-9 and abs(s["floor"]["yes"] - 0.1) < 1e-9
    assert abs(s["anchored"] - 0.5) < 1e-9, "halfway between stranger and own hand"
    assert s["raters"] == 2 and s["agreement"] == 1.0
    lo, hi = s["generated"]["ci95"]
    assert lo < 0.5 < hi
