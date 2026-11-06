"""Score gate 3 from the answer files raters sent back.

    python scripts/gate3_score.py output/gate3/answers/*.json

Joins each answer to the plan written beside the page (which probe was
generated, the writer's own, or a stranger's) and prints the "same person"
rate for each with a 95% bootstrap interval, the anchored score
(generated - floor) / (ceiling - floor), and how often two raters agreed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.evaluate.gate3 import CONDITIONS, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--plan", default=str(ROOT / "output/gate3/plan.json"))
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    condition = {t["trial_id"]: t["condition"] for t in plan["trials"]}
    rows = []
    for path in args.answers:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("stamp") != plan["stamp"]:
            print(f"{path}: answers are for a different page; skipped")
            continue
        rows += [a | {"condition": condition[a["trial_id"]]} for a in data["answers"]]
    if not rows:
        print("no answers")
        return 1
    s = score(rows)
    print(f"{s['raters']} raters, {len(rows)} judgements")
    labels = {"generated": "model's letter", "ceiling": "their own letter", "floor": "a stranger's letter"}
    for c in CONDITIONS:
        lo, hi = s[c]["ci95"]
        print(f"  {labels[c]:20s} judged same person {s[c]['yes']:.0%}  (95% {lo:.0%}-{hi:.0%}, n={s[c]['n']})")
    print(f"  anchored score {s['anchored']:.2f}  (0 = no more like the writer than a stranger, 1 = as their own)")
    print(f"  rater agreement {s['agreement']:.0%}")
    if s["ceiling"]["yes"] <= s["floor"]["yes"]:
        print("  ceiling is not above floor: these raters cannot tell these hands apart; the test says nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
