"""Build the gate 3 page: is this letter by the same person? (private: real handwriting)

Writes ``output/gate3/index.html``, one self-contained page to open in any
browser and hand to raters, and ``output/gate3/plan.json``, which says which
probe was generated, genuine or a stranger's. The page itself holds only trial
numbers and images, so a rater reading its source learns nothing. Both files
show real people's handwriting: they stay on disk and out of git
(``output/gate3/`` is ignored). Share the page privately, never publish it.

Every letter shown goes through the same treatment as in the font: the
writer's letters are thickened to the model's weight, and everything is shown
as clean black ink, so weight and blur do not give the answer away.

Each rater types a name, answers yes/no per trial (keys Y and N), and at the
end downloads ``gate3_<name>.json``. Score with scripts/gate3_score.py.

    python scripts/gate3_page.py --checkpoint runs/hw40k/best.pt
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_writers import find_photos  # noqa: E402
from hfont.charset import LATIN_CORE  # noqa: E402
from hfont.evaluate.gate3 import plan_trials, trials_as_dicts  # noqa: E402
from hfont.generate import content_images_from_font, load_checkpoint, match_seed_weight  # noqa: E402
from hfont.intake import load_freehand_photo  # noqa: E402
from how_good import leave_one_out  # noqa: E402

CHAR = {g.key: g.char for g in LATIN_CORE}
KEY = {g.char: g.key for g in LATIN_CORE}


def png(image: np.ndarray, size: int = 96) -> str:
    from PIL import Image

    ink = (image > 0.5).astype(np.float32)
    tile = Image.fromarray(((1.0 - ink) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    tile.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Same hand?</title>
<style>
:root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --line:#d9d9d2; --accent:#2f5d8a; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecece6; --muted:#a3a39b; --line:#3a3a36; --accent:#8fb6dd; --card:#22221f; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; }
p { margin: 0 0 12px; }
.muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 16px 0; }
.refs { display:flex; flex-wrap:wrap; gap:6px; }
.refs img, .probe img { background:#fff; border:1px solid var(--line); border-radius:6px; }
.refs img { width: 72px; height: 72px; }
.probe { display:flex; justify-content:center; margin: 8px 0 4px; }
.probe img { width: 144px; height: 144px; border: 2px solid var(--accent); }
.buttons { display:flex; gap:12px; justify-content:center; margin-top: 12px; }
button { font: inherit; padding: 10px 22px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); }
input { font: inherit; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--fg); width: 100%; max-width: 320px; }
.bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); width: 0; }
.hidden { display:none; }
</style></head>
<body><main>
<h1>Same hand?</h1>
<section id="start">
  <p>Each screen shows a few letters written by one person, then one more letter.
  Decide whether that last letter was written by <b>the same person</b>.</p>
  <p class="muted">Some are, some are not. There is no time limit; go with your judgement.
  Keys: <b>Y</b> for yes, <b>N</b> for no. Your answers stay in this browser until you download them at the end.</p>
  <p><input id="name" placeholder="Your name or initials" autocomplete="off"></p>
  <button class="primary" id="go">Start</button>
</section>
<section id="trial" class="hidden">
  <div class="bar"><div id="progress"></div></div>
  <p class="muted" id="count"></p>
  <div class="card"><p class="muted">Letters by this person:</p><div class="refs" id="refs"></div></div>
  <div class="card"><p>Is this letter by the same person?</p>
    <div class="probe"><img id="probe" alt="letter to judge"></div>
    <div class="buttons"><button class="primary" id="yes">Yes, same person (Y)</button><button id="no">No (N)</button></div>
  </div>
</section>
<section id="done" class="hidden">
  <p>Done, thank you. Please download your answers and send the file back.</p>
  <button class="primary" id="save">Download answers</button>
</section>
</main>
<script>
const IMAGES = __IMAGES__;
const TRIALS = __TRIALS__;
const STORE = "gate3:" + __STAMP__;
let state = null;
function load() { try { return JSON.parse(localStorage.getItem(STORE)); } catch (e) { return null; } }
function keep() { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} }
function shuffled(n) { const a = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
const $ = (id) => document.getElementById(id);
function show(id) { for (const s of ["start", "trial", "done"]) $(s).classList.toggle("hidden", s !== id); }
let shownAt = 0;
function render() {
  if (state.next >= state.order.length) { show("done"); return; }
  const t = TRIALS[state.order[state.next]];
  $("refs").replaceChildren(...t.refs.map((id) => { const im = new Image(); im.src = IMAGES[id]; im.alt = "reference letter"; return im; }));
  $("probe").src = IMAGES[t.probe];
  $("count").textContent = (state.next + 1) + " of " + state.order.length;
  $("progress").style.width = (100 * state.next / state.order.length) + "%";
  show("trial"); shownAt = performance.now();
}
function answer(same) {
  if (!state || state.next >= state.order.length || $("trial").classList.contains("hidden")) return;
  const t = TRIALS[state.order[state.next]];
  state.answers.push({ rater: state.rater, trial_id: t.id, same: same, ms: Math.round(performance.now() - shownAt), position: state.next });
  state.next += 1; keep(); render();
}
$("go").onclick = () => {
  const name = $("name").value.trim(); if (!name) { $("name").focus(); return; }
  state = { rater: name, order: shuffled(TRIALS.length), next: 0, answers: [] }; keep(); render();
};
$("yes").onclick = () => answer(true);
$("no").onclick = () => answer(false);
document.addEventListener("keydown", (e) => { if (e.key === "y" || e.key === "Y") answer(true); if (e.key === "n" || e.key === "N") answer(false); });
$("save").onclick = () => {
  const blob = new Blob([JSON.stringify({ stamp: __STAMP__, answers: state.answers }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "gate3_" + state.rater.replace(/[^A-Za-z0-9_-]+/g, "_") + ".json"; a.click();
};
state = load();
if (state && state.rater) { $("name").value = state.rater; render(); }
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/phase1c/hfont_step030000.pt"))
    ap.add_argument("--content", default=str(ROOT / "data/fonts/NotoSans[wdth,wght].ttf"))
    ap.add_argument("--out", default=str(ROOT / "output/gate3"))
    ap.add_argument("--per-writer", type=int, default=4, help="probe letters per writer and condition")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    model = load_checkpoint(args.checkpoint)
    content = content_images_from_font(args.content, model.charset)
    letters, blind = {}, {}
    for name, photo in find_photos(ROOT).items():
        refs = load_freehand_photo(str(photo), LATIN_CORE, "seed30", model.image_size)
        rows = leave_one_out(model, refs, content)
        generated = {CHAR[k]: r["gen"] for k, r in rows.items()}
        # As in the font: the writer's letters at the model's weight.
        shown = match_seed_weight(refs, list(generated.values()))
        letters[name] = {CHAR[k]: v for k, v in shown.items()}
        blind[name] = generated
        print(f"{name}: {len(refs)} letters", flush=True)

    trials = plan_trials({w: set(v) for w, v in letters.items()}, args.per_writer, seed=args.seed)
    rng = random.Random(args.seed + 1)
    images: dict[str, str] = {}

    def add(image: np.ndarray) -> str:
        while (handle := f"i{rng.randrange(16**6):06x}") in images:
            pass
        images[handle] = png(image)
        return handle

    ref_ids = {(w, c): add(img) for w, d in letters.items() for c, img in d.items()}
    page_trials = []
    for t in trials:
        probe = blind[t.writer][t.char] if t.condition == "generated" else letters[t.source][t.char]
        page_trials.append({"id": t.trial_id, "refs": [ref_ids[(t.writer, c)] for c in t.references],
                            "probe": add(probe)})
    stamp = f"{rng.randrange(16**8):08x}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    html = (PAGE.replace("__IMAGES__", json.dumps(images))
                .replace("__TRIALS__", json.dumps(page_trials))
                .replace("__STAMP__", json.dumps(stamp)))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "plan.json").write_text(json.dumps({"stamp": stamp, "checkpoint": args.checkpoint,
                                               "trials": trials_as_dicts(trials)}, indent=1),
                                   encoding="utf-8")
    print(f"{len(trials)} trials ({len(trials) // 3} per condition) -> {out / 'index.html'}")
    print("private: real handwriting. Share it directly, do not publish or commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
