"""Live demo: a photo of the 20-box sheet in, the writer's joining Arabic font out.

    python scripts/demo_app.py                 # then open http://localhost:8000
    python scripts/demo_app.py --host 0.0.0.0  # also reachable from a phone on the same Wi-Fi

The page shows each step: the letters as read off the photo, the model's raw
output (several versions of joined forms it was never shown), any sentence
set in the new font, and the .otf to download. "Write it again" draws new
versions, as a hand never writes a letter the same way twice.

Every photo, image and font made here is of a real person's handwriting:
it stays in ``output/demo/`` (git-ignored) and is never committed.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hfont import arabic_sheet as sheet  # noqa: E402
from hfont.intake import TemplateError  # noqa: E402

OUT = ROOT / "output/demo"
MODEL = ROOT / "runs/v1_drop/best.pt"            # the final model (tag a6-final-v1)
CHECKER = ROOT / "runs/c1/best.pt"               # flags a box whose writing is not its letter
SAMPLE = ROOT / "output/ahawp/short_sim_val0.png"
SHOW = ["uni0628.init", "uni0628.medi", "uni0633.medi", "uni0639.medi", "uni0647.medi", "uni062C.init",
        "uni0643.init", "uni0645.fina"]
DEFAULT_TEXT = "سلام عليكم"
POOL = None                                      # worker processes, started once (see start_pool)


def _warm(_):
    import hfont.fontbuild.joined  # noqa: F401  (each worker pays its imports before the first sheet)
    return os.getpid()


def start_pool(workers: int | None = None):
    """Worker processes for tracing the font's glyphs, started and warmed once, so a sheet
    does not wait for them (on Windows a new process takes seconds to start)."""
    global POOL
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    POOL = ProcessPoolExecutor(workers)
    list(POOL.map(_warm, range(workers)))
    return POOL


def _map(fn, jobs):
    return POOL.map(fn, jobs, chunksize=4) if POOL is not None else map(fn, jobs)


# ------------------------------------------------------------------ the pipeline
def read_photo(photo: Path):
    """(letters, flagged, turned) off a photo of the short sheet."""
    checker = CHECKER if CHECKER.is_file() else None
    return sheet.read_upright(sheet.load_photo(photo), checker, sheet.short_layout())


def check_sheet(letters: dict) -> list[str]:
    """What the writer should know about their sheet, from the stress test (val writers): photo
    quality, faint ink and pen made no difference; letters cut off at the box edge and very
    small writing did (small writing is rescaled for the model, see ``variation.SIZE_RANGE``)."""
    import variation

    notes = []
    touching = [s for s, im in letters.items()
                if (m := np.asarray(im) > 0.5)[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any()]
    if touching:
        notes.append("These letters reach the edge of their box and may be cut off: "
                     + " ".join(sheet.presentation_form(s) for s in touching)
                     + ". Writing them a little smaller would give a better font.")
    if letters:
        h, factor = variation.letter_height(letters), variation.size_factor(letters)
        if factor > 1.0:
            notes.append(f"The letters are small ({h:.0f} px; the training writers' were "
                         f"{variation.SIZE_RANGE[0]:.0f}-{variation.SIZE_RANGE[1]:.0f}): they were enlarged "
                         "for the model. Larger writing would give a better font.")
        elif factor < 1.0:
            notes.append(f"The letters are large ({h:.0f} px): they were reduced for the model.")
    if 3 <= len(letters) < 15:
        notes.append(f"Only {len(letters)} of 20 letters were read: the more letters, the closer the style.")
    return notes


def versions_image(drawn: dict[str, list[np.ndarray]]) -> np.ndarray:
    """A row per joined form, its label on the right (as Arabic reads), its versions to the left."""
    from PIL import Image, ImageDraw

    from ahawp_forms import specs_by_name
    from hfont.intake import _label_font

    specs = specs_by_name()
    cell_h, gap = 150, 24
    width = max(sum(im.shape[1] + gap for im in ims) for ims in drawn.values()) + 120
    page = Image.new("RGB", (width, cell_h * len(drawn)), "white")
    draw = ImageDraw.Draw(page)
    font = _label_font(44)
    for row, (name, ims) in enumerate(drawn.items()):
        y = row * cell_h
        draw.text((width - 90, y + 40), sheet.presentation_form(specs[name]), fill=(200, 30, 30), font=font)
        x = width - 120
        for im in ims:
            ink = Image.fromarray((255 * (1 - np.asarray(im, np.float32))).astype(np.uint8))
            if ink.height > cell_h - 10:
                ink = ink.resize((max(1, ink.width * (cell_h - 10) // ink.height), cell_h - 10))
            x -= ink.width + gap
            page.paste(ink.convert("RGB"), (x, y + (cell_h - ink.height) // 2))
        draw.line([(0, y + cell_h - 1), (width, y + cell_h - 1)], fill=(230, 230, 230))
    return np.asarray(page)


def make(job: Path, seed: int, family: str, background: bool = False) -> dict:
    """Read ``job/photo.*`` and write everything the result page shows; returns its notes.

    With ``background``, returns once the letters are read and the model's versions drawn
    (seconds); the font, which takes longer, is then built on a thread and appears as
    ``font.otf`` when done (``font_error.txt`` if it fails)."""
    import threading

    from PIL import Image

    import variation

    photo = next(job.glob("photo.*"))
    t0 = time.time()
    for name in ("font.otf", "font_error.txt"):
        (job / name).unlink(missing_ok=True)
    letters, flagged, turned = read_photo(photo)
    layout = sheet.short_layout()
    Image.fromarray(sheet.boxes_image(letters, flagged, layout)).save(job / "boxes.png")
    np.savez_compressed(job / "letters.npz", **{s.name: im for s, im in letters.items()})
    notes = {
        "family": family, "seed": seed, "turned": turned, "read": len(letters), "boxes": len(layout.specs()),
        "empty": [sheet.presentation_form(s) for s in layout.specs() if s not in letters],
        "flagged": [f"{sheet.presentation_form(s)} looks like {likely}" for s, (_, likely) in flagged.items()],
        "checks": check_sheet(letters),
    }
    style = variation.photo_style(letters, map_fn=_map)        # registered once, for both uses
    drawn = variation.photo_versions(letters, str(MODEL), SHOW, n=4, seed=seed, style=style)
    Image.fromarray(versions_image(drawn)).save(job / "versions.png")
    notes["seconds"] = round(time.time() - t0)
    (job / "notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")

    def build_font():
        try:
            font, _ = variation.photo_font(letters, str(MODEL), versions=3, family=family, seed=seed,
                                           style=style, map_fn=_map)
            part = job / "font.otf.part"
            font.save(part)
            part.replace(job / "font.otf")                     # whole or absent, never half written
            notes["font_seconds"] = round(time.time() - t0)
            (job / "notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
        except Exception:
            traceback.print_exc()
            (job / "font_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    if background:
        threading.Thread(target=build_font, daemon=True).start()
    else:
        build_font()
    return notes


def sentence_png(job: Path, text: str) -> bytes:
    from PIL import Image

    from ahawp_font import render

    lines = [line.strip() for line in text.splitlines() if line.strip()] or [DEFAULT_TEXT]
    image = Image.fromarray(render((job / "font.otf").read_bytes(), lines[:6], px_per_em=110))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


# ------------------------------------------------------------------ the pages
STYLE = """
body{font-family:system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;color:#222;background:#fafaf7}
h1{font-size:1.6em}h2{font-size:1.15em;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
img{max-width:100%;border:1px solid #ddd;background:#fff}
.note{color:#555;font-size:.92em}.warn{color:#a40}.err{color:#b00;font-weight:600}
button,input[type=submit]{font-size:1.05em;padding:8px 16px;margin:4px 4px 4px 0;cursor:pointer}
textarea{width:100%;font-size:1.4em;direction:rtl}
.wait{display:none;color:#555}
"""


def page(title: str, body: str, refresh: int = 0) -> str:
    meta = f"<meta http-equiv=refresh content={refresh}>" if refresh else ""
    return (f"<!doctype html><html><head><meta charset=utf-8>{meta}<meta name=viewport content='width=device-width'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>{body}</body></html>")


def home(error: str = "") -> str:
    err = f"<p class=err>{html.escape(error)}</p>" if error else ""
    sample = ("<form method=post action=/sample onsubmit=\"document.getElementById('w2').style.display='block'\">"
              "<button>Use the prepared sample sheet</button><p id=w2 class=wait>Working, about 10 seconds…</p></form>"
              if SAMPLE.is_file() else "")
    return page("Handwriting font", f"""
<h1>Your handwriting as an Arabic font</h1>
<p>Write each letter once, on its own, in the 20 boxes of the <a href=/sheet.png target=_blank>sheet</a>,
then photograph it flat with all four black squares in view.</p>
{err}
<form method=post action=/make enctype=multipart/form-data
      onsubmit="document.getElementById('w1').style.display='block'">
  <p><input type=file name=photo accept="image/*" capture=environment required></p>
  <p>Font name: <input name=family value="My Handwriting" maxlength=40></p>
  <input type=submit value="Make my font">
  <p id=w1 class=wait>Reading the sheet and drawing the letters, about 10 seconds…</p>
</form>
{sample}
<p class=note>Everything stays on this computer.</p>""")


def result(job_id: str, notes: dict, text: str, font_state: str = "ready") -> str:
    q = html.escape(text)
    warn = []
    if notes["turned"]:
        warn.append("The page was upside down; it was read turned round.")
    if notes["empty"]:
        warn.append("Empty boxes: " + " ".join(notes["empty"]) + " (the model still draws their joined forms).")
    warn += [f"Check the box: {f}." for f in notes["flagged"]]
    warn += notes.get("checks", [])
    warn_html = "".join(f"<p class=warn>{html.escape(w)}</p>" for w in warn)
    if font_state == "ready":
        font_html = f"""<h2>3. Any text in the new font</h2>
<form method=get action=/result/{job_id}>
  <textarea name=text rows=2>{q}</textarea><br><input type=submit value="Write it">
</form>
<p><img src="/job/{job_id}/sentence.png?text={html.escape(text, quote=True)}"></p>
<h2>4. The font</h2>
<p><a href=/job/{job_id}/font.otf download="{html.escape(notes['family'])}.otf"><button>Download .otf</button></a>
<form method=post action=/again/{job_id} style=display:inline
      onsubmit="document.getElementById('w3').style.display='block'">
  <button>Write it again (new versions)</button></form></p>
<p id=w3 class=wait>Drawing new versions, a few seconds…</p>
"""
    elif font_state == "error":
        font_html = "<h2>3. The font</h2><p class=err>Building the font failed; the details are in the console.</p>"
    else:
        font_html = ("<h2>3. The font</h2><p class=wait style=display:block>Building the font from these "
                     "drawings, a few more seconds… (this page updates by itself)</p>")
    return page(notes["family"], f"""
<h1>{html.escape(notes['family'])}</h1>
<p class=note>{notes['read']} of {notes['boxes']} letters read and the model's versions drawn in
{notes['seconds']} s{f", the font in {notes['font_seconds']} s" if 'font_seconds' in notes else ""}
(seed {notes['seed']}).</p>
{warn_html}
<h2>1. What the model was given: the letters as read off the photo</h2>
<img src=/job/{job_id}/boxes.png>
<h2>2. What the model drew: joined forms it was never shown, 4 versions each</h2>
<p class=note>Each row is one letter form (its label on the right). None of these forms was on the sheet:
the model draws them from the isolated letters above, in this writer's style, differently each time.</p>
<img src=/job/{job_id}/versions.png>
{font_html}<p class=note>The font has every joined form in 3 versions and cycles through them as you type,
so a repeated letter is not a copy. <a href=/>Start again</a></p>""", refresh=0 if font_state != "building" else 2)


# ------------------------------------------------------------------ the app
def create_app():
    from flask import Flask, Response, abort, redirect, request, send_file

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

    def job_dir(job_id: str) -> Path:
        if not job_id.isalnum():
            abort(404)
        job = OUT / job_id
        if not job.is_dir():
            abort(404)
        return job

    def run(job: Path, seed: int, family: str):
        try:
            make(job, seed, family, background=True)
        except TemplateError:
            return home("The four black squares were not found. Photograph the whole sheet, flat, "
                        "with all four squares in view, and try again.")
        except ValueError as e:
            return home(f"{e}. Fill in more boxes, in dark pen, and try again.")
        except Exception:                                       # a live demo must answer, not crash
            traceback.print_exc()
            return home("Something went wrong reading this photo; the details are in the console.")
        return redirect(f"/result/{job.name}")

    @app.get("/")
    def index():
        return home()

    @app.get("/sheet.png")
    def blank_sheet():
        path = ROOT / "output/ahawp/arabic_sheet_short.png"
        if not path.is_file():
            sheet.save_sheet(path, sheet.short_layout())
        return send_file(path)

    @app.post("/make")
    def make_from_upload():
        upload = request.files.get("photo")
        if not upload or not upload.filename:
            return home("Choose a photo of the filled-in sheet first.")
        job = OUT / uuid.uuid4().hex[:12]
        job.mkdir(parents=True)
        suffix = Path(upload.filename).suffix.lower() or ".jpg"
        upload.save(job / f"photo{suffix if suffix in ('.jpg', '.jpeg', '.png', '.heic', '.webp') else '.jpg'}")
        return run(job, 0, (request.form.get("family") or "My Handwriting").strip()[:40])

    @app.post("/sample")
    def make_from_sample():
        job = OUT / uuid.uuid4().hex[:12]
        job.mkdir(parents=True)
        (job / "photo.png").write_bytes(SAMPLE.read_bytes())
        return run(job, 0, "Sample writer")

    @app.post("/again/<job_id>")
    def again(job_id):
        job = job_dir(job_id)
        notes = json.loads((job / "notes.json").read_text(encoding="utf-8"))
        return run(job, notes["seed"] + 1, notes["family"])

    @app.get("/result/<job_id>")
    def show(job_id):
        job = job_dir(job_id)
        notes = json.loads((job / "notes.json").read_text(encoding="utf-8"))
        state = ("ready" if (job / "font.otf").is_file() else
                 "error" if (job / "font_error.txt").is_file() else "building")
        return result(job_id, notes, request.args.get("text", DEFAULT_TEXT), state)

    @app.get("/job/<job_id>/sentence.png")
    def sentence(job_id):
        return Response(sentence_png(job_dir(job_id), request.args.get("text", DEFAULT_TEXT)[:300]),
                        mimetype="image/png")

    @app.get("/job/<job_id>/<name>")
    def files(job_id, name):
        if name not in ("boxes.png", "versions.png", "font.otf"):
            abort(404)
        return send_file(job_dir(job_id) / name)

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from a phone on the same Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not MODEL.is_file():
        raise SystemExit(f"no model at {MODEL}")
    OUT.mkdir(parents=True, exist_ok=True)
    print("starting the worker processes...", flush=True)
    start_pool()
    print(f"open http://localhost:{args.port}" + ("  (or this computer's Wi-Fi address from a phone)"
                                                 if args.host == "0.0.0.0" else ""))
    create_app().run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
