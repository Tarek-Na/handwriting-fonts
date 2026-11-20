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
import re
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


def warm_up() -> float:
    """Run one whole sheet (the sample) through the pipeline at launch, so the first real sheet does
    not pay for loading the model code and PyTorch in this process and in the workers (cold: 27 s
    to the page and 68 s to the font; warm: about 5 s and 20 s). Returns the seconds it took."""
    import shutil

    t0 = time.time()
    if not SAMPLE.is_file():
        return 0.0
    job = OUT / "warmup"
    job.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE, job / "photo.png")
    make(job, 0, "warm-up", background=False)
    return time.time() - t0


def _map(fn, jobs):
    return POOL.map(fn, jobs, chunksize=4) if POOL is not None else map(fn, jobs)


# ------------------------------------------------------------------ the pipeline
def read_photo(photo: Path):
    """(letters, flagged, turned) off a photo of the short sheet."""
    checker = CHECKER if CHECKER.is_file() else None
    return sheet.read_upright(sheet.load_photo(photo), checker, sheet.short_layout())


def flag_text(spec, likely: str) -> str:
    """A box the form classifier doubts, said in letters, not glyph names."""
    from ahawp_forms import specs_by_name

    other = specs_by_name().get(likely)
    looks = f" It looks a little like {sheet.presentation_form(other)}." if other is not None else ""
    return f"Check the {sheet.presentation_form(spec)} box: is it that letter, written on its own?{looks}"


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


def save_versions(job: Path, drawn: dict[str, list[np.ndarray]], margin: int = 6) -> None:
    """Each drawn version as ``v<form><version>.png``: dark ink on white, cut to the ink."""
    from PIL import Image

    for i, ims in enumerate(drawn.values()):
        for k, im in enumerate(ims):
            m = np.asarray(im, np.float32) > 0.5
            rows, cols = np.nonzero(m)
            if rows.size:
                m = m[max(rows.min() - margin, 0):rows.max() + margin + 1, max(cols.min() - margin, 0):cols.max() + margin + 1]
            Image.fromarray(np.where(m, 24, 255).astype(np.uint8)).save(job / f"v{i}{k}.png", optimize=False)


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
        "flagged": [flag_text(s, likely) for s, (_, likely) in flagged.items()],
        "checks": check_sheet(letters),
    }
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as both:                         # the two models' fittings side by side
        style_job = both.submit(variation.photo_style, letters, map_fn=_map)   # registered once, for both uses
        built_job = both.submit(variation.construction_drawings, letters, n=4, seed=seed, map_fn=_map)
        style, built = style_job.result(), built_job.result()
    drawn = variation.photo_versions(letters, str(MODEL), SHOW, n=4, seed=seed, style=style, constructions=built)
    Image.fromarray(versions_image(drawn)).save(job / "versions.png")
    save_versions(job, drawn)
    notes["forms"], notes["versions"] = list(drawn), max(len(v) for v in drawn.values())
    notes["seconds"] = round(time.time() - t0)
    (job / "notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")

    def build_font():
        try:
            font, _ = variation.photo_font(letters, str(MODEL), versions=3, family=family, seed=seed,
                                           style=style, map_fn=_map, constructions=built)
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
BRAND = "Khatti"
STATIC = Path(__file__).resolve().parent / "demo_static"
#: Sample sentences for the live typing chips: every letter in them is in a sheet font.
CHIPS = ["سلام عليكم", "مرحبا بكم في الجامعة", "كلب جميل", "شمس وقمر"]
LETTER_NAMES = {"uni0628": ("beh", "باء"), "uni0633": ("seen", "سين"), "uni0639": ("ain", "عين"),
                "uni0647": ("heh", "هاء"), "uni062C": ("jeem", "جيم"), "uni0643": ("kaf", "كاف"),
                "uni0645": ("meem", "ميم")}
POSITIONS = {"init": ("initial", "في أول الكلمة"), "medi": ("medial", "في وسط الكلمة"),
             "fina": ("final", "في آخر الكلمة")}
#: Headline results, test split (16 held-out writers, scored once), from
#: reports/Arabic variation model November 2026.md, section 4.
RESULTS = [("Writer's own other copies (ceiling)", "0.713", "0.692", "+0.208", "ceiling"),
           ("Khatti: conditional VAE + style encoder", "0.605", "0.518", "+0.017", "ours"),
           ("Gaussian + ridge on the writer's letters (no network)", "0.583", "0.506", "−0.000", ""),
           ("Gaussian shape model (no network)", "0.572", "0.506", "−0.004", "")]


def _icon(path: str, extra: str = "") -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {extra}>{path}</svg>')


ICON = {
    "print": _icon('<path d="M7 9V3h10v6"/><rect x="3" y="9" width="18" height="8" rx="2"/><path d="M7 14h10v7H7z"/>'),
    "pen": _icon('<path d="M4 20l4-1 11-11-3-3L5 16z"/><path d="M14 6l3 3"/>'),
    "camera": _icon('<path d="M4 7h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/>'),
    "upload": _icon('<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 16v4h16v-4"/>'),
    "download": _icon('<path d="M12 4v12"/><path d="M7 11l5 5 5-5"/><path d="M4 20h16"/>'),
    "refresh": _icon('<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>'),
    "info": _icon('<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>'),
    "alert": _icon('<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4"/><path d="M12 17h.01"/>'),
    "spark": _icon('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6"/>'),
}
STAGES = ["Reading your letters", "Drawing the joined forms", "Building your font"]


def stages_html(list_id: str, css: str = "stages") -> str:
    items = "".join(f'<li class=stage><span class=dot>{i + 1}</span><span>{name}</span></li>'
                    for i, name in enumerate(STAGES))
    return f'<ol class="{css}" id={list_id}>{items}</ol>'


def page(title: str, body: str, current: str = "") -> str:
    def link(href, label, key):
        return f'<a href={href}{" aria-current=page" if current == key else ""}>{label}</a>'

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · {BRAND}</title><link rel=stylesheet href=/static/khatti.css></head><body>
<a class=skip href=#main>Skip to content</a>
<nav class=nav aria-label=Main><div class=logo aria-hidden=true><span class=ar>خ</span></div>
<a class=brand href=/>{BRAND}</a>
<div class=nav-links>{link("/", "Make a font", "home")}{link("/how", "How it works", "how")}
<span class=badge-local>Runs on this computer</span></div></nav>
<main id=main>{body}</main>
<footer>{BRAND}: an Arabic font in your handwriting, from 20 letters. The joined forms are drawn by a neural
network. Photos and fonts stay on this computer.</footer>
<div class=overlay id=overlay role=dialog aria-modal=true aria-labelledby=overlay-title>
<div class=progress-card><h2 id=overlay-title>Making your font</h2>{stages_html("overlay-stages")}
<p class=small style="margin:16px 0 0">About 4 seconds, then the font is finished on the next page.</p></div></div>
<script src=/static/khatti.js defer></script></body></html>"""


def home(error: dict | None = None) -> str:
    err = ""
    if error:
        err = (f'<div class="card error-card" role=alert style="margin-bottom:22px">'
               f'<h2>{html.escape(error["title"])}</h2><p style=margin:0>{html.escape(error["reason"])}</p>'
               f'<div class=fix><b>What to do:</b> {html.escape(error["fix"])}</div></div>')
    sample = ("" if not SAMPLE.is_file() else
              f'<div class=or>or</div><form method=post action=/sample data-busy="Trying the sample sheet">'
              f'<button class="btn btn-secondary btn-block">{ICON["spark"]}Try it with a sample sheet</button></form>')
    return page("Your handwriting as an Arabic font", f"""
<section class=hero>
  <h1>Write 20 letters. Get an Arabic font in your handwriting.</h1>
  <p>A neural network learns your style from 20 isolated letters and draws every joined form you never
  wrote, a little differently each time, as a hand does.</p>
</section>
{err}
<div class=start>
  <div class="card guide">
    <h2>Three steps</h2>
    <ol class=steps>
      <li class=step><div class=n>1</div><div><h3>Print the sheet</h3>
        <p>One A4 page with 20 boxes, one per letter. Print it in colour.</p></div>{ICON["print"]}</li>
      <li class=step><div class=n>2</div><div><h3>Write each letter once</h3>
        <p>On its own, resting on the red line, in dark pen, without touching the box edges.</p></div>{ICON["pen"]}</li>
      <li class=step><div class=n>3</div><div><h3>Photograph and upload</h3>
        <p>Flat, in good light, with all four black squares in view.</p></div>{ICON["camera"]}</li>
    </ol>
    <a class="btn btn-secondary btn-block" href=/sheet.png target=_blank rel=noopener>{ICON["print"]}Open the sheet to print</a>
  </div>
  <div class=card>
    <h2>Make your font</h2>
    <form method=post action=/make enctype=multipart/form-data data-busy="Making your font">
      <label class=drop id=drop style=margin-top:14px>
        <input type=file name=photo id=photo accept="image/*" required aria-describedby=chosen>
        <img id=thumb alt="">
        <span class=hint>{ICON["upload"]}<span><strong>Drop the photo here</strong> or choose one</span>
        <span class=small>On a phone, this opens the camera or your photos.</span></span>
      </label>
      <p class="small sr-only" id=chosen aria-live=polite></p>
      <label class=field for=family>Font name</label>
      <input type=text id=family name=family value="My Handwriting" maxlength=40>
      <button class="btn btn-primary btn-block" id=go disabled style=margin-top:16px>Make my font</button>
    </form>
    {sample}
  </div>
</div>""", current="home")


def _form_label(name: str) -> tuple[str, str, str, str]:
    """(shaped glyph, English, Arabic, position) for a joined form's name, e.g. uni0628.medi."""
    from ahawp_forms import specs_by_name

    base, pos = name.split(".")
    en, ar = LETTER_NAMES.get(base, (base, ""))
    pos_en, pos_ar = POSITIONS.get(pos, (pos, ""))
    return sheet.presentation_form(specs_by_name()[name]), f"{en} · {pos_en}", f"{ar} {pos_ar}", pos_en


def result(job_id: str, notes: dict, text: str, font_state: str = "ready") -> str:
    tips = []
    if notes["turned"]:
        tips.append(("info", "The page was upside down", "It was read turned round; nothing to do."))
    if notes["empty"]:
        tips.append(("info", "Some boxes were empty",
                     "Empty: " + " ".join(notes["empty"]) + ". The model still draws their joined forms."))
    tips += [("alert", "A box to check", f) for f in notes["flagged"]]
    tips += [("alert", "Tip for a better font", c) for c in notes.get("checks", [])]
    tips_html = "".join(f'<div class=notice role=note>{ICON[kind]}<div><b>{html.escape(title)}</b>'
                        f'{html.escape(body)}</div></div>' for kind, title, body in tips)
    family = html.escape(notes["family"])
    forms = notes.get("forms", [])
    cards = []
    for i, name in enumerate(forms):
        glyph, en, ar, pos = _form_label(name)
        imgs = "".join(f'<img src="/job/{job_id}/v{i}{k}.png?v={notes["seed"]}" alt="Version {k + 1} of {en}">'
                       for k in range(notes.get("versions", 4)))
        cards.append(f'<div class=form role=group aria-label="{html.escape(en)}, 4 versions">'
                     f'<div class=versions>{imgs}</div>'
                     f'<div class=label><span class="glyph ar" lang=ar>{glyph}</span>'
                     f'<span class=name>{html.escape(en)}</span><span class="name-ar ar" lang=ar>{ar}</span></div></div>')
    drawings = (f'<div class=forms>{"".join(cards)}</div>' if cards else
                f'<div class=paper><img src=/job/{job_id}/versions.png alt="The network\'s drawings"></div>')
    chips = "".join(f'<button type=button class=chip lang=ar aria-pressed=false data-text="{html.escape(c)}">'
                    f'{html.escape(c)}</button>' for c in CHIPS)
    return page(notes["family"], f"""
<div id=result data-job={job_id} data-seed={notes["seed"]} data-font={font_state}
     data-text="{html.escape(DEFAULT_TEXT)}">
<div class=result-head>
  <div><p class=eyebrow>Your font</p><h1>{family}</h1></div>
  <div class=right>
    <a class="btn btn-primary when-ready {'hidden' if font_state != 'ready' else ''}"
       href="/job/{job_id}/font.otf?v={notes['seed']}" download="{family}.otf">{ICON["download"]}Download .otf</a>
    <a class="btn btn-secondary" href=/>New sheet</a>
  </div>
  <p class=meta>{notes['read']} of {notes['boxes']} letters read · style learned and joined forms drawn in
  {notes['seconds']} s · <span id=status-line aria-live=polite>{"Your font is ready." if font_state == "ready"
  else "Building your font…"}</span></p>
</div>
<div class=stack>
  <section class=card aria-labelledby=drawn-title>
    <div class=section-head><h2 id=drawn-title>What the network drew</h2>
      <span class=tag>{ICON["spark"].replace('aria-hidden="true"', 'aria-hidden="true" width="14" height="14"')}
      Neural network output</span>
      <p class=lead>Drawn by the network. You never wrote these forms: it learned your style from the 20
      isolated letters and drew each joined form four times, never quite the same.</p></div>
    {drawings}
  </section>

  <section id=building class="card {'' if font_state == 'building' else 'hidden'}" aria-live=polite>
    <div class=section-head><h2>Building your font</h2>
      <p class=lead>Turning every drawing into an outline and joining them into an OpenType font,
      about 10 more seconds.</p></div>
    {stages_html("font-stages", "stages inline-stages")}
  </section>
  <section id=failed class="card {'' if font_state == 'error' else 'hidden'}">
    <div class="notice err" role=alert>{ICON["alert"]}<div><b>The font could not be built</b>
    The drawings above are fine; building the font failed. The details are in the console.</div></div>
  </section>

  <section class="typing when-ready {'' if font_state == 'ready' else 'hidden'}" aria-labelledby=type-title>
    <div class=card>
      <div class=section-head><h2 id=type-title>Type anything</h2>
        <p class=lead>Written in your new font as you type, by the font itself.</p></div>
      <div class=chips role=group aria-label="Sample sentences">{chips}</div>
      <label class=sr-only for=text>Text to write in your font</label>
      <textarea id=text rows=2 dir=rtl lang=ar spellcheck=false>{html.escape(text)}</textarea>
      <div class="paper live" id=live lang=ar dir=rtl aria-live=polite></div>
    </div>
    <div class=card>
      <div class=section-head><h2>Your font</h2>
        <p class=lead>Every joined form in 3 versions; the font cycles through them, so a repeated letter
        is never a copy.</p></div>
      <div class=download>
        <a class="btn btn-primary btn-block" href="/job/{job_id}/font.otf?v={notes['seed']}"
           download="{family}.otf">{ICON["download"]}Download {family}.otf</a>
        <form method=post action=/again/{job_id} data-busy="Drawing new versions">
          <button class="btn btn-secondary btn-block">{ICON["refresh"]}Draw new versions</button></form>
        <p class=small style=margin:0>To install: open the .otf file and choose Install. It then appears in
        Word and other apps as “{family}”.</p>
      </div>
    </div>
  </section>

  {f'<div class=notices style=margin:0 aria-label="Notes on the photo">{tips_html}</div>' if tips else ''}
  <details class=check>
    <summary>{ICON["info"].replace('aria-hidden="true"', 'aria-hidden="true" width="20" height="20"')}
    Check what was read from the photo</summary>
    <div class=body><p class=small style="margin:0 0 12px">The 20 letters the model was given. A red frame
    marks a box worth checking.</p>
    <div class=paper><img src=/job/{job_id}/boxes.png alt="The letters as read from the photo"></div></div>
  </details>
</div>
</div>""")


def how() -> str:
    rows = "".join(f'<tr class="{cls}"><td>{html.escape(m)}</td><td>{r}</td><td>{c}</td><td>{s}</td></tr>'
                   for m, r, c, s, cls in RESULTS)
    return page("How it works", f"""
<section class=hero>
  <p class=eyebrow>How it works</p>
  <h1>From 20 letters to a joining Arabic font.</h1>
  <p>Arabic letters change shape when they join. Khatti asks only for each letter on its own, and a
  generative network draws the joined forms in the writer's style.</p>
</section>
<div class=stack>
  <section class=card aria-labelledby=pipe-title>
    <div class=section-head><h2 id=pipe-title>The pipeline</h2></div>
    <svg class=pipeline viewBox="0 0 1040 200" role=img
         aria-label="Isolated letters, to the style encoder, to a conditional VAE over skeleton deformations, to joined forms in several versions, to an OpenType font">
      <defs><marker id=arrowhead viewBox="0 0 10 10" refX=9 refY=5 markerWidth=7 markerHeight=7 orient=auto>
        <path d="M0 0L10 5L0 10z" fill="currentColor" style="fill:var(--muted)"/></marker></defs>
      <rect x="196" y="20" width="420" height="160" rx="16" fill="none" stroke="var(--accent)" stroke-dasharray="5 5"/>
      <text class=nnlabel x="210" y="40">NEURAL NETWORK</text>
      <rect class=box x="10" y="60" width="150" height="90" rx="12"/>
      <text class=t x="85" y="98" text-anchor=middle>20 letters</text>
      <text class=s x="85" y="120" text-anchor=middle>isolated, from the photo</text>
      <rect class="box nn" x="216" y="60" width="170" height="90" rx="12"/>
      <text class=t x="301" y="98" text-anchor=middle>Style encoder</text>
      <text class=s x="301" y="120" text-anchor=middle>attention over the letters</text>
      <rect class="box nn" x="426" y="60" width="174" height="90" rx="12"/>
      <text class=t x="513" y="92" text-anchor=middle>Conditional VAE</text>
      <text class=s x="513" y="114" text-anchor=middle>over skeleton</text>
      <text class=s x="513" y="131" text-anchor=middle>deformations</text>
      <rect class=box x="650" y="60" width="170" height="90" rx="12"/>
      <text class=t x="735" y="98" text-anchor=middle>Joined forms</text>
      <text class=s x="735" y="120" text-anchor=middle>several versions each</text>
      <rect class=box x="860" y="60" width="170" height="90" rx="12"/>
      <text class=t x="945" y="98" text-anchor=middle>OpenType font</text>
      <text class=s x="945" y="120" text-anchor=middle>.otf that joins</text>
      <path class=arrow d="M160 105H212"/><path class=arrow d="M386 105H422"/>
      <path class=arrow d="M600 105H646"/><path class=arrow d="M820 105H856"/>
    </svg>
    <ul class=stages-list>
      <li><b>Read the sheet</b>Four squares fix the photo; each box gives one letter.</li>
      <li><b>Style encoder</b>Reads the letters as a set and pools them into one style vector.</li>
      <li><b>Conditional VAE</b>Each form is a template skeleton plus moves; the network samples moves in
      this style.</li>
      <li><b>Joined forms</b>Several versions per form, because nobody writes a letter the same way twice.</li>
      <li><b>OpenType font</b>Positional forms, joins, and alternates the font cycles through.</li>
    </ul>
    <p class=small style="margin:14px 0 0">A design choice in this demo: medial ain, medial kaf and
    initial yeh are drawn from another way of building the letter that the multi-template version of
    the network learned (a flat-loop ain, an upright kaf, a short-toothed yeh), still in the writer's
    style, because the training writers' typical construction reads poorly as type. The results below
    are for the network as evaluated.</p>
  </section>
  <section class=card aria-labelledby=res-title>
    <div class=section-head><h2 id=res-title>Results on writers the model never saw</h2>
      <p class=lead>AHAWP dataset, 16 held-out test writers, scored once. Each method draws 5 versions of
      every joined form from the writer's isolated letters; they are compared with the writer's real
      copies (centreline F1, 0 to 1).</p></div>
    <div style=overflow-x:auto><table class=results>
      <thead><tr><th scope=col>Method</th><th scope=col>Realism</th><th scope=col>Coverage</th>
      <th scope=col>Specificity</th></tr></thead><tbody>{rows}</tbody></table></div>
    <div class=defs>
      <div><b>Realism</b>How closely each drawn version matches one of the writer's real copies.</div>
      <div><b>Coverage</b>How well the versions span the ways the writer really writes the form.</div>
      <div><b>Specificity</b>How much better they match this writer than another one. Zero means the
      method cannot tell writers apart.</div>
    </div>
    <p class=small style="margin:16px 0 0">The network beats both non-network baselines on all three
    measures (paired 95% intervals above zero). Given another writer's letters, its specificity flips sign,
    so the style encoder really reads who wrote them.</p>
  </section>
  <p><a class="btn btn-primary" href=/>Make a font</a></p>
</div>""", current="how")


ERRORS = {
    "squares": {"title": "We couldn't find the sheet in this photo",
                "reason": "The four black squares in the corners of the sheet were not found, so the page could "
                          "not be straightened.",
                "fix": "Photograph the whole sheet, flat, in good light, with all four black squares in view, "
                       "and upload it again."},
    "letters": {"title": "Not enough letters could be read",
                "reason": "{detail}.",
                "fix": "Fill in at least 3 boxes, ideally all 20, in dark pen, then photograph the sheet again."},
    "nophoto": {"title": "No photo chosen",
                "reason": "The form was sent without a photo.",
                "fix": "Choose or drop a photo of your filled-in sheet first."},
    "other": {"title": "Something went wrong with this photo",
              "reason": "The photo could not be read; the details are in the console.",
              "fix": "Try another photo of the sheet, or use the sample sheet to check the demo itself."},
}


def error_page(kind: str, detail: str = "") -> str:
    e = dict(ERRORS[kind])
    e["reason"] = e["reason"].format(detail=detail)
    return home(e)


# ------------------------------------------------------------------ the app
def create_app():
    from flask import Flask, Response, abort, redirect, request, send_file, send_from_directory

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
            return error_page("squares")
        except ValueError as e:
            return error_page("letters", str(e)[:1].upper() + str(e)[1:])
        except Exception:                                       # a live demo must answer, not crash
            traceback.print_exc()
            return error_page("other")
        return redirect(f"/result/{job.name}")

    @app.get("/")
    def index():
        return home()

    @app.get("/how")
    def how_it_works():
        return how()

    @app.get("/static/<name>")
    def static_file(name):
        if name not in ("khatti.css", "khatti.js"):
            abort(404)
        return send_from_directory(STATIC, name, max_age=0)

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
            return error_page("nophoto")
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

    @app.get("/job/<job_id>/status")
    def status(job_id):
        job = job_dir(job_id)
        state = ("ready" if (job / "font.otf").is_file() else
                 "error" if (job / "font_error.txt").is_file() else "building")
        return {"font": state}

    @app.get("/job/<job_id>/sentence.png")
    def sentence(job_id):
        return Response(sentence_png(job_dir(job_id), request.args.get("text", DEFAULT_TEXT)[:300]),
                        mimetype="image/png")

    @app.get("/job/<job_id>/<name>")
    def files(job_id, name):
        if name not in ("boxes.png", "versions.png", "font.otf") and not re.fullmatch(r"v\d\d\.png", name):
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
    print("warming up on the sample sheet (about 30 s, once)...", flush=True)
    try:
        print(f"  ready ({warm_up():.0f} s)", flush=True)
    except Exception:                                   # the demo still runs, only the first sheet is slower
        traceback.print_exc()
    print(f"open http://localhost:{args.port}" + ("  (or this computer's Wi-Fi address from a phone)"
                                                 if args.host == "0.0.0.0" else ""))
    create_app().run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
