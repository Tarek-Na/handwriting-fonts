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
BRAND = "Khatti"
STYLE = """
:root{--bg:#f6f5f1;--surface:#fff;--ink:#1c1c1e;--muted:#6b6b70;--line:#e7e3da;--accent:#0f6e5c;
--accent-ink:#fff;--accent-soft:#e6f2ee;--warn-bg:#fff5e1;--warn:#7a4b00;--err-bg:#fdecea;--err:#a8231a;
--paper:#fff;--shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.06);
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
--arabic:"Segoe UI","Noto Naskh Arabic","Geeza Pro","Arial",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#121213;--surface:#1c1c1e;--ink:#f2f2f3;--muted:#a1a1a8;
--line:#2d2d30;--accent:#3cc4a0;--accent-ink:#06261f;--accent-soft:#163029;--warn-bg:#33270f;--warn:#f1c27d;
--err-bg:#3a1714;--err:#ff9b91;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35)}}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.nav{display:flex;align-items:center;gap:12px;max-width:1080px;margin:0 auto;padding:18px 20px}
.logo{width:36px;height:36px;border-radius:10px;background:var(--accent);color:var(--accent-ink);display:grid;
place-items:center;font:600 22px/1 var(--arabic)}
.brand{font-weight:650;font-size:1.15rem;letter-spacing:-.01em;text-decoration:none;color:var(--ink)}
.pill{margin-left:auto;font-size:.8rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:4px 10px}
main{max-width:1080px;margin:0 auto;padding:8px 20px 40px}
.hero{padding:28px 0 22px}.hero h1{font-size:clamp(1.9rem,4vw,2.8rem);line-height:1.12;letter-spacing:-.025em;margin:0 0 12px}
.hero p{font-size:1.1rem;color:var(--muted);max-width:640px;margin:0}
.grid{display:grid;grid-template-columns:1fr 1.15fr;gap:20px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:22px}
.card h2{font-size:1.05rem;margin:0 0 4px;letter-spacing:-.01em}.card .sub{color:var(--muted);margin:0 0 16px;font-size:.95rem}
.steps{list-style:none;margin:0;padding:0}.steps li{display:flex;gap:14px;padding:12px 0;border-top:1px solid var(--line)}
.steps li:first-child{border-top:0}.num{flex:none;width:28px;height:28px;border-radius:50%;background:var(--accent-soft);
color:var(--accent);display:grid;place-items:center;font-weight:650;font-size:.9rem}
.steps b{display:block}.steps span{color:var(--muted);font-size:.93rem}
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;text-align:center;
min-height:190px;border:2px dashed var(--line);border-radius:14px;padding:18px;cursor:pointer;transition:.15s;color:var(--muted)}
.drop:hover,.drop.over{border-color:var(--accent);background:var(--accent-soft);color:var(--ink)}
.drop img{max-height:220px;max-width:100%;border-radius:8px;display:none}.drop.has img{display:block}
.drop.has .hint{display:none}.drop svg{width:34px;height:34px;color:var(--accent)}
input[type=file]{display:none}
label.field{display:block;margin:16px 0 6px;font-size:.88rem;color:var(--muted)}
input[type=text]{width:100%;font:inherit;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
background:var(--bg);color:var(--ink)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;font:600 1rem var(--sans);border-radius:11px;
padding:12px 18px;border:1px solid transparent;cursor:pointer;text-decoration:none;transition:.15s}
.btn.primary{background:var(--accent);color:var(--accent-ink);width:100%;margin-top:16px}
.btn.primary:disabled{opacity:.45;cursor:not-allowed}.btn.primary:not(:disabled):hover{filter:brightness(1.07)}
.btn.ghost{background:transparent;color:var(--ink);border-color:var(--line)}.btn.ghost:hover{border-color:var(--accent);color:var(--accent)}
.or{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:.85rem;margin:16px 0}
.or:before,.or:after{content:"";flex:1;height:1px;background:var(--line)}
.wide{width:100%}
.banner{border-radius:12px;padding:12px 14px;margin:0 0 16px;font-size:.95rem}
.banner.err{background:var(--err-bg);color:var(--err)}.banner.warn{background:var(--warn-bg);color:var(--warn)}
.banner ul{margin:6px 0 0;padding-left:18px}
.head{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px;margin:18px 0 18px}
.head h1{margin:0;font-size:clamp(1.6rem,3.4vw,2.2rem);letter-spacing:-.02em}.head .meta{color:var(--muted);width:100%}
.head .actions{margin-left:auto}
.stack{display:grid;gap:20px}
.tag{display:inline-block;font-size:.75rem;font-weight:600;color:var(--accent);background:var(--accent-soft);
border-radius:999px;padding:3px 9px;margin-left:8px;vertical-align:middle}
.paper{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto}
.paper img{display:block;max-width:100%;height:auto;margin:0 auto}
textarea{width:100%;font:1.6rem/1.5 var(--arabic);direction:rtl;padding:12px 14px;border-radius:12px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);resize:vertical}
.preview{min-height:120px;display:grid;place-items:center}.preview img{transition:opacity .2s}
.split{display:grid;grid-template-columns:1.6fr 1fr;gap:20px}@media(max-width:820px){.split{grid-template-columns:1fr}}
.download{display:flex;flex-direction:column;gap:10px}.download .btn{width:100%}
.small{font-size:.86rem;color:var(--muted)}
.building{display:flex;align-items:center;gap:14px;color:var(--muted)}
.spin{width:22px;height:22px;border-radius:50%;border:3px solid var(--accent-soft);border-top-color:var(--accent);
animation:spin .8s linear infinite;flex:none}@keyframes spin{to{transform:rotate(360deg)}}
.overlay{position:fixed;inset:0;background:rgba(20,20,22,.55);backdrop-filter:blur(3px);display:none;place-items:center;z-index:9}
.overlay.on{display:grid}.overlay .box{background:var(--surface);border-radius:16px;padding:26px 30px;min-width:300px;
box-shadow:var(--shadow);text-align:center}.overlay .spin{width:40px;height:40px;margin:0 auto 14px;border-width:4px}
.overlay .stage{font-weight:600}.overlay .small{margin-top:4px}
.hidden{display:none!important}
footer{max-width:1080px;margin:0 auto;padding:0 20px 30px;color:var(--muted);font-size:.85rem}
.grid>*,.split>*,.stack>*{min-width:0}
@media(max-width:560px){.pill{display:none}main{padding:4px 16px 32px}.nav{padding:14px 16px}
.card{padding:18px;border-radius:14px}.hero{padding:16px 0 18px}footer{padding:0 16px 24px}.head .actions{margin-left:0}}
"""

LOGO = "<div class=logo aria-hidden=true>خ</div>"
UPLOAD_ICON = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
               'stroke-linejoin="round"><path d="M4 7h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/></svg>')
OVERLAY = """<div class=overlay id=overlay><div class=box><div class=spin></div>
<div class=stage id=stage>Straightening the photo</div><div class=small>About 5 seconds</div></div></div>"""
SCRIPT = """<script>
const stages=["Straightening the photo","Reading your 20 letters","Learning your style","Drawing the joined forms"];
function busy(first){const o=document.getElementById('overlay');if(!o)return;o.classList.add('on');
let i=0;const s=document.getElementById('stage');if(first)s.textContent=first;
setInterval(()=>{i=Math.min(i+1,stages.length-1);s.textContent=stages[i]},1300)}
document.querySelectorAll('form[data-busy]').forEach(f=>f.addEventListener('submit',()=>busy(f.dataset.busy)));
</script>"""

HOME_SCRIPT = """<script>
const drop=document.getElementById('drop'),input=document.getElementById('photo'),thumb=document.getElementById('thumb'),
go=document.getElementById('go');
function show(file){if(!file)return;const r=new FileReader();r.onload=e=>{thumb.src=e.target.result;drop.classList.add('has')};
r.readAsDataURL(file);go.disabled=false}
input.addEventListener('change',()=>show(input.files[0]));
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',e=>{if(e.dataTransfer.files.length){input.files=e.dataTransfer.files;show(input.files[0])}});
</script>"""

RESULT_SCRIPT = """<script>
const job="__JOB__",fallback="__TEXT__",text=document.getElementById('text'),img=document.getElementById('sentence');
let timer=null;
function draw(){const t=text.value.trim()||fallback;img.style.opacity=.4;
const u="/job/"+job+"/sentence.png?text="+encodeURIComponent(t)+"&n="+Date.now();
const next=new Image();next.onload=()=>{img.src=u;img.style.opacity=1};next.src=u}
text.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(draw,350)});
function poll(){fetch("/job/"+job+"/status").then(r=>r.json()).then(s=>{
if(s.font==="ready"){document.getElementById('building').classList.add('hidden');
document.getElementById('ready').classList.remove('hidden');draw()}
else if(s.font==="error"){document.getElementById('building').classList.add('hidden');
document.getElementById('failed').classList.remove('hidden')}
else setTimeout(poll,800)}).catch(()=>setTimeout(poll,1500))}
__START__;
</script>"""


def page(title: str, body: str, script: str = "") -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)} · {BRAND}</title><style>{STYLE}</style></head><body>"
            f"<nav class=nav>{LOGO}<a class=brand href=/>{BRAND}</a>"
            f"<span class=pill>Live demo · runs on this computer</span></nav>"
            f"<main>{body}</main>"
            f"<footer>{BRAND}: your handwriting as an Arabic font, drawn by a neural network from 20 letters. "
            f"Photos and fonts stay on this computer.</footer>{OVERLAY}{SCRIPT}{script}</body></html>")


def home(error: str = "") -> str:
    err = f"<div class='banner err'>{html.escape(error)}</div>" if error else ""
    sample = ("<div class=or>or</div><form method=post action=/sample data-busy='Opening the sample sheet'>"
              "<button class='btn ghost wide'>Try it with a sample sheet</button></form>" if SAMPLE.is_file() else "")
    return page("Your handwriting as a font", f"""
<section class=hero>
  <h1>Your handwriting,<br>as an Arabic font.</h1>
  <p>Write 20 letters. A neural network draws every joined form in your style, a little differently
  each time, as a hand does, and you get a real font to type with.</p>
</section>
<div class=grid>
  <div class=card>
    <h2>How it works</h2><p class=sub>Two minutes, one sheet of paper.</p>
    <ol class=steps>
      <li><div class=num>1</div><div><b>Print the sheet</b><span>One page with 20 boxes, one per letter.
        <a href=/sheet.png target=_blank>Open the sheet</a></span></div></li>
      <li><div class=num>2</div><div><b>Write each letter once</b><span>On its own, resting on the red line,
        in dark pen. Fill the box without touching its edges.</span></div></li>
      <li><div class=num>3</div><div><b>Take a photo and upload it</b><span>Flat, in good light, with all four
        black squares in view.</span></div></li>
    </ol>
  </div>
  <div class=card>
    <h2>Make your font</h2><p class=sub>Upload the photo of your filled-in sheet.</p>
    {err}
    <form method=post action=/make enctype=multipart/form-data data-busy='Straightening the photo'>
      <label class=drop id=drop>
        <input type=file name=photo id=photo accept="image/*" required>
        <img id=thumb alt="">
        <div class=hint>{UPLOAD_ICON}<div><b>Drop a photo here</b> or click to choose one</div>
        <div class=small>JPG or PNG, from a phone is fine</div></div>
      </label>
      <label class=field for=family>Font name</label>
      <input type=text id=family name=family value="My Handwriting" maxlength=40>
      <button class='btn primary' id=go disabled>Make my font</button>
    </form>
    {sample}
  </div>
</div>""", HOME_SCRIPT)


def result(job_id: str, notes: dict, text: str, font_state: str = "ready") -> str:
    advice = []
    if notes["turned"]:
        advice.append("The page was upside down; it was read turned round.")
    if notes["empty"]:
        advice.append("Empty boxes: " + " ".join(notes["empty"]) + ". The model still draws their joined forms.")
    advice += notes["flagged"]
    advice += notes.get("checks", [])
    advice_html = ("<div class='banner warn'><b>Tips for a better font</b><ul>"
                   + "".join(f"<li>{html.escape(a)}</li>" for a in advice) + "</ul></div>") if advice else ""
    family = html.escape(notes["family"])
    ready = font_state == "ready"
    hide = lambda on: "" if on else "hidden"  # noqa: E731
    script = (RESULT_SCRIPT.replace("__JOB__", job_id).replace("__TEXT__", DEFAULT_TEXT)
              .replace("__START__", "draw()" if ready else "poll()"))
    return page(notes["family"], f"""
<div class=head>
  <h1>{family}</h1>
  <div class=actions><a class='btn ghost' href=/>New sheet</a></div>
  <div class=meta>{notes['read']} of {notes['boxes']} letters read · style learned and joined forms drawn in
  {notes['seconds']} s</div>
</div>
{advice_html}
<div class=stack>
  <div class=card>
    <h2>Your letters</h2>
    <p class=sub>What the model was given: the 20 letters, read straight off the photo.</p>
    <div class=paper><img src=/job/{job_id}/boxes.png alt="The letters as read"></div>
  </div>
  <div class=card>
    <h2>What the network drew <span class=tag>Neural network output</span></h2>
    <p class=sub>Joined forms that were not on the sheet, drawn from your isolated letters in your style.
    Each row is one form, its label on the right; each is drawn four times, never quite the same.</p>
    <div class=paper><img src=/job/{job_id}/versions.png alt="The model's versions of joined forms"></div>
  </div>
  <div id=building class="card {hide(font_state == 'building')}">
    <div class=building><div class=spin></div><div><b>Building your font</b><br>
    <span class=small>Turning every drawing into an outline, about 10 more seconds.</span></div></div>
  </div>
  <div id=failed class="card {hide(font_state == 'error')}">
    <div class='banner err' style=margin:0>Building the font failed. The details are in the console.</div>
  </div>
  <div id=ready class="split {hide(ready)}">
    <div class=card>
      <h2>Type anything</h2><p class=sub>Set in your new font as you type.</p>
      <textarea id=text rows=2 spellcheck=false>{html.escape(text)}</textarea>
      <div class="paper preview" style=margin-top:14px><img id=sentence alt="Your text in your handwriting"></div>
    </div>
    <div class=card>
      <h2>Your font</h2><p class=sub>Every joined form in 3 versions; the font cycles through them,
      so a repeated letter is never a copy.</p>
      <div class=download>
        <a class='btn primary' style=margin:0 href=/job/{job_id}/font.otf download="{family}.otf">Download {family}.otf</a>
        <form method=post action=/again/{job_id} data-busy='Drawing new versions'>
          <button class='btn ghost wide'>Draw new versions</button></form>
        <p class=small>To install: open the .otf file and choose Install. It then appears in Word and
        other apps as “{family}”.</p>
      </div>
    </div>
  </div>
</div>""", script)


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
