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


# ---------------------------------------------------------------- the pages (Flask test client, no model)
import io  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

flask = pytest.importorskip("flask")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import demo_app

    monkeypatch.setattr(demo_app, "OUT", tmp_path)
    return demo_app.create_app().test_client()


def _fake_job(root, font=True) -> str:
    """A finished job as make() leaves it, with blank images: no handwriting, no model."""
    from PIL import Image

    job = root / "abc123"
    job.mkdir()
    notes = {"family": "Test Hand", "seed": 0, "turned": False, "read": 18, "boxes": 20, "empty": ["ﻉ", "ﻩ"],
             "flagged": [], "checks": ["Only 18 of 20 letters were read."], "seconds": 4,
             "forms": ["uni0628.init", "uni0633.medi"], "versions": 4}
    (job / "notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
    for name in ["boxes.png", "versions.png"] + [f"v{i}{k}.png" for i in range(2) for k in range(4)]:
        Image.new("L", (40, 20), 255).save(job / name)
    if font:
        (job / "font.otf").write_bytes(b"OTTO")
    return job.name


def test_the_stylesheet_and_script_are_served_from_the_repo_and_nothing_else_is(client):
    assert client.get("/static/khatti.css").status_code == 200
    js = client.get("/static/khatti.js")
    assert js.status_code == 200 and b"FontFace" in js.data
    for bad in ("/static/demo_app.py", "/static/..%2Fdemo_app.py", "/static/other.css"):
        assert client.get(bad).status_code == 404, bad


def test_pages_load_no_outside_resources(client, tmp_path):
    job = _fake_job(tmp_path)
    for url in ("/", "/how", f"/result/{job}"):
        body = client.get(url).get_data(as_text=True)
        assert "http://" not in body and "https://" not in body, f"{url} would need the internet"


def test_how_it_works_shows_the_pipeline_and_the_test_results_without_images(client):
    body = client.get("/how").get_data(as_text=True)
    assert "Style encoder" in body and "Conditional VAE" in body and "<svg" in body
    for number in ("0.605", "0.518", "+0.017", "0.583", "0.713"):
        assert number in body
    assert "<img" not in body, "numbers only, no handwriting"


def test_the_result_page_labels_each_drawn_form_and_serves_its_versions(client, tmp_path):
    job = _fake_job(tmp_path)
    body = client.get(f"/result/{job}").get_data(as_text=True)
    assert "beh · initial" in body and "seen · medial" in body
    assert "You never wrote these forms" in body
    assert body.index("What the network drew") < body.index("Type anything") < body.index("Check what was read")
    assert "Some boxes were empty" in body and "Only 18 of 20" in body
    assert client.get(f"/job/{job}/v13.png").status_code == 200
    for bad in ("v1.png", "v123.png", "notes.json", "photo.png"):
        assert client.get(f"/job/{job}/{bad}").status_code == 404, bad


def test_a_job_whose_font_is_still_building_says_so_and_reports_it(client, tmp_path):
    job = _fake_job(tmp_path, font=False)
    body = client.get(f"/result/{job}").get_data(as_text=True)
    assert 'data-font=building' in body and "Building your font" in body
    assert client.get(f"/job/{job}/status").get_json() == {"font": "building"}


def test_errors_say_what_happened_and_what_to_do(client):
    from PIL import Image

    none = client.post("/make", data={"family": "x"}, content_type="multipart/form-data").get_data(as_text=True)
    assert "No photo chosen" in none and "What to do" in none
    b = io.BytesIO()
    Image.new("RGB", (800, 600), (120, 120, 120)).save(b, "JPEG")
    b.seek(0)
    blank = client.post("/make", data={"photo": (b, "x.jpg"), "family": "x"},
                        content_type="multipart/form-data").get_data(as_text=True)
    assert "find the sheet in this photo" in blank and "four black squares" in blank


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
