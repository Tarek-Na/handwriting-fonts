"""Stress test of the short-sheet pipeline: how far can a new writer's sheet differ from AHAWP?

    python scripts/stress_test.py --split val --out runs/stress_val.json

For each writer of a split and each condition, the writer's real isolated
letters (copy 0) are changed as a different hand, pen or phone would change
them, written into the 20-box sheet, photographed (simulated), read back,
and given to the model, which draws 5 versions of every joined form. Those
are scored against the writer's real joined copies as ``variation.py
score`` does (realism, coverage, and realism against the next writer for
specificity). A writer who writes 1.5x larger should get joined forms 1.5x
larger, so drawings are scaled back by the same factor before scoring.

Conditions: letter size (x0.6 to x1.5), pen (thinner, thicker), a harsh
photo (strong tilt, dim uneven light, a shadow, blur, noise, JPEG 50), and
faint pale-blue ink. Each row also records what the demo's guard can see
on a sheet: the letters' median height, the pen half-width, how many boxes
were read, and the largest component (how far outside training the input
lies).

Every image is of real handwriting and stays in memory; only numbers are
written.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

MODEL = ROOT / "runs/v1_drop/best.pt"
CHOSEN = [c for c in os.environ.get("STRESS_CONDITIONS", "").split(",") if c]   # (run a subset)
INK_BLUE = np.array([30, 45, 120], np.float64)
INK_FAINT = np.array([110, 130, 200], np.float64)
CONDITIONS = {
    "clean": {},
    "size x0.6": {"scale": 0.6},
    "size x0.8": {"scale": 0.8},
    "size x1.25": {"scale": 1.25},
    "size x1.5": {"scale": 1.5},
    "thin pen": {"pen": -1},
    "thick pen": {"pen": 2},
    "harsh photo": {"photo": "harsh"},
    "faint ink": {"ink": "faint"},
}


def change_letter(ink: np.ndarray, scale: float = 1.0, pen: int = 0) -> np.ndarray:
    """One letter's ink cut to the ink, its stroke made thinner (pen < 0) or thicker, then scaled."""
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion

    m = np.asarray(ink) > 0.5
    rows, cols = np.nonzero(m)
    m = np.pad(m[rows.min():rows.max() + 1, cols.min():cols.max() + 1], 4)
    if pen > 0:
        m = binary_dilation(m, iterations=pen)
    elif pen < 0:
        thinner = binary_erosion(m, iterations=-pen)
        m = thinner if thinner.sum() > 0.3 * m.sum() else m        # a hairline pen still leaves a line
    if scale != 1.0:
        im = Image.fromarray(m.astype(np.uint8) * 255)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BILINEAR)
        m = np.asarray(im) > 127
    return m.astype(np.float32)


def harsh(page: np.ndarray, seed: int) -> np.ndarray:
    """A bad phone photo: strong tilt, dim and uneven light with a shadow, blur, noise, JPEG 50."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage.transform import ProjectiveTransform, warp

    rng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    pad = int(0.1 * max(h, w))
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 70.0)
    canvas[pad:pad + h, pad:pad + w] = page
    H, W = canvas.shape[:2]
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    t = ProjectiveTransform()
    t.estimate(src, src + rng.uniform(-0.08, 0.08, size=(4, 2)) * [W, H])
    photo = np.stack([warp(canvas[..., c], t, output_shape=(H, W), cval=70, preserve_range=True)
                      for c in range(3)], -1)
    yy, xx = np.mgrid[0:H, 0:W]
    light = 0.45 + 0.4 * (0.7 * xx / W + 0.3 * (1 - yy / H))
    shadow = 1 - 0.35 * (np.abs(xx - W * rng.uniform(0.3, 0.7)) < 0.12 * W)
    photo = photo * (light * gaussian_filter(shadow.astype(float), 25))[..., None] * np.array([1.0, 0.95, 0.85])
    photo = np.stack([gaussian_filter(photo[..., c], 2.0) for c in range(3)], -1)
    photo = np.clip(photo + rng.normal(0, 8.0, photo.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="JPEG", quality=50)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def sheet_photo(letters: dict, cond: dict, seed: int) -> tuple[np.ndarray, int]:
    """The sheet with these letters written in (centred in their boxes), photographed;
    and how many letters had to be cut to fit their box."""
    import arabic_sheet as script
    from simulate_template_photo import degrade

    from hfont import arabic_sheet as sheet

    page, layout = sheet.build_sheet(sheet.short_layout())
    changed = {s: change_letter(im, cond.get("scale", 1.0), cond.get("pen", 0)) for s, im in letters.items()}
    box_h, box_w = layout.cell_h - 12, layout.cell_w - 12
    cut = sum(im.shape[0] > box_h or im.shape[1] > box_w for im in changed.values())
    old = script.INK_RGB
    script.INK_RGB = INK_FAINT if cond.get("ink") == "faint" else INK_BLUE
    try:
        written = script.write_letters(page, layout, changed)
    finally:
        script.INK_RGB = old
    return (harsh(written, seed) if cond.get("photo") == "harsh" else degrade(written, seed)), cut


def guard_features(read: dict, wx: dict) -> dict:
    """What the demo can measure on any sheet (see ``demo_app.check_sheet``)."""
    from hfont.data import ahawp

    heights = [float(np.ptp(np.nonzero(np.asarray(im) > 0.5)[0]) + 1) for im in read.values()]
    comps = [float(np.abs(x[0]).max()) for x in wx.values()]
    return {"height": float(np.median(heights)) if heights else 0.0,
            "pen": float(ahawp.pen_radius(read)) if read else 0.0,
            "read": len(read), "max_component": max(comps) if comps else 0.0}


def _writer(args) -> list[dict]:
    writer, other, seed = args
    import torch

    import ahawp_forms
    import variation as v
    from hfont import arabic_sheet as sheet
    from hfont import skeleton
    from hfont.data import ahawp
    from hfont.models.variation import VariationModel
    from skeleton_moves import load_templates
    from variation import _score_writer

    data = ahawp_forms.load()
    _, inks = v.load_cache()
    pca, temps = v.load_pca(), load_templates()
    names = v.form_names(pca)
    layout = sheet.short_layout()
    specs = {s.name: s for s in layout.specs()}
    letters = {specs[n]: c[0].astype(np.float32) for n, c in data[writer].items() if n in specs and c[0].any()}
    net = VariationModel.from_checkpoint(torch.load(MODEL)).eval()
    rows = []
    for k, (label, cond) in enumerate(CONDITIONS.items()):
        if CHOSEN and label not in CHOSEN:
            continue
        photo, cut = sheet_photo(letters, cond, seed + k)
        read = sheet.read_sheet(photo, layout)
        wx, factor = v.photo_style(read, pca)              # as the demo does: rescaled into the trained range
        row = {"writer": writer, "condition": label, "written": len(letters), "cut": cut, "factor": factor,
               **guard_features(read, wx)}
        if len(wx) < v.MIN_LETTERS:
            rows.append({**row, "failed": True})
            continue
        joined = [n for n in names if "." in n and n in inks[writer]]
        ix, iform, imask = v.style_inputs(wx, pca, lambda d: d[min(d)])
        t = torch.from_numpy
        b = len(joined)
        torch.manual_seed(0)
        with torch.no_grad():
            c = net.condition(torch.tensor([names.index(n) for n in joined]), t(np.repeat(ix[None], b, 0)),
                              t(np.repeat(iform[None], b, 0)), t(np.repeat(imask[None], b, 0)))
        drawn = net.sample(c, v.VERSIONS, 0.4).numpy()
        scale = cond.get("scale", 1.0)
        radius = ahawp.pen_radius(read) / scale
        versions = {n: [skeleton.render_points((temps[n][0] + v.decode(pca, n, x)) / factor / scale, temps[n][1],
                                               radius)
                        for x in xs] for n, xs in zip(joined, drawn)}
        scored = _score_writer((writer, versions, inks[writer], inks[other], ahawp.pen_radius(read)))
        rows.append({**row, "failed": False,
                     **{m: float(np.mean([r[m] for r in scored])) for m in ("realism", "coverage", "other")}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import variation as v
    from hfont.data import ahawp

    _, inks = v.load_cache()
    writers = sorted(w for w in inks if ahawp.split_of(w) == args.split)[:args.limit]
    jobs = [(w, writers[(i + 1) % len(writers)], 1000 * i) for i, w in enumerate(writers)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(_writer, jobs) for r in part]
    args.out.write_text(json.dumps(rows))
    print(f"{len(writers)} {args.split} writers, {time.time() - t0:.0f}s")
    print(f"{'condition':12} {'read':>5} {'failed':>6} {'realism':>8} {'coverage':>9} {'specif.':>8} "
          f"{'height':>7} {'pen':>5} {'max comp':>9}")
    for label in CONDITIONS:
        rs = [r for r in rows if r["condition"] == label]
        if not rs:
            continue
        ok = [r for r in rs if not r["failed"]]
        mean = lambda k, src=ok: np.mean([r[k] for r in src]) if src else float("nan")  # noqa: E731
        spec = np.mean([r["realism"] - r["other"] for r in ok]) if ok else float("nan")
        print(f"{label:12} {np.mean([r['read'] for r in rs]):5.1f} {len(rs) - len(ok):6d} {mean('realism'):8.3f} "
              f"{mean('coverage'):9.3f} {spec:+8.3f} {mean('height', rs):7.1f} {mean('pen', rs):5.2f} "
              f"{mean('max_component', rs):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
