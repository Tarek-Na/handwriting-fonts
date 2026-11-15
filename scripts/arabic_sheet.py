"""The Arabic sheet: print it, read a photo of it, and test the reader without one.

    python scripts/arabic_sheet.py template --out output/ahawp/arabic_sheet.png
    python scripts/arabic_sheet.py read PHOTO --font output/ahawp/mine.otf --preview output/ahawp/mine.png \
        --boxes output/ahawp/mine_boxes.png
    python scripts/arabic_sheet.py simulate --split test

``simulate`` needs no photographed sheet: it writes each test writer's real
AHAWP letters into the template as blue ink, degrades the page as a phone
photo does (``simulate_template_photo.degrade``: perspective, uneven light,
blur, noise, JPEG), reads it back with ``read_sheet`` and scores each box
against the letter that went in (``aligned_centreline_f1``), then builds the
joining font from what was read and checks its joins. The pages are images
of real handwriting: they stay in ``output/``, never in git.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hfont import arabic_sheet as sheet  # noqa: E402
from hfont.data import ahawp  # noqa: E402
from hfont.evaluate.metrics import aligned_centreline_f1  # noqa: E402

INK_RGB = np.array([30, 45, 120], dtype=np.float64)        # blue ballpoint


def write_letters(page: np.ndarray, layout: sheet.SheetLayout, letters: dict, inset: int = 6) -> np.ndarray:
    """Each letter image (AHAWP framing: 128 rows, any width) inked into its box, centred across."""
    page = page.astype(np.float64).copy()
    for spec, im in letters.items():
        try:
            x0, y0, x1, y1 = layout.cell_box(spec)
        except StopIteration:
            continue
        box_w, box_h = x1 - x0 - 2 * inset, y1 - y0 - 2 * inset
        ink = np.asarray(im, np.float64)[:box_h]
        cols = np.nonzero(ink.any(axis=0))[0]
        if cols.size == 0:
            continue
        ink = ink[:, cols[0]:cols[-1] + 1][:, :box_w]
        top = y0 + inset + (box_h - ink.shape[0]) // 2
        left = x0 + inset + (box_w - ink.shape[1]) // 2
        cover = ink[..., None]
        region = page[top:top + ink.shape[0], left:left + ink.shape[1]]
        page[top:top + ink.shape[0], left:left + ink.shape[1]] = region * (1 - cover) + INK_RGB * cover
    return np.clip(page, 0, 255).astype(np.uint8)


def _swap(letters: dict, a, b) -> None:
    if a in letters and b in letters:
        letters[a], letters[b] = letters[b], letters[a]


def simulate(split: str, seed: int, save: Path | None, checker: Path | None = None) -> None:
    import ahawp_font
    import ahawp_forms
    from simulate_template_photo import degrade

    from hfont.charset import GlyphSpec

    data = ahawp_forms.load()
    names = {s.name: s for s in sheet.sheet_specs()}
    page, layout = sheet.build_sheet()
    writers = sorted(w for w in data if ahawp.split_of(w) == split)
    scores, missed, extra, joins = [], 0, 0, []
    flagged_ok, caught, planted = 0, {"alike": 0, "random": 0}, 0
    rng = np.random.default_rng(seed)
    for i, writer in enumerate(writers):
        letters = {names[n]: v[0].astype(np.float32) for n, v in data[writer].items() if n in names and v[0].any()}
        photo = degrade(write_letters(page, layout, letters), seed + i)
        if checker is not None:
            # Two mistakes a writer could make, on a copy of the page: forms that
            # differ only in their dots swapped, and two random boxes swapped.
            wrong = dict(letters)
            alike = (GlyphSpec("ب", sheet.Form.INIT), GlyphSpec("ن", sheet.Form.INIT))
            pool = [s for s in wrong if s not in alike and s.name != "uni0637.init"]
            rand = tuple(pool[j] for j in rng.choice(len(pool), 2, replace=False))
            _swap(wrong, *alike)
            _swap(wrong, *rand)
            flags = sheet.check_boxes(sheet.read_sheet(degrade(write_letters(page, layout, wrong), seed + i + 1000),
                                                       layout), checker)
            caught["alike"] += sum(s in flags for s in alike)
            caught["random"] += sum(s in flags for s in rand)
            planted += 2
        if save is not None and i == 0:
            from PIL import Image
            save.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(photo).save(save)
        read = sheet.read_sheet(photo, layout)
        missed += sum(1 for s in letters if s not in read)
        extra += sum(1 for s in read if s not in letters)
        scores += [aligned_centreline_f1(read[s], letters[s]) for s in letters if s in read]
        if checker is not None:
            flagged_ok += len(sheet.check_boxes(read, checker))
        font, glyphs, _ = ahawp_font.letters_font(read, f"Sheet {writer}")
        buffer = io.BytesIO()
        font.save(buffer)
        joins.append(ahawp_font.check_joins(buffer.getvalue(), glyphs, ahawp_font.WORDS + ahawp_font.COMPOSED_WORDS))
    total = sum(len([n for n in data[w] if n in names and data[w][n][0].any()]) for w in writers)
    print(f"{len(writers)} {split} writers, {total} written boxes, each page degraded as a phone photo")
    print(f"  boxes read                       {total - missed} ({missed} missed, {extra} read from empty boxes)")
    print(f"  read letter vs letter written    centreline F1 {np.mean(scores):.3f} (min {np.min(scores):.3f})")
    print(f"  joins that miss in test words    {sum(j['misses'] for j in joins)} of {sum(j['joins'] for j in joins)}"
          f" (furthest off the line {max(j['drift'] for j in joins)} units,"
          f" letters with no glyph {sum(j['missing'] for j in joins)})")
    if checker is not None:
        print(f"  correctly filled boxes flagged   {flagged_ok} ({flagged_ok / len(writers):.1f} a sheet)")
        print(f"  dots-only swap caught            {caught['alike']} of {planted} boxes")
        print(f"  random swap caught               {caught['random']} of {planted} boxes")


def read(photo_path: Path, font_path: Path | None, preview: Path | None, checker: Path | None = None,
         boxes: Path | None = None) -> None:
    from PIL import Image

    import ahawp_font

    letters, flagged, turned = sheet.read_upright(sheet.load_photo(photo_path), checker)
    if turned:
        print("the page was upside down: read it turned round")
    print(f"{len(letters)} of {len(sheet.sheet_specs())} boxes have writing")
    empty = [s for s in sheet.sheet_specs() if s not in letters]
    if empty:
        print("  empty:", " ".join(sheet.presentation_form(s) for s in empty))
    for spec, (own, likely) in flagged.items():
        print(f"  check the {sheet.presentation_form(spec)} box ({spec.name}): looks like {likely}"
              f" (p {own:.2f} for its own label)")
    if boxes:
        boxes.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sheet.boxes_image(letters, flagged)).save(boxes)
        print("boxes as read ->", boxes)
    font, glyphs, _ = ahawp_font.letters_font(letters, photo_path.stem)
    buffer = io.BytesIO()
    font.save(buffer)
    if font_path:
        font_path.parent.mkdir(parents=True, exist_ok=True)
        font_path.write_bytes(buffer.getvalue())
        print("font ->", font_path)
    if preview:
        image = ahawp_font.render(buffer.getvalue(), ahawp_font.WORDS + ahawp_font.COMPOSED_WORDS)
        Image.fromarray(image).save(preview)
        print("preview ->", preview)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--out", type=Path, default=ROOT / "output/ahawp/arabic_sheet.png")
    r = sub.add_parser("read")
    r.add_argument("photo", type=Path)
    r.add_argument("--font", type=Path, default=None)
    r.add_argument("--preview", type=Path, default=None)
    r.add_argument("--boxes", type=Path, default=None, help="PNG of every box as read (private)")
    r.add_argument("--checker", type=Path, default=ROOT / "runs/c1/best.pt",
                   help="form classifier that flags boxes not matching their label (scripts/train_formclass.py)")
    s = sub.add_parser("simulate")
    s.add_argument("--split", default="val")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--save", type=Path, default=None, help="write the first simulated photo here")
    s.add_argument("--checker", type=Path, default=None, help="also test the form classifier's flags")
    args = ap.parse_args()
    if args.cmd == "template":
        print("sheet ->", sheet.save_sheet(args.out))
    elif args.cmd == "read":
        read(args.photo, args.font, args.preview, args.checker if args.checker.exists() else None, args.boxes)
    else:
        simulate(args.split, args.seed, args.save, args.checker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
