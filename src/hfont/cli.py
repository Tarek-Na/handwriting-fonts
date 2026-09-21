"""Command line entry points for the whole pipeline.

    hfont fetch      download the Google Fonts corpus
    hfont index      scan font directories into a manifest
    hfont prepare    render the manifest into a glyph cache
    hfont train      train the generator
    hfont template   write the printable handwriting template
    hfont font       photographed template -> installable font
    hfont report     gate metrics on held-out fonts, beside the baseline
    hfont generate   font from a directory of per-letter images
    hfont roundtrip  trace-and-rebuild a real font, as a pipeline check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(message)s",
        datefmt="%H:%M:%S",
    )
    # fontTools is extremely chatty about minor table quirks in third-party
    # fonts, and a corpus scan produces thousands of such warnings.
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def cmd_fetch(args) -> int:
    from .data.corpus import clone_google_fonts

    dest = clone_google_fonts(args.dest)
    print(f"corpus at {dest}")
    return 0


def cmd_index(args) -> int:
    from .charset import get_charset
    from .data.corpus import build_manifest

    charset = get_charset(args.charset)
    manifest = build_manifest(
        [Path(r) for r in args.roots],
        charset,
        min_coverage=args.min_coverage,
        max_per_family=args.max_per_family,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        source=args.source,
    )
    manifest.save(args.out)
    print(manifest.summary())
    print(f"written to {args.out}")
    return 0


def cmd_prepare(args) -> int:
    from .charset import get_charset
    from .data.corpus import Manifest
    from .data.prepare import build_cache, pack_cache
    from .data.render import RenderConfig

    manifest = Manifest.load(args.manifest)
    charset = get_charset(args.charset or manifest.charset)
    config = RenderConfig(size=args.size, supersample=args.supersample)
    build_cache(manifest, args.out, charset, config, workers=args.workers,
                overwrite=args.overwrite)
    if not args.no_pack:
        pack_cache(args.out)
    print(f"cache at {args.out}")
    return 0


def cmd_train(args) -> int:
    from .train import TrainConfig, Trainer

    cfg = TrainConfig(cache_dir=args.cache, out_dir=args.out)
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = _apply_overrides(cfg, overrides)
    for item in args.set or []:
        key, _, value = item.partition("=")
        cfg = _apply_overrides(cfg, {key: _coerce(value)})

    Trainer(cfg).train()
    return 0


def _apply_overrides(cfg, overrides: dict):
    """Apply dotted-key overrides onto a nested dataclass config."""
    for key, value in overrides.items():
        head, _, rest = key.partition(".")
        if not hasattr(cfg, head):
            raise SystemExit(f"unknown config key: {key}")
        if rest:
            cfg = replace(cfg, **{head: _apply_overrides(getattr(cfg, head), {rest: value})})
        else:
            cfg = replace(cfg, **{head: value})
    return cfg


def _coerce(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def cmd_roundtrip(args) -> int:
    """Render a real font, trace it, rebuild it, and report the fidelity.

    This is the pipeline's self-check with the model taken out of the loop. If
    it does not score well, nothing the generator produces can either, and the
    problem is in the tracer or the font builder rather than in training.
    """
    import numpy as np

    from .charset import get_charset
    from .data.render import FontFrame, FontRenderer, RenderConfig
    from .fontbuild.build import BuildConfig, FontMetadata, rasters_to_font, save_font

    charset = get_charset(args.charset)
    rc = RenderConfig(size=args.size)
    original, _ = FontRenderer(args.font, rc).render(charset)

    bc = BuildConfig(
        image_size=rc.size, baseline=rc.baseline, margin=rc.margin,
        ascender_em=rc.ascender_em,
    )
    fb = rasters_to_font(
        {k: v.image for k, v in original.items()},
        {k: v.advance for k, v in original.items()},
        charset,
        FontMetadata(family=args.family),
        bc,
    )
    out = save_font(fb, args.out)

    frame = FontFrame(
        scale=rc.pixels_per_em / 1000.0, baseline_y=rc.baseline_px, units_per_em=1000
    )
    rebuilt, _ = FontRenderer(out, rc).render(charset, frame=frame)

    scores = []
    for key in sorted(set(original) & set(rebuilt)):
        a = original[key].image > 0.5
        b = rebuilt[key].image > 0.5
        scores.append(((a & b).sum() / max((a | b).sum(), 1), key))
    values = np.asarray([s for s, _ in scores])

    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB, {len(scores)} glyphs)")
    print(f"  IoU mean {values.mean():.4f}  min {values.min():.4f}  p05 {np.percentile(values, 5):.4f}")
    worst = sorted(scores)[:6]
    print("  worst: " + ", ".join(f"{chr(int(k.split('.')[0], 16))!r}={s:.3f}" for s, k in worst))
    return 0 if values.mean() >= args.threshold else 1


def cmd_font(args) -> int:
    """Photographed template -> installable font."""
    from .generate import font_from_photo

    path, quality = font_from_photo(
        args.checkpoint, args.photo, args.out, args.content_font,
        family=args.family, seed=args.seed, freehand=args.freehand,
        rows=tuple(int(n) for n in args.rows.split(",")), debug_path=args.debug,
    )
    print(f"font written to {path}")
    if quality:
        score = quality["tol_f1"]
        # Bands from the held-out corpus: test-split mean 0.84, handwriting 0.69.
        verdict = ("good" if score >= 0.80 else "usable" if score >= 0.65
                   else "rough - expect visible errors in generated letters")
        print(f"predicted quality (leave-one-out tol-F1): {score:.3f} - {verdict}")
    return 0


def cmd_template(args) -> int:
    """Write the printable handwriting template."""
    from .charset import get_charset
    from .intake import save_template

    path = save_template(args.out, get_charset("latin_core"), args.seed)
    print(f"template written to {path} - print it in colour")
    return 0


def cmd_report(args) -> int:
    """Gate metrics on a held-out split, beside the content-copy baseline."""
    from .evaluate.report import evaluate_split, format_summary, write_report

    results = evaluate_split(
        args.checkpoint, args.cache, split=args.split,
        max_fonts=args.max_fonts, with_loo=not args.no_loo,
    )
    summary = write_report(results, args.out)
    print(format_summary(summary))
    print(f"\nwritten to {args.out}")
    return 0


def cmd_generate(args) -> int:
    from .generate import generate_font

    path = generate_font(
        checkpoint=args.checkpoint,
        samples=args.samples,
        out_path=args.out,
        family=args.family,
        cache_dir=args.cache,
    )
    print(f"font written to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hfont", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="clone the Google Fonts corpus")
    p.add_argument("--dest", default="data/google-fonts")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("index", help="scan fonts into a manifest")
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", default="data/manifest.json")
    p.add_argument("--charset", default="latin_core")
    p.add_argument("--min-coverage", type=float, default=1.0)
    p.add_argument("--max-per-family", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--test-frac", type=float, default=0.08)
    p.add_argument("--source", default="google-fonts")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("prepare", help="render a manifest into a glyph cache")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--out", default="data/cache")
    p.add_argument("--charset", default=None)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--supersample", type=int, default=4)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-pack", action="store_true",
                   help="skip consolidating into a memory-mapped array (slow for big corpora)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train the generator")
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--out", default="runs/phase1")
    p.add_argument("--config", default=None, help="JSON file of config overrides")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config field, e.g. --set loss.adversarial=1.0")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("roundtrip", help="trace-and-rebuild check with no model")
    p.add_argument("font")
    p.add_argument("--out", default="data/roundtrip.otf")
    p.add_argument("--charset", default="latin_core")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--family", default="RoundTrip")
    p.add_argument("--threshold", type=float, default=0.90)
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("template", help="write the printable handwriting template")
    p.add_argument("--seed", default="seed30",
                   help="seed12 / seed24 / seed30 (30 adds i j f, which the model draws worst)")
    p.add_argument("--out", default="output/template.png")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("font", help="photographed template -> installable font")
    p.add_argument("photo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--content-font", default="data/fonts/NotoSans[wdth,wght].ttf")
    p.add_argument("--seed", default="seed30", help="must match the template you printed")
    p.add_argument("--family", default="My Handwriting")
    p.add_argument("--out", default="output/my_handwriting.otf")
    p.add_argument("--freehand", action="store_true",
                   help="letters written on blank paper instead of a printed template")
    p.add_argument("--rows", default="10,10,10",
                   help="characters per written row, with --freehand")
    p.add_argument("--debug", default=None,
                   help="write an overlay of the letters found, with --freehand")
    p.set_defaults(func=cmd_font)

    p = sub.add_parser("report", help="gate metrics on held-out fonts")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max-fonts", type=int, default=None)
    p.add_argument("--no-loo", action="store_true", help="skip leave-one-out on seed letters")
    p.add_argument("--out", default="runs/report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("generate", help="build a font from handwriting samples")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", required=True, help="directory of reference glyph images")
    p.add_argument("--out", default="output/handwriting.otf")
    p.add_argument("--family", default="My Handwriting")
    p.add_argument("--cache", default=None, help="glyph cache, for content glyphs")
    p.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
