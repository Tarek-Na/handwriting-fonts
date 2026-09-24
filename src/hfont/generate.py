"""Inference: style references in, installable font out.

This is the deliverable the brief asks for — "sample intake through to an
installed, typeable font" — and it is deliberately the same code path whether
the references come from a held-out corpus font or from a photograph of
someone's handwriting. Anything that only works on clean typeset input is
measuring the easy distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .charset import Charset, get_charset, seed_charset
from .data.dataset import GlyphStore
from .fontbuild.build import BuildConfig, FontMetadata, rasters_to_font, save_font
from .models.generator import GeneratorConfig, GlyphGenerator
from .utils import pick_device

log = logging.getLogger(__name__)


class LoadedModel:
    """A checkpoint restored into an eval-ready generator."""

    def __init__(self, generator: GlyphGenerator, charset: Charset, image_size: int) -> None:
        self.generator = generator
        self.charset = charset
        self.image_size = image_size
        self.device = next(generator.parameters()).device


def load_checkpoint(
    path: str | Path, device: torch.device | None = None, use_ema: bool = True
) -> LoadedModel:
    """Restore a generator, preferring the EMA weights.

    EMA weights are the default because the raw weights carry step-to-step
    jitter in the glyph edges, and that jitter becomes visible wobble once the
    raster is traced into curves.
    """
    device = device or pick_device()
    state = torch.load(str(path), map_location=device, weights_only=False)

    config = state["config"]
    gen_cfg = GeneratorConfig(**config["generator"])
    charset = get_charset(state.get("charset", config.get("data", {}).get("charset", "latin_core")))

    generator = GlyphGenerator(gen_cfg).to(device)
    generator.load_state_dict(state["generator"])

    if use_ema and "ema" in state:
        shadow = state["ema"]["shadow"]
        missing = 0
        with torch.no_grad():
            for name, param in generator.named_parameters():
                if name in shadow:
                    param.copy_(shadow[name].to(param.dtype).to(device))
                else:
                    missing += 1
        if missing:
            log.warning("%d parameters had no EMA shadow; left at raw values", missing)

    generator.eval()
    log.info(
        "loaded %s (step %s, %.2fM params, %d glyphs)",
        Path(path).name, state.get("step", "?"),
        generator.num_parameters() / 1e6, len(charset),
    )
    return LoadedModel(generator, charset, gen_cfg.image_size)


def content_images_from_font(
    font_path: str | Path, charset: Charset, image_size: int = 128
) -> dict[str, np.ndarray]:
    """Render content glyphs straight from a font file.

    The rasterizer is deterministic, so rendering the content font here gives
    byte-identical images to the ones in the training cache. That removes the
    need to carry a multi-gigabyte cache to wherever inference runs.
    """
    from .data.render import FontRenderer, RenderConfig

    rasters, _ = FontRenderer(font_path, RenderConfig(size=image_size)).render(charset)
    # Quantize exactly as the cache does (prepare.render_one), so the model sees
    # the same 8-bit values it was trained on rather than unrounded floats.
    return {
        k: np.round(v.image * 255.0).astype(np.uint8).astype(np.float32) / 255.0
        for k, v in rasters.items()
    }


def _stack_refs(images: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    refs = np.stack(images).astype(np.float32)
    tensor = torch.from_numpy(refs).unsqueeze(1).unsqueeze(0) * 2.0 - 1.0
    mask = torch.ones(1, len(images), dtype=torch.bool)
    return tensor.to(device), mask.to(device)


@torch.no_grad()
def generate_rasters(
    model: LoadedModel,
    reference_images: dict[str, np.ndarray],
    content_images: dict[str, np.ndarray],
    batch_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Generate every glyph of the charset in the referenced style.

    Reference glyphs are passed through unchanged rather than regenerated. The
    user actually wrote those, so a generated substitute could only be worse,
    and keeping them anchors the font to real samples.
    """
    device = model.device
    refs, mask = _stack_refs(list(reference_images.values()), device)
    # A model with reference attention needs the per-reference features too;
    # encoding only the pooled code would leave those weights unused at
    # inference and silently undo whatever they learned.
    attends = getattr(model.generator, "ref_attention", None) is not None
    if attends:
        style, ref_feats = model.generator.encode_style_full(refs, mask)
    else:
        style, ref_feats = model.generator.encode_style(refs, mask), None

    wanted = [g for g in model.charset if g.key in content_images]
    missing = [g.key for g in model.charset if g.key not in content_images]
    if missing:
        log.warning("no content glyph for %d entries; they will be absent", len(missing))

    images: dict[str, np.ndarray] = {}
    advances: dict[str, float] = {}

    glyph_index = {g.key: i for i, g in enumerate(model.charset)}

    for start in range(0, len(wanted), batch_size):
        chunk = wanted[start : start + batch_size]
        content = torch.from_numpy(
            np.stack([content_images[g.key] for g in chunk]).astype(np.float32)
        ).unsqueeze(1).to(device) * 2.0 - 1.0
        char_ids = torch.tensor([glyph_index[g.key] for g in chunk], device=device)

        n = len(chunk)
        out = model.generator.decode(
            content, char_ids, style.expand(n, -1),
            ref_feats.expand(n, *ref_feats.shape[1:]) if attends else None,
            mask.expand(n, -1) if attends else None,
        )
        rendered = ((out["image"].float() + 1.0) * 0.5).clamp(0, 1).cpu().numpy()[:, 0]
        for spec, image in zip(chunk, rendered):
            images[spec.key] = image
        if "advance" in out:
            for spec, value in zip(chunk, out["advance"].float().cpu().numpy()):
                advances[spec.key] = float(value)

    # Real samples win over generated ones wherever we have them.
    for key, image in reference_images.items():
        images[key] = image

    return images, advances


def generate_from_cache_font(
    checkpoint: str | Path,
    cache_dir: str | Path,
    out_path: str | Path,
    font_id: str | None = None,
    family: str = "Generated",
    ref_seed: str = "seed24",
    split: str = "val",
) -> Path:
    """Generate a font imitating one corpus font, using only its seed glyphs.

    The evaluation workhorse: ground truth exists for every glyph, so the output
    can be scored, while the model still only ever sees the ~24 glyphs a user
    would realistically write.
    """
    store = GlyphStore(cache_dir)
    model = load_checkpoint(checkpoint)

    records = store.split(split) or store.records
    if font_id is None:
        font_id = records[0].font_id
    log.info("style source: %s", font_id)

    seed = seed_charset(store.charset, ref_seed)
    available = set(store.keys(font_id))
    reference_images = {
        g.key: store.image(font_id, g.key) for g in seed if g.key in available
    }
    if not reference_images:
        raise ValueError(f"font {font_id} has none of the seed glyphs")

    content_font = store.pick_content_font("train")
    content_keys = set(store.keys(content_font))
    content_images = {
        g.key: store.image(content_font, g.key)
        for g in store.charset
        if g.key in content_keys
    }

    images, advances = generate_rasters(model, reference_images, content_images)

    # Fall back to the real advance where the model has no prediction.
    for key in images:
        advances.setdefault(key, store.advance(font_id, key) if key in available else 0.5)

    render_cfg = store.render_config
    build_cfg = BuildConfig(
        image_size=int(render_cfg["size"]),
        baseline=float(render_cfg.get("baseline", 0.75)),
        margin=float(render_cfg.get("margin", 0.06)),
        ascender_em=float(render_cfg.get("ascender_em", 0.8)),
    )
    fb = rasters_to_font(
        images, advances, store.charset, FontMetadata(family=family), build_cfg
    )
    return save_font(fb, out_path)


def font_from_photo(
    checkpoint: str | Path,
    photo: str | Path,
    out_path: str | Path,
    content_font: str | Path,
    family: str = "My Handwriting",
    seed: str = "seed24",
    freehand: bool = False,
    rows: tuple[int, ...] = (10, 10, 10),
    debug_path: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    """The product entry point: photographed template in, installable font out.

    ``freehand`` reads letters written on blank paper instead of a printed
    template — the same path, minus the printer.

    Also returns the leave-one-out score on the writer's own letters. Across 461
    held-out fonts it tracks the true quality of the generated glyphs at
    r = 0.98, so it is a trustworthy preview of how good this font will be —
    the one quality signal available when the writer is a real person and only
    their seed letters have ground truth.
    """
    from .evaluate.report import leave_one_out
    from .evaluate.shaping import check_font
    from .intake import FreehandConfig, load_freehand_photo, load_template_photo

    model = load_checkpoint(checkpoint)
    if freehand:
        references = load_freehand_photo(
            photo, model.charset, seed, model.image_size,
            FreehandConfig(rows=tuple(rows)), debug_path,
        )
    else:
        references = load_template_photo(photo, model.charset, seed, model.image_size)
    if not references:
        raise ValueError(f"no letters recovered from {photo}")
    log.info("recovered %d letters from the template", len(references))

    content = content_images_from_font(content_font, model.charset, model.image_size)
    images, advances = generate_rasters(model, references, content)
    for key in images:
        advances.setdefault(key, 0.5)

    fb = rasters_to_font(images, advances, model.charset, FontMetadata(family=family), BuildConfig())
    path = save_font(fb, out_path)

    quality = leave_one_out(model, references, content)
    report = check_font(path, charset=model.charset)
    if not report.ok:
        log.warning("exported font failed validation:\n%s", report.summary())
    return path, quality


def generate_font(
    checkpoint: str | Path,
    samples: str | Path,
    out_path: str | Path,
    family: str = "My Handwriting",
    cache_dir: str | Path | None = None,
) -> Path:
    """Generate a font from a directory of handwriting sample images.

    Each file must be named for the character it shows (``A.png``, ``a.png``,
    ``comma.png`` or ``U+0041.png``). Intake from a photographed template is
    handled by :mod:`hfont.intake`, which writes exactly this layout.
    """
    from .intake import load_sample_directory

    model = load_checkpoint(checkpoint)
    reference_images = load_sample_directory(samples, model.charset, model.image_size)
    if not reference_images:
        raise ValueError(f"no usable sample images found in {samples}")
    log.info("loaded %d handwriting samples", len(reference_images))

    if cache_dir is None:
        raise ValueError(
            "a glyph cache is required to supply content glyphs; pass --cache"
        )
    store = GlyphStore(cache_dir)
    content_font = store.pick_content_font("train")
    content_keys = set(store.keys(content_font))
    content_images = {
        g.key: store.image(content_font, g.key)
        for g in model.charset
        if g.key in content_keys
    }

    images, advances = generate_rasters(model, reference_images, content_images)
    for key in images:
        advances.setdefault(key, 0.5)

    render_cfg = store.render_config
    build_cfg = BuildConfig(
        image_size=int(render_cfg["size"]),
        baseline=float(render_cfg.get("baseline", 0.75)),
        margin=float(render_cfg.get("margin", 0.06)),
        ascender_em=float(render_cfg.get("ascender_em", 0.8)),
    )
    fb = rasters_to_font(
        images, advances, model.charset, FontMetadata(family=family), build_cfg
    )
    return save_font(fb, out_path)
