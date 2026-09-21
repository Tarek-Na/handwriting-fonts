# Few-Shot Handwriting Font Generation

## What this is

A system that takes a small sample of a person's handwriting — roughly 20 to 30 letters — and produces a complete, installable OpenType font in that handwriting.

The user writes a small subset. A model synthesizes the rest. The output is a real font file that types correctly in any application.

The end target is **Arabic**, which is where the problem is hard and unsolved. The work is staged so that **English/Latin comes first** and acts as the gate: the full pipeline is proven on the easy script before the hard one is attempted.

## Staging

**Phase 1 — Latin.** Prove the complete pipeline end to end: few-shot style transfer, raster-to-vector conversion, font assembly, installable output. Latin is deliberately the easy case — 26 letters in two cases, no contextual forms, no mandatory joining, and an enormous free font corpus for pretraining. If the pipeline cannot produce convincing Latin fonts from a handful of samples, nothing downstream is worth attempting.

**Phase 2 — Arabic.** Only if Phase 1 clearly succeeds. Arabic adds contextual forms, cursive joining and complex OpenType shaping. This is where the novel research contribution lives.

Phase 1 is a validation gate, not throwaway work. The architecture, training loop, vectorization and font-export code carry over. Arabic changes the glyph inventory, the shaping tables, and adds the joining constraint.

A useful intermediate, if time allows: **Latin cursive**. It introduces stroke connection between adjacent glyphs — the core Arabic difficulty — without the contextual-form explosion. Treat it as an optional bridge rather than a required stage.

### What "works really well" should mean before moving on

Not prescribed here, but the gate should be concrete and decided before Phase 1 starts rather than after. Reasonable things to require: generated glyphs are visually indistinguishable from held-out ground truth at a defensible rate; the exported font installs and types correctly across real applications; and the system works on actual photographed handwriting, not only on held-out typeset fonts. That last condition is the one most likely to fail quietly.

## Why this doesn't already exist

Two separate bodies of work exist, and they have never been connected.

**Commercial handwriting-font tools are not AI.** Calligraphr, Fontifier, Scanahand, MyScriptFont and the various iOS "handwriting font" apps all work the same way: print a template, hand-write every single glyph, scan, threshold, segment the grid cells, trace to Bézier curves, emit a TTF. There is no generative model. The user does all the work — 100+ cells for Latin, far more for Arabic. Apps that advertise "AI" are using it for character extraction and segmentation, not synthesis.

This matters for Phase 1 framing. Latin handwriting fonts are a solved *product*, so Phase 1 is not a novelty claim — it is a technical validation. The claim is few-shot: the user fills in a fraction of the template and the rest is generated.

**Academic few-shot font generation does not produce fonts.** There is a large and active literature (zi2zi, DG-Font, AGIS-Net, FontDiffuser, DA-Font, DRG-Font, GAS-NeXt) framing font generation as image-to-image translation with disentangled content and style representations. Three limitations matter here:

1. It is overwhelmingly targeted at Chinese, Japanese and Korean. The stated motivation is always that a CJK font library contains tens of thousands of characters. Latin's 26 letters make the few-shot problem look trivial, so nobody publishes on it.
2. Output is raster glyph images. Almost nothing connects the model to an installable font file. VecFusion (vector font generation with diffusion) is the main exception.
3. Arabic is nearly untouched. What exists is artistic rather than functional — e.g. StyleGAN2-ada trained on Nastaliq calligraphy to generate aesthetic samples, not fonts.

## Why Arabic is the real target

Arabic is where the problem is genuinely hard and genuinely unsolved, for reasons that don't apply to any of the scripts the literature covers.

**Contextual forms.** Each letter takes up to four shapes depending on its position in a word: isolated, initial, medial, final. Some letters (the six non-connectors: ا د ذ ر ز و) only take two. A ~28-letter alphabet expands to roughly 100–120 required glyphs, plus mandatory ligatures such as lam-alef.

**Cursive joining.** This is the core technical problem. Arabic letters connect along a baseline. A generated medial glyph's exit stroke must meet the next glyph's entry stroke at a consistent height and angle, or the rendered text visibly breaks apart. Nothing in the CJK font-generation literature addresses this, because Chinese characters do not join. This is the most defensible novel contribution available in this project.

**Shaping tables.** For the font to work, the exported font needs correct OpenType GSUB tables so that the text shaper substitutes the right contextual form as the user types. This is the step that separates "a paper figure" from "a font you can install." The research pipeline consistently skips it. Latin needs almost none of this, which is part of why Phase 1 is cheap.

## Data situation

Training data is free and requires no manual labeling, for both phases.

**Latin.** Thousands of freely licensed font families are available (Google Fonts alone is well over a thousand). Rendering uppercase, lowercase, digits and common punctuation across them yields a very large, perfectly labeled corpus. Handwriting-style and script-style families within that corpus are the most relevant slice for style range.

**Arabic.** Several hundred freely licensed fonts exist. Rendering every glyph — all contextual forms, ligatures, digits, punctuation — yields on the order of 50,000 to 100,000 images, labeled by character, contextual form and font identity.

In both cases the corpus is typeset fonts, while the inference-time target is a real person's handwriting. That distribution gap is a known risk and should be measured, not assumed away. It is also cheaper to discover in Phase 1 than in Phase 2.

## Modeling notes

These are observations about the problem space, not a prescribed solution. The implementation plan is open.

- The natural framing is content/style disentanglement: one encoder takes a reference glyph specifying *which* character (and, for Arabic, which contextual form), another encoder takes the style reference, a decoder combines them. Training mixes content from one font with style from another and supervises with reconstruction against ground truth.
- Fine-tuning Stable Diffusion is likely a poor fit. Its VAE is tuned for photographic texture and degrades thin, high-contrast strokes — exactly the signal that matters, since generated rasters are subsequently vectorized. Its prior (photos, art, faces) does not transfer to glyph topology. There is no pretrained glyph-domain diffusion model to fine-tune. If diffusion is used, pixel-space at modest resolution is the sensible form.
- Glyph images are binary, low-entropy and topologically constrained. Generative diversity is undesirable here — the goal is one deterministic correct letterform per (character, form, style).
- The compute footprint is small. Models in the tens of millions of parameters at 128×128 grayscale train on a single consumer GPU in hours, not days. Fast iteration is more valuable than model scale, especially in Phase 1 where the point is to learn quickly whether the approach holds.
- Existing repos (DG-Font, FontDiffuser) are reasonable starting points for the training loop and losses. Their radical/component decomposition logic is CJK-specific and does not transfer.
- Raster-to-vector conversion is a solved classical problem (potrace-style tracing); it does not need to be learned. Font assembly is a FontTools job. Both are shared across phases.
- For Arabic, the joining constraint likely needs explicit modeling rather than emerging for free. Plausible directions: auxiliary prediction of connection-point coordinates with a cross-glyph consistency loss; training on letter pairs with a seam-discontinuity penalty; post-hoc baseline alignment. Worth testing which is necessary.

## Evaluation

Style similarity against held-out fonts is the obvious automatic metric, but it does not capture whether the font is *usable*. Evaluation should cover at least:

- Reconstruction and style-similarity metrics on held-out fonts, where ground truth exists.
- Whether the exported font shapes and renders correctly in real text engines (HarfBuzz, browsers, word processors).
- Human judgment on real handwriting samples, since held-out typeset fonts are an easier distribution than actual handwriting.
- For Arabic specifically: joining quality — a measurable discontinuity metric at glyph seams in rendered words, not just per-glyph quality.

The same evaluation harness should serve both phases, so that Phase 1 results are directly comparable to Phase 2 and the gate decision rests on numbers rather than impressions.

## Constraints

- Timeline: approximately two months total, covering both phases.
- Fully software. No hardware component.
- Single GPU, consumer or free-tier cloud.
- Arabic target is ordinary Naskh-style handwriting, not decorative or Nastaliq calligraphy.
- The deliverable is a working end-to-end product — sample intake through to an installed, typeable font — not only a trained model.

## Known risks

- **Phase 1 succeeds and Phase 2 doesn't fit in the remaining time.** The most likely outcome if Phase 1 runs long. Budget accordingly; a polished Latin system with a partial Arabic result is a worse outcome than planning the split honestly from the start.
- **Joining may not be learnable at this scale.** If Arabic glyph-level generation works but connections look wrong, the project still has a result, but the framing shifts.
- **Domain gap.** Models pretrained on clean typeset fonts may degrade badly on photographed handwriting. Preprocessing and augmentation strategy matters. Phase 1 is the cheap place to find this out.
- **Export complexity.** Correct Arabic OpenType shaping (GSUB, contextual substitution, ligature handling) is intricate and easy to underestimate. Latin export is simple enough that Phase 1 will not surface these problems — they need separate early validation before Phase 2 begins.

## If Arabic proves out of reach

Latin cursive preserves the core research question — stroke connection between adjacent glyphs — in an easier form, and remains something existing template-based tools handle poorly. It is the natural landing point if the Arabic contextual-form and shaping complexity cannot be absorbed in the time available.
