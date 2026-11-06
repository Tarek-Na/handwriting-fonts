# Handwriting-specific style generation, stroke/vector glyph generation, and composition/retrieval for a ~30-sample handwriting font (2018-2026)

Notes written 2026-11-06. Sources marked **[fetched]** had their arXiv or official page read this session. Sources marked **[search]** are known only from search-result snippets. Sources marked **[prior knowledge, not re-verified]** come from background knowledge of the literature, with the canonical arXiv URL. Treat their numbers as approximate and check them before quoting.

Project framing used for the inferences below: the user has about 30 isolated Latin glyphs, written in boxes and photographed. The system must produce the other ~45 of 75 glyphs, keep the writer's *letter shapes* (not just stroke weight), and run on one consumer GPU. The failure being addressed is "style collapse" in a font-trained AdaIN encoder-decoder.

## Q1. Styled handwritten text/character generation (GANwriting → One-DM → 2025-26 successors)

### Takeaway
Almost all of this literature generates **word or line images** conditioned on 1-15 style images from IAM or CVL. Writer style is captured mostly as global slant, stroke width, cursiveness and ink texture, scored with FID or HWD. Only One-DM (ECCV 2024) reports **isolated-character** experiments, and those are Chinese and Japanese. One-DM also says single-letter references work poorly. None of these models is a drop-in way to make isolated Latin glyphs from isolated-glyph references. The closest is HandwritingAgent (June 2026), which is a glyph-bank *retrieval-and-adapt* system rather than a generative model.

### Cited Findings
**One-DM (Dai et al., ECCV 2024, arXiv 2409.04004)** [fetched]
- Generates offline handwritten text images, mainly words. It also has character-level experiments on Chinese (ICDAR-2013) and Japanese (UP_Kuchibue). Content glyphs come from Unifont, so any Unicode character can be requested — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- Uses **one** reference image, against 15 or more for earlier few-shot baselines. Two parallel style encoders read the reference and its Laplacian high-frequency component. A "LaplacianNCE" contrastive loss makes the high-frequency features writer-discriminative, a gate suppresses background, and cross-attention fuses style with content to condition a latent diffusion model. The authors stress contour cues such as slant and cursive joins over texture — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- Reported IAM styled FID (IV-S / IV-U / OOV-S / OOV-U): One-DM 89.47 / 98.36 / 93.30 / 102.75. WordStylist 98.10 / 104.27 / 109.45 / 115.52. VATr 103.75 / 101.73 / 111.64 / 108.76. CVL FID: 51.78 for One-DM, 54.44 for VATr, 58.22 for HWT. Chinese character FID: 27.24 vs 34.61 for WordStylist. Japanese: 95.43 vs 101.93. OCR CER on IAM: 11.69% vs 26.98% for VATr (4.98% for real data). No HWD reported — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- Compute: four RTX 3090s, 700 epochs at batch 384, then about 4,500 fine-tuning iterations with a CTC recognizer. DDIM sampling with 50 steps. Total wall-clock time not reported — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- Stated limitations: **a single-letter reference performs poorly, and the authors recommend references longer than four letters.** Long-tail characters (some digits and punctuation) are sometimes wrong. Ink colour varies between outputs — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- Code: github.com/dailenson/One-DM, MIT licence, PyTorch — [ecosyste.ms listing](https://awesome.ecosyste.ms/projects/github.com%2Fdailenson%2FOne-DM); [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf)

**DiffusionPen (Nikolaidou, Retsinas, Sfikas, Liwicki, ECCV 2024, arXiv 2409.06065)** [search]
- A **5-shot** styled handwritten-text generator built on latent diffusion and evaluated on IAM. Its generated data improves handwriting text recognition (HTR) — [ECCV page](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11492_ECCV_2024_paper.php); [arXiv](https://www.arxiv.org/pdf/2409.06065)
- Code at github.com/koninik/DiffusionPen. Pretrained weights, including the style encoder, are on Hugging Face — [HF model card](https://huggingface.co/konnik/DiffusionPen)

**2025-2026 successors** [search unless noted]
- **Emuru** (arXiv 2503.17074, March 2025): a VAE combined with an autoregressive Transformer over continuous tokens. It targets zero-shot styled text images of any length and separates text style from background style — [arXiv](https://arxiv.org/pdf/2503.17074)
- **"Autoregressive Styled Text Image Generation, but Make it Reliable"** (arXiv 2510.23240, Oct 2025): a reliability follow-up to Emuru — [arXiv](https://arxiv.org/pdf/2510.23240)
- **DiffBrush** (arXiv 2508.03256, Aug 2025): diffusion for handwritten **text-line** generation — [arXiv](https://arxiv.org/pdf/2508.03256)
- **Dual Orthogonal Guidance** (arXiv 2508.17017, 2025): diffusion-based handwritten text generation. Known only from a citation; abstract not read — cited in [arXiv 2510.23240](https://arxiv.org/pdf/2510.23240)
- **Eruku** (Zaccagnino et al., 2026): an autoregressive latent-image generator. Known only as a citation in HandwritingAgent — [arXiv 2606.18788](https://arxiv.org/html/2606.18788)
- **"Quo Vadis HTG for HTR?"** (arXiv 2508.09936, Aug 2025): a benchmark of VATr++, DiffusionPen and Emuru as data generators for HTR — [arXiv](https://arxiv.org/html/2508.09936v1)

**HandwritingAgent (Sesay, Yu, Karlsson, arXiv 2606.18788, 17 June 2026)** [fetched]. This is the most directly relevant recent system.
- Writes handwriting as SVG stroke sequences. Each character is an ordered sequence of stroke points on a grid, smoothed with cubic Béziers into `<path>` elements — [arXiv](https://arxiv.org/html/2606.18788)
- Style conditioning: the reference image or strokes are segmented and labelled into a **glyph bank**. Characters that appear in the sample are matched to those exemplars. **Missing characters come from a language-specific backbone glyph bank, adapted to the sample's slant, proportions, spacing, curvature and ligatures.** There is no style-specific training. The backbone LLM is Gemini 3.1 Pro with thinking enabled — [arXiv](https://arxiv.org/html/2606.18788)
- Reported results:
  - IAM Word: SSIM 0.67, FID 88.08, HWD 1.33, all best among compared methods. ΔCER 0.09, against 0.05 for the best baseline.
  - IAM Line: best SSIM (0.77). Emuru has the best FID (19.69 vs 70.92 here) and best HWD (1.33 vs 1.50 here).
  - DiffBrush numbers were quoted from its paper because the authors "could not run its released code reliably."
  — [arXiv](https://arxiv.org/html/2606.18788)
- Code: github.com/Jaykef/HandwritingAgent — [arXiv](https://arxiv.org/html/2606.18788)

**Earlier canonical models** [prior knowledge, not re-verified]
- **GANwriting** (Kang et al., ECCV 2020): a GAN conditioned on **15 word images** of a writer that generates words. Trained on IAM — [arXiv 2003.02567](https://arxiv.org/abs/2003.02567)
- **ScrabbleGAN** (Fogel et al., CVPR 2020): semi-supervised, with each character generated by a separate overlapping filter bank so words can be any length. Style comes from a noise vector, not from reference images. Built for HTR augmentation — [arXiv 2003.10557](https://arxiv.org/abs/2003.10557)
- **HWT, Handwriting Transformers** (Bhunia et al., ICCV 2021): a Transformer encoder over **15 style word images**, with a decoder that cross-attends from per-character queries. The paper reports capturing both global and character-level style — [arXiv 2104.03964](https://arxiv.org/abs/2104.03964)
- **VATr** (Pippi et al., CVPR 2023): content characters are given as "visual archetypes" (GNU Unifont renderings), so rare and unseen characters work better. Style comes from a backbone pre-trained on a large synthetic set of calligraphic fonts. Few-shot (15 words) on IAM — [arXiv 2303.15269](https://arxiv.org/abs/2303.15269)
- **VATr++** (Vanherle et al., 2024): VATr with better training and data handling, plus a proposed standardised evaluation protocol — [arXiv 2402.10798](https://arxiv.org/abs/2402.10798)
- **WordStylist** (Nikolaidou et al., ICDAR 2023): latent diffusion conditioned on a **writer ID** embedding. It works only for writers seen in training, not for new few-shot writers — [arXiv 2303.16576](https://arxiv.org/abs/2303.16576)
- **HiGAN / HiGAN+** (Gan et al., AAAI 2021 / ACM TOG 2022-23): style disentangled from a reference image so arbitrary-length text can be imitated — [HiGAN+ ACM TOG](https://dl.acm.org/doi/10.1145/3550070)
- **HWD, Handwriting Distance** (Pippi et al., BMVC 2023): a handwriting-specific style metric using features from a network trained for writer identification. It is proposed as better suited than FID for comparing writer style — [arXiv 2310.20316](https://arxiv.org/abs/2310.20316)

### Inferences
- In this family, "style" means word-level slant, width, cursive joins and ink appearance, scored with FID or HWD averaged over many writers. Character-shape fidelity (for example, a writer's particular 'g' or '4') is not measured per glyph anywhere. None of these papers shows that per-letter allographs from a specific writer survive generation.
- The project's input mismatches these models: they expect word images as references and produce word images. One-DM says single-letter references perform poorly. A workaround would be to tile the writer's isolated glyphs into pseudo-words as references, then generate single-character or short "words" and crop. This is untested. Unifont content conditioning (VATr, One-DM) may also pull outputs toward a typeset skeleton, which looks like the project's "style collapse."
- These models are trained on IAM or CVL cursive words, not boxed isolated letters. Domain shift to the project's photographed boxes is likely.
- For fine-tuning or inference, One-DM and DiffusionPen are the most practical: code and weights are released, they are latent-diffusion based, and inference fits one GPU. Training from scratch (One-DM used 4×3090 for 700 epochs) is beyond one consumer GPU in a reasonable time.
- HandwritingAgent is evidence that a 2026 state-of-the-art-competitive handwriting imitator can be essentially **retrieval plus geometric adaptation of a glyph bank**. That supports a composition/retrieval baseline for this project, though it relies on a frontier LLM API.

### Gaps
- No paper found that evaluates isolated **Latin** character generation from isolated-character references with any of these models.
- VATr++, HiGAN+ and GANwriting numbers and details were not re-fetched this session.
- Emuru, DiffBrush and Dual Orthogonal Guidance: reference counts, compute and character-level ability were not read (abstract snippets only).
- No per-glyph shape-fidelity metric was found in this literature. HWD and FID are distribution- or writer-level.

## Q2. Online/stroke-based handwriting (Graves, DeepWriting, DSD/BRUSH, SDT, DiffInk) and offline-to-online recovery

### Takeaway
Pen-trajectory models are the only family with a published **character-level, few-reference writer imitation** setting:
- SDT (CVPR 2023) disentangles writer-level and character-level style for single online characters.
- Kotani et al.'s DSD (ECCV 2020) does Latin characters and words from a few online samples.

These models need online ink, both for training and as style references. The project has photos, so it would need offline-to-online recovery. InkSight (Google, TMLR) is the strongest recent recovery result: 87% valid tracings on HierText.

### Cited Findings
- **SDT** (Dai et al., CVPR 2023, pp. 5977-5986, arXiv 2303.14736): a dual-head style encoder disentangles **writer-level and character-level** style. Two complementary contrastive objectives capture what the references share and the detailed pattern of each sample. Output is stylised **online handwritten characters**. Reports clear gains on Chinese online handwriting and claims effectiveness "on various language scripts." Code at github.com/dailenson/SDT — [CVPR open access](https://openaccess.thecvf.com/content/CVPR2023/html/Dai_Disentangling_Writer_and_Character_Styles_for_Handwriting_Generation_CVPR_2023_paper.html); [arXiv](https://arxiv.org/abs/2303.14736) [search]
- The SDT paper's related work notes that MC-GAN generates a whole consistent letter set from a few examples, and points to the BRUSH dataset work for Latin handwriting generation — [arXiv PDF](https://arxiv.org/pdf/2303.14736) [search]
- **DSD, "Generating Handwriting via Decoupled Style Descriptors"** (Kotani, Tellex, Tompkin, ECCV 2020): introduces the **BRUSH** dataset (online English handwriting) and represents style as writer-specific and character-specific descriptors, so a writer's handwriting can be imitated from a few samples — [ar5iv 2008.11354](https://ar5iv.arxiv.org/html/2008.11354) [search; details prior knowledge]
- **Graves** (2013): an RNN with mixture-density outputs that synthesises continuous handwriting trajectories. Style priming with a sample from the writer is its form of conditioning — [arXiv 1308.0850](https://arxiv.org/abs/1308.0850) [prior knowledge]; cited in [HandwritingAgent](https://arxiv.org/html/2606.18788)
- **DeepWriting** (Aksan, Pece, Hilliges, CHI 2018): a conditional VRNN that separates content from style in digital ink, so text can be rewritten in a user's style — [arXiv 1801.08379](https://arxiv.org/abs/1801.08379) [prior knowledge]
- **DiffInk** (arXiv 2509.23624, Sept 2025, accepted ICLR 2026): InkVAE (a sequential VAE with OCR and style-classification regularisers) plus InkDiT (a latent diffusion Transformer). Claimed as the first latent diffusion Transformer for **full-line** online handwriting, conditioned on target text and reference styles. The abstract has no numbers or code link — [arXiv](https://arxiv.org/abs/2509.23624) [fetched]
- **Offline-to-online recovery:**
  - **InkSight** (arXiv 2402.05804, Feb 2024, TMLR) does "derendering" of photographed handwriting into digital ink, built on vision-language models with reading and writing priors. On HierText, human raters judged **87% of outputs valid tracings and 67% human-like pen trajectories** — [arXiv](https://arxiv.org/abs/2402.05804) [fetched]. The abstract page does not confirm a weights release.
  - An end-to-end encoder-decoder for character-level trajectory recovery (2018) — [arXiv 1801.07211](https://arxiv.org/pdf/1801.07211)
  - **TRACE** (2021): line-level, differentiable recovery, with the first recovery benchmarks on IAM-On and IAM-Off — [arXiv 2105.11559](https://arxiv.org/pdf/2105.11559)
  - **Writing-order recovery for long static handwriting** (Diaz et al., June 2024) works on thinned trajectories and notes there is still no general approach for complex, long handwriting — [summary](https://arxiv.deeppaper.ai/papers/2406.03194v1)
  - [search for all three]

### Inferences
- The project's input is isolated Latin letters, and these are the *easiest* case for offline-to-online recovery. One letter is usually 1-3 strokes, and skeleton tracing plus heuristic ordering is often enough. Even a perfectly recovered trajectory then needs re-rendering with the writer's pen (width profile).
- A pipeline of photo → skeleton or InkSight-style ink → SDT/DSD-style few-shot generator → stroke rendering → outline → OTF is plausible. However, SDT is trained mostly on Chinese online data. A Latin model needs online Latin data such as BRUSH, IAM-On or Deepwriting.
- Stroke models tend to keep **shape** (topology, stroke order, proportions) rather than texture, because they model geometry directly. This is the property the project lacks. That is a reasoned inference, not a measured comparison.

### Gaps
- No published system found that does few-shot **Latin isolated-glyph** generation from *offline photographed* references with stroke models end to end.
- Not verified: InkSight weights availability or licence, SDT Latin results, or compute.
- No numbers found for DiffInk or diffusion-over-trajectory character models.

## Q3. Vector font generation (DeepSVG → DeepVecFont(-v2) → DualVector → VecFusion → 2026 LLM-based)

### Takeaway
Vector-font models are trained on typeset font corpora: Google Fonts, Envato, and the DeepVecFont dataset. They demonstrate few-shot **font** style transfer on Latin and Chinese. Reported styles are typographic families, and no source checked here tests personal handwriting. They would probably share the project's style-collapse problem, plus vector-specific artefacts. Their best use here is likely downstream: vectorising raster or stroke outputs cleanly.

### Cited Findings
- **DeepSVG** (Carlier et al., NeurIPS 2020): a hierarchical Transformer VAE over SVG commands — [arXiv 2007.11301](https://arxiv.org/abs/2007.11301) [prior knowledge]; cited in [HandwritingAgent](https://arxiv.org/html/2606.18788)
- **DeepVecFont** (Wang & Lian, SIGGRAPH Asia 2021): dual-modality (image plus sequence) learning for vector fonts — [arXiv 2110.06688](https://arxiv.org/pdf/2110.06688) [search]
- **DeepVecFont-v2** (CVPR 2023): moves from RNNs to Transformers and adds a relaxation representation of outlines, auxiliary sampled points and a Transformer self-refinement module to remove artefacts. Evaluated on English and Chinese fonts. Code at github.com/yizhiwang96/deepvecfont-v2 — [arXiv](https://arxiv.org/abs/2303.14585) [fetched]. The abstract does not state the reference-glyph count.
- **VecFusion** (Thamizharasan et al., CVPR 2024): cascaded diffusion (raster then vector). It is compared with DeepVecFont-v2 and DualVector on **few-shot style transfer** using the DeepVecFont-v2 dataset — [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Thamizharasan_VecFusion_Vector_Font_Generation_with_Diffusion_CVPR_2024_paper.pdf) [search]
- **DualVector** (Liu et al., CVPR 2023): a dual-part (positive and negative shape) representation, learned from glyph images without vector supervision — [arXiv 2305.10462](https://arxiv.org/abs/2305.10462) [prior knowledge]
- **VecFontSDF** (CVPR 2023): reconstructs and synthesises vector fonts through signed distance functions — [arXiv 2303.12675](https://arxiv.org/pdf/2303.12675) [search]
- **VecGlypher** (arXiv 2602.21461, Feb 2026, accepted CVPR 2026): a multimodal LLM that autoregressively emits SVG path tokens.
  - Training: continued pre-training on 39K noisy Envato fonts, then post-training on 2.5K expert-annotated Google Fonts.
  - Inputs: a style prompt, optional reference glyph images, and the target character.
  - Claims "marked gains" over DeepVecFont-v2 and DualVector for image-referenced generation. No numbers on the abstract page, and no handwriting styles mentioned.
  — [arXiv](https://arxiv.org/abs/2602.21461) [fetched]
- **DesigNet** (arXiv 2604.06494, 2026): one-shot and few-shot results on Latin and Chinese typefaces, compared with DualVector and DeepVecFont-v2 fine-tuned from their official checkpoints — [arXiv](https://arxiv.org/html/2604.06494v1) [search]
- **VecFontLLM** (arXiv 2607.17251): Chinese vector fonts. Seen only as a search title and not verified — [arXiv](https://arxiv.org/pdf/2607.17251) [search]
- **HFH-Font** (arXiv 2410.06488, 2024): few-shot *Chinese* font synthesis at higher resolution, from raster to vector — [arXiv](https://arxiv.org/pdf/2410.06488) [search]

### Inferences
- These models learn a prior over typeface outlines. The project's AdaIN model already collapses toward typeset shapes from a Google Fonts prior, so the same is expected here, and possibly worse: outline-token models are strongly regularised by typographic conventions. This is inference only; no handwriting evaluation was found.
- A LLM-scale model like VecGlypher is not trainable on one consumer GPU. DeepVecFont-v2 and DualVector have released checkpoints and could be fine-tuned locally, but on fonts rather than handwriting.
- More practical vector routes for handwriting: trace raster outputs (potrace-style), or render stroke skeletons with a pen-width profile and convert to outlines.

### Gaps
- Exact reference counts (DeepVecFont-v2's and DualVector's few-shot settings), training compute and handwriting results were not verified.
- VecGlypher code release not confirmed.

## Q4. Composition/retrieval: building missing glyphs from the writer's own strokes or parts; commercial services; nearest-font baseline

### Takeaway
Composition and retrieval have a published lineage: skeleton-and-stroke reuse (2017, Chinese), rule-based stroke reuse for Gujarati (2024), and component-memory few-shot font models such as DM-Font. In 2026, HandwritingAgent showed that a glyph bank of the writer's own exemplars plus adapted fallback glyphs can be competitive with generative handwriting models on IAM. No Latin-specific published evaluation of "u from n, d from a plus ascender" composition was found. The commercial services found require the user to write every glyph. Calligraphr's randomised variants and ligatures are reuse of what the user drew, not generation.

### Cited Findings
- **HandwritingAgent** (2026): characters present in the sample are reused as exemplars. Missing characters are pulled from a backbone glyph bank and **adapted to the sample's slant, proportions, spacing, curvature and ligatures**. Reported IAM-Word SSIM 0.67, FID 88.08 and HWD 1.33 were best in its comparison — [arXiv](https://arxiv.org/html/2606.18788) [fetched]
- Non-learned skeleton-based stroke reuse (arXiv 1701.05703, 2017): strokes are extracted from a few sample characters via their skeletons and placed on a transformed skeleton of the target character. Chinese-focused — [arXiv](https://arxiv.org/pdf/1701.05703v1.pdf) [search]
- Gujarati stroke reuse (arXiv 2404.03277, 2024): strokes are extracted and labelled from segmented characters, then reused by rules to generate other characters — [arXiv](https://arxiv.org/pdf/2404.03277v1) [search]
- **JointFontGAN**: jointly generates skeleton and glyph from very few font samples — [NSF PAR](https://par.nsf.gov/servlets/purl/10199594) [search]
- **DM-Font** (Cha et al., ECCV 2020): a component-wise memory for few-shot generation in compositional scripts (Korean, Chinese) — [arXiv 2005.10510](https://www.arxiv-vanity.com/papers/2005.10510/) [search]
- "Handwritten Chinese Font Generation with Collaborative Stroke Refinement" (2019) — [arXiv 1904.13268](https://arxiv.org/pdf/1904.13268) [search]
- **MC-GAN** (Azadi et al., CVPR 2018): generates all 26 letters of a glyph set consistently from a few observed letters, with a glyph network plus an ornamentation network — [arXiv 1712.00516](https://arxiv.org/abs/1712.00516) [prior knowledge]
- **Calligraphr** (commercial):
  - Variants come from uploading more than one template to a font and ticking "randomize." The FAQ recommends extra variants for frequent letters such as 'a' and 'e'.
  - Ligatures are a Pro feature and need OTF output.
  - A "Minimal English" template exists, but its character count was not found.
  — [Calligraphr FAQ](https://www.calligraphr.com/en/docs/faq/); [features](https://www.calligraphr.com/en/features/); [blog walkthrough](https://scrapbookcampus.com/2024/10/featured-resource-calligraphr/) [search]

### Inferences
- Latin lowercase has strong part sharing in most hands:
  - n → h (longer stem), n → m (extra arch), n → u (rotated 180°), n → r (truncated arch)
  - o, c and e share a bowl
  - a + ascender → d; q from a, d or g; b ↔ d and p ↔ q (mirror plus small adjustments)
  - l, i and j share the stem
  - v → w, v → y
  - Digits are weaker: 6 ↔ 9 by rotation.

  Composition from the writer's own strokes therefore *guarantees* their stroke texture, width and much of their letterform idiosyncrasy. That is exactly what the raster model loses. The risk is visible "pasted" symmetry, and letters with no donors (k, x, z, s, g).
- A practical hybrid ranked by expected shape fidelity: (1) the writer's own glyph, (2) composition or affine transform from the writer's parts, (3) a generated glyph only for letters without donors. Generated glyphs should be corrected toward the writer's measured slant, x-height, width and stroke width, which is the same adaptation HandwritingAgent describes.
- **Nearest-font retrieval** (pick the handwriting-style Google Font closest to the 30 samples by a learned embedding, then rescale and reweight) is a cheap, strong baseline for the "missing letter" set. It is expected to beat style-collapsed generation on legibility, but not on personal shape. No published evaluation of this baseline for handwriting fonts was found.
- Skeletonise the writer's strokes (e.g. Zhang-Suen or medial axis), estimate a width profile per stroke, and re-stroke composed or generated skeletons with that profile. This gives consistent "pen" rendering across real, composed and generated glyphs, and converts cleanly to OTF outlines.

### Gaps
- No published Latin handwriting-font evaluation of part composition was found, nor any evaluation of commercial glyph-reuse practice.
- MyScriptFont and other services were not checked this session. The understanding that they need every glyph handwritten in a template is unverified.
- Whether Calligraphr or similar services have since added AI completion of missing glyphs was not verified.

## Q5. Which approaches could realistically produce same-hand glyph-level output, and their data needs

### Takeaway
Ranked by expected glyph-shape fidelity for 30 isolated Latin samples on one consumer GPU: composition/retrieval from the writer's own glyphs and parts first, then few-shot stroke models over recovered trajectories, then word-level styled handwriting text generation (HTG) diffusion (One-DM, DiffusionPen) adapted to characters, and last vector-font models. Only the first has direct (2026) evidence of competitive style imitation without training. The rest are inference from adjacent results.

### Cited Findings
- One-DM needs about one reference word longer than four letters, and single-letter references perform poorly. Training used 4×3090 — [arXiv HTML](https://arxiv.org/html/2409.04004v2)
- DiffusionPen needs 5 reference word images, and code plus weights are released — [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11492_ECCV_2024_paper.php); [HF](https://huggingface.co/konnik/DiffusionPen)
- SDT does per-character online generation with writer and character style disentangled. Code is released — [CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Dai_Disentangling_Writer_and_Character_Styles_for_Handwriting_Generation_CVPR_2023_paper.html)
- InkSight turns photos into ink, with 87% valid tracings on HierText — [arXiv](https://arxiv.org/abs/2402.05804)
- HandwritingAgent combines a glyph bank with adaptation and needs no style training — [arXiv](https://arxiv.org/html/2606.18788)
- VecGlypher (39K Envato plus 2.5K Google Fonts) and other vector models are trained only on fonts — [arXiv](https://arxiv.org/abs/2602.21461)

### Inferences

| Approach | Data needed | Compute | Expected shape fidelity |
|---|---|---|---|
| Composition, retrieval and adaptation | the 30 glyphs plus a handcrafted Latin part-donor table | CPU | high for donor-able letters, none for the rest |
| Stroke-model route (SDT/DSD-like) | online Latin training data (BRUSH, IAM-On); photo-to-ink recovery | training feasible on 1 GPU at character scale, magnitude unverified | potentially high (geometry-level) |
| HTG diffusion (One-DM, DiffusionPen) | IAM/CVL pretrained weights; pseudo-word references built from the writer's glyphs | inference and fine-tuning on 1 GPU | unknown at glyph level; risk of a Unifont/typeset pull |
| Vector font models | font corpora | fine-tune on 1 GPU (DVF-v2, DualVector) | low for personal shape (font prior) |

- Evaluation advice drawn from the literature: HWD is the handwriting-specific writer-style metric, but it is computed at word level. Per-glyph checks should be added, for example a writer-ID classifier on single glyphs, or a held-out-glyph reconstruction comparison against the writer's real glyph. The project's ten photographed writers must stay a held-out test only.

### Gaps
- No head-to-head study was found that compares composition, stroke models and raster diffusion on isolated Latin glyph sets from one writer.
- Single-GPU training times for SDT, DSD and DiffusionPen fine-tuning were not found.
