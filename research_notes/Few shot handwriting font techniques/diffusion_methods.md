# Diffusion-based few-shot font/glyph generation and parameter-efficient style adaptation (2022-2026)

Scope note: the notes were compiled on 2026-11-06 from about 20 searches and fetches. Paper PDFs could not always be parsed. Figures marked "reported" come from the paper, repo or model card. Anything under "Inferences" is my reasoning and has not been measured. Project context: about 30 photographed Latin letters in, the other ~45 of 75 glyphs out, at 128x128. The current CNN shows "style collapse" toward typeset shapes. The team has one consumer GPU with 12-24 GB.

## Q1. Which diffusion font-generation methods exist, how do they inject style, what data do they use, and is there evidence on idiosyncratic or large-variation styles?

### Takeaway
Diffusion font generation is still mostly Chinese/Korean, with 80-96 px glyphs. The flagship methods are Diff-Font (2022/IJCV 2024) and FontDiffuser (AAAI 2024). Both beat GAN baselines mainly on complex characters and unseen fonts. Neither reports quantitative Latin or handwriting results. The closest evidence for idiosyncratic Latin hands comes from handwritten-text-generation (HTG) diffusion models trained on IAM word images (One-DM, DiffusionPen), not from font papers. Methods that drop the source content glyph (IF-Font, autoregressive) claim the biggest gains when the target style is far from the training fonts.

### Cited Findings
**Diff-Font (He, Chen et al.; arXiv Dec 2022, v3 May 2023; IJCV 2024)**
- It is described as the first diffusion model for font generation. It is one-shot: character content comes from predefined embedding tokens, not a source image, and style comes from one reference image — [arXiv 2212.05895](https://arxiv.org/abs/2212.05895v3)
- Style injection: a style encoder pretrained in DG-Font is frozen. Its features go through an MLP and are concatenated with the content embedding and an optional stroke/component embedding into a 512-d condition vector — [ar5iv 2212.05895](https://ar5iv.labs.arxiv.org/html/2212.05895)
- Stroke-count encoding is used for Chinese and component encoding for Korean. Both replace StrokeGAN's one-bit encoding — [arXiv 2212.05895](https://arxiv.org/abs/2212.05895v3)
- Data: 410 Chinese fonts with 6,625 characters each, at 80x80. The small set has 400 training fonts and 800 characters; the large set uses all 6,625 characters. Korean: 201 fonts with 2,350 characters. Latin/Greek appear only as qualitative examples — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.05895)
- Reported FID: small set 27.30 against 35.20 (FUNIT), 37.15 (MX-Font) and 43.44 (DG-Font); large set 16.20 against 26.64-28.63. Korean FID 10.69 against 43.36 (DG-Font). In a human study it got 48.75% of 640 votes — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.05895)
- Large style variation (difficult styles, four art fonts) is shown only qualitatively, with fewer structural distortions than the GANs. The paper says extremely intricate characters or uncommon styles still fail — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.05895)

**FontDiffuser (Yang et al., AAAI 2024; arXiv 2312.12142)**
- MCA (multi-scale content aggregation): features from three content-encoder layers are concatenated into the UNet and reweighted by channel attention. Cross-attention injects the style embedding, with style as K/V and content features as Q — [ar5iv 2312.12142](https://ar5iv.labs.arxiv.org/html/2312.12142)
- RSI (reference-structure interaction): reference structure maps cross-attend with UNet features to predict offsets for a deformable convolution on the skip connections, which handles structural changes — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- SCR (style contrastive refinement): a VGG-based style extractor with a contrastive loss on the predicted x0 (τ=0.07, 16 negatives). It is used only in training phase 2 and is not a conditioning path — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Data: 424 Chinese fonts (400 seen, 800 seen characters) at 96x96. Complexity buckets are easy (6-10 strokes), medium (11-20) and hard (≥21) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Reported results (FID/SSIM/LPIPS/L1), unseen fonts and unseen characters, average: FontDiffuser 8.54/0.421/0.150/0.387 against MX-Font FID 10.77. On hard characters, FID 13.12 against 16.40. On seen fonts, easy and medium characters, CG-GAN has the better FID (8.23 vs 8.51; 9.11 vs 9.46). Diff-Font FID is 12.08 in the same table. The baselines are the GANs FUNIT, LF-Font, DG-Font, MX-Font, Fs-Font, CG-GAN and CF-Font — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Ablation, unseen fonts and characters: the baseline FID of 8.12 got *worse* (8.54) with all three modules added, though SSIM, LPIPS and L1 improved — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Chinese→Korean cross-lingual results are qualitative only. The repo shows no Latin results — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142); [GitHub](https://github.com/yeungchenwa/FontDiffuser)

**IF-Font (Chen, Ke, Guo; NeurIPS 2024)**
- It replaces the source glyph with an Ideographic Description Sequence (IDS), quantizes reference glyphs into tokens and models target-token distributions. This avoids content/style disentanglement. It is token-based, and the snippets do not say whether it is a diffusion model — [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/94397); [mlanthology](https://mlanthology.org/neurips/2024/chen2024neurips-iffont)
- It claims large gains in one-shot and few-shot settings, "particularly when the target styles differ significantly from the training font styles". Stated limitation: fancy or irregular styles, decorations, and extremely flat or narrow layouts — [mlanthology](https://mlanthology.org/neurips/2024/chen2024neurips-iffont)

**DA-Font (ACM MM 2025; arXiv 2509.16632, Sep 2025)**
- It adds a Dual-Attention Hybrid Module (component attention plus relation attention) and corner-consistency and elastic-mesh feature losses for geometric alignment. Code is at github.com/wrchen2001/DA-Font. The abstract gives no dataset or metric numbers — [arXiv 2509.16632](https://arxiv.org/abs/2509.16632)
- A search snippet says DA-Font is built on a pre-trained unconditional denoising diffusion model. I did not verify this in the full text — [arXiv PDF](https://arxiv.org/pdf/2509.16632)

**HFH-Font (Li & Lian; SIGGRAPH Asia 2024 / TOG; arXiv 2410.06488)**
- Diffusion with component-aware conditioning and multi-level style learning that adapts to the number of references. Score-distillation gives **1-step inference**, and a style-guided super-resolution module outputs 1024x1024 or higher for vectorization. The authors claim the first large-scale Chinese vector fonts comparable to professional ones. Code: github.com/grovessss/HFH-Font — [arXiv 2410.06488](https://arxiv.org/abs/2410.06488)

**GRIF-DM (Kang et al., 2024; arXiv 2408.07259)**
- A Latin-alphabet diffusion model that takes one letter plus impression keywords, using dual cross-attention between letter features and keywords. It reports better FID and Intra-FID than GAN methods. Code: github.com/leitro/GRIF-DM — [bohrium summary of 2408.07259](https://bohrium.dp.tech/paper/arxiv/2408.07259); [ar5iv](https://ar5iv.labs.arxiv.org/html/2408.07259)
- Related Latin work, font style interpolation with diffusion (Kondo et al.), compares against FANnet, described as state of the art for one-shot Latin font generation — [arXiv 2402.14311](https://arxiv.org/html/2402.14311v1)

**DP-Font (IJCAI 2024)**
- Chinese calligraphy generation using diffusion plus a physics-informed neural network, conditioned on content, stroke order and style — [IJCAI 2024](https://www.ijcai.org/proceedings/2024/bibtex/0863)

**FontStudio (MSRA, ECCV 2024; arXiv 2406.08392)**
- It generates font *effects* (texture inside a glyph mask), not glyph shapes. Shape-adaptive diffusion is conditioned on a segmentation mask, and training-free effect transfer from a reference letter keeps the letters consistent. It claims a user-study win against Adobe Firefly — [arXiv 2406.08392](https://arxiv.org/abs/2406.08392); [project](https://font-studio.github.io)

**2025-2026 successors and adjacent work**
- GAR-Font, "Beyond Patches" (CVPR 2026; arXiv 2601.01593, v2 May 2026), is autoregressive, not diffusion. It has a global-aware tokenizer, a multimodal text+image style encoder with a lightweight language-style adapter, and post-refinement. It claims to beat existing few-shot methods but gives no numbers in the abstract, and no code is linked — [arXiv 2601.01593](https://arxiv.org/abs/2601.01593)
- Others found by title only, not examined: DRG-Font (arXiv 2604.13797), Few-Part-Shot Font Generation (arXiv 2509.10006), FontCrafter (arXiv 2603.22054, artistic element-driven fonts), "Few-shot font generation via denoising diffusion and component-level fine-grained style" (Expert Systems with Applications, 2025), and CalliffusionV2 (arXiv 2410.03787, personalized calligraphy) — [search results: DRG-Font](https://arxiv.org/pdf/2604.13797), [Few-Part-Shot](https://arxiv.org/abs/2509.10006), [FontCrafter](https://arxiv.org/pdf/2603.22054), [ESWA 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425026041), [CalliffusionV2](https://arxiv.org/pdf/2410.03787)
- An Aug 2025 survey of Chinese font generation sorts methods into many-shot and few-shot and covers both GANs and diffusion. I found no Latin- or diffusion-specific survey — [arXiv 2508.06900](https://arxiv.org/abs/2508.06900v1)

**Handwriting diffusion (HTG): the closest evidence for writer idiosyncrasy in Latin script**
- DiffusionPen (ECCV 2024) is a 5-shot latent diffusion model. It uses a continuous style embedding from k=5 reference samples (metric learning plus classification) instead of a writer index, on IAM with Stable Diffusion v1.5 VAE and DDIM sampling — [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf); [GitHub](https://github.com/koninik/DiffusionPen)
- One-DM (ECCV 2024) imitates a writer from a single reference image, using high-frequency components of the reference and a style-enhanced module. It reports beating methods that use more than 10 samples. It covers English, Chinese and Japanese, but the first code release is English only — [arXiv 2409.04004](https://arxiv.org/pdf/2409.04004v1); [GitHub](https://github.com/dailenson/One-DM)
- A 2026 preprint applies DiffusionPen to Ukrainian Cyrillic and reports that writer-aware few-shot latent diffusion generalizes beyond Latin — [arXiv 2605.27487](https://arxiv.org/html/2605.27487v1)
- DiffBrush, from the One-DM authors, was accepted to ICCV 2025 according to the One-DM repo news — [GitHub](https://github.com/dailenson/One-DM)

### Inferences
- FontDiffuser and Diff-Font take letter identity from a source glyph or a token. FontDiffuser in particular aggregates *source-glyph* content features at several scales. That design pushes outputs toward the source skeleton, so I would expect the same "textbook k" bias for topology changes such as a two-stroke k. RSI's deformable offsets model geometric warps, not stroke-topology changes. No paper tests this on handwriting.
- Most "large style variation" evidence is Chinese, where variation is mostly stroke texture and proportion within a fixed component topology. It may not carry over to Latin handwriting, where the letter construction itself varies by writer.
- HTG models (One-DM, DiffusionPen) are trained on many real writers (IAM). They are the only diffusion family here whose training distribution contains idiosyncratic hands, which makes them a better source of writer-style priors than Google-Fonts-only training. They generate words, not isolated 128 px glyphs, and IAM is low resolution (the One-DM config path suggests "IAM64").
- The project's measured failure (content image used only for letter identity, skips unused) fits a design where letter identity comes from a token, as in Diff-Font or IF-Font. Shape would then come from the style or reference path, and the content glyph would add no topology.

### Gaps
- I found no quantitative diffusion font result on handwritten Latin glyphs, and no evaluation of idiosyncratic letter *construction*, such as stroke count or topology.
- VQ-Font (ICCV 2023) and CF-Font (CVPR 2023): no sources were fetched. CF-Font appears as a GAN baseline in FontDiffuser's table ([ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)). My recollection that VQ-Font is VQGAN-based and not diffusion is unverified here.
- The original Calliffusion (2023) paper was not found under that name. Only the CalliffusionV2 title surfaced.
- DA-Font's datasets and metrics, and IF-Font's architecture details and numbers, were not retrieved.

## Q2. Training compute, inference speed, model size, and feasibility at 128 px on one consumer GPU

### Takeaway
FontDiffuser was trained on a single RTX 3090 (24 GB) at 96 px with batch 16: 440k steps plus 30k steps. It samples in 20 DPM-Solver++ steps with CFG. Training a comparable pixel-space model at 128 px on one 24 GB card is in the published regime. Wall-clock time and parameter counts are mostly unreported.

### Cited Findings
- FontDiffuser: 1x RTX 3090, AdamW. Phase 1: 440,000 steps, batch 16, lr 1e-4, loss MSE + content perceptual (λ=0.01) + offset (λ=0.5). Phase 2: 30,000 steps, batch 16, lr 1e-5, plus style contrastive loss. Condition dropout 0.1. Total training time and parameter count are **not reported** — [ar5iv 2312.12142](https://ar5iv.labs.arxiv.org/html/2312.12142)
- FontDiffuser inference: DPM-Solver++, 20 steps, CFG scale 7.5. Time per glyph is not reported. The paper names slower sampling than GANs as a limitation — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Diff-Font: small set batch 24, 300k conditional plus 300k fine-tuning iterations; large set batch 64, 420k plus 380k. UNet with 128 channels, multipliers 1-4, attention at 40/20/10, 1,000 diffusion steps. Sampling: DDIM, 25 steps, guidance 3/3. GPU, time and size are not reported, and the paper notes low inference efficiency — [ar5iv 2212.05895](https://ar5iv.labs.arxiv.org/html/2212.05895)
- HFH-Font distils to 1-step inference — [arXiv 2410.06488](https://arxiv.org/abs/2410.06488)
- One-DM's training script uses 4 GPUs. Epochs and batch size sit in configs I did not see — [GitHub](https://github.com/dailenson/One-DM)
- FontAdapter, an SD3 adapter: customization in "11 seconds on a single RTX 3090", 30 denoising steps, training batch 48. Training GPU-hours are not stated — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)
- The FontAdapter paper cites about 40 min on 4x RTX 3090 for LoRA fine-tuning SD3 on one font — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)

### Inferences
- A full font costs 75 glyphs x 20 steps x 2 (CFG) ≈ 3,000 UNet evaluations. A small pixel-space UNet at 96-128 px can batch all 75 glyphs, so I would expect seconds to tens of seconds per font on a 3090-class GPU. This is unmeasured.
- 128 px is (128/96)² ≈ 1.8x the pixels of FontDiffuser's 96 px. With batch 16 on a 24 GB card it should still fit; on 12 GB, smaller batch, gradient accumulation or mixed precision would probably be needed. At roughly 470k steps, training would plausibly take days on one GPU; no paper reports hours.
- A latent-space alternative (SD VAE at f=8 gives 16x16 latents for 128 px) is cheaper per step. Fine strokes may suffer in VAE reconstruction at that size; this is untested.

### Gaps
- None of the font papers retrieved report parameter counts or GPU-hours.
- I found no measured per-glyph sampling latency for FontDiffuser or Diff-Font.

## Q3. Code and weights availability, licenses, maintenance

### Takeaway
Most methods have public code. FontDiffuser has released weights but a **non-commercial research-only** license. One-DM and DiffusionPen are MIT-licensed with weights. Chinese-trained checkpoints are of limited direct use for Latin glyphs.

### Cited Findings
- FontDiffuser: phase-1 and phase-2 training scripts, `unet.pth`, `content_encoder.pth` and `style_encoder.pth` weights plus SCR weights on Google Drive and Baidu. "Only for non-commercial research purposes"; commercial use requires contacting Prof. Lianwen Jin. The last news item is 2024-01-27 (phase-2 release), and SCR pretraining code is still "coming soon". Sampling from strings needs a TTF containing Chinese characters — [GitHub](https://github.com/yeungchenwa/FontDiffuser)
- Diff-Font: the code, pretrained models and datasets are stated to be at github.com/Hxyz-123/Font-diff — [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.05895)
- One-DM: MIT. Checkpoint plus OCR and ResNet-18 auxiliary models released 2024-10-24. Latest news 2025-06-26 — [GitHub](https://github.com/dailenson/One-DM)
- DiffusionPen: MIT. IAM weights and style-encoder weights on Hugging Face — [GitHub](https://github.com/koninik/DiffusionPen)
- IF-Font: github.com/Stareven233/IF-Font — [search summary of NeurIPS listing](https://mlanthology.org/neurips/2024/chen2024neurips-iffont)
- DA-Font: github.com/wrchen2001/DA-Font — [arXiv](https://arxiv.org/abs/2509.16632)
- HFH-Font: github.com/grovessss/HFH-Font — [arXiv](https://arxiv.org/abs/2410.06488)
- GRIF-DM: github.com/leitro/GRIF-DM — [bohrium](https://bohrium.dp.tech/paper/arxiv/2408.07259)
- FontAdapter: a project page exists (fontadapter.github.io), but the paper does not state that code or weights are released — [arXiv](https://arxiv.org/html/2506.05843v1)
- GAR-Font: no code linked — [arXiv](https://arxiv.org/abs/2601.01593)

### Inferences
- A public repo is allowed to contain FontDiffuser-derived code, but its non-commercial license matters if the project is ever distributed commercially.
- For this project the reusable part is the training code, retrained on Latin fonts. Chinese weights are unlikely to help, beyond perhaps initialization.

### Gaps
- Licenses for Diff-Font, IF-Font, DA-Font, HFH-Font and GRIF-DM were not checked, and neither was their commit activity in 2025-2026.

## Q4. Parameter-efficient few-shot adaptation (LoRA, DreamBooth, textual inversion, adapters, IP-Adapter) with 10-30 examples

### Takeaway
No retrieved paper directly measures LoRA or DreamBooth overfitting on *unseen glyphs* from 10-30 examples. General evidence says DreamBooth at 10-30 images memorizes, and LoRA reduces copying. The trend for fonts (FontAdapter, 2025) is to avoid per-font tuning and use an IP-Adapter-style encoder trained across many fonts. That agrees with the project's own finding that per-writer fine-tuning on 24 letters overfits.

### Cited Findings
- With only 10-30 reference images, DreamBooth is prone to overfitting and "essentially memorizes" the training data. Adding LoRA (DB^L) reduces the tendency to produce training copies. This is for character art, not fonts (arXiv 2510.09475; ESWA 2026) — [arXiv 2510.09475](https://arxiv.org/pdf/2510.09475); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426001405)
- FontAdapter (KAIST/SNU, arXiv 2506.05843, Jun 2025) is IP-Adapter-style on SD3. Reference-glyph tokens go through a Resampler into image-image attention layers that run in parallel with the MM-DiT text attention; only these are trained. It is trained on about 1,500 fonts with English words and conditions on a single reference glyph image, with no per-font fine-tuning. Reported accuracy: WordAcc 0.5303, NED 0.7904 — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)
- FontAdapter's authors frame per-font fine-tuning as taking tens of minutes, about 40 min on 4x RTX 3090 for SD3 LoRA. They used 50 LoRA "expert" models only to make training data and report no quality or overfitting comparison — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)
- GAR-Font (CVPR 2026) uses a lightweight language-style adapter. A search snippet says it updates LoRA layers of its Transformer generator with a few reference glyphs during few-shot adaptation, but the abstract does not mention LoRA — [arXiv 2601.01593](https://arxiv.org/abs/2601.01593) (conflicting or unverified)
- An ESWA 2025 paper builds few-shot font generation on DreamBooth to capture character detail beyond text prompts. Results on unseen characters were not retrieved — [ScienceDirect search result](https://www.sciencedirect.com/science/article/abs/pii/S0957417425026041)
- An AAAI paper on Chinese character art uses LoRAs trained on small datasets with user handwriting fed through ComfyUI img2img. Sizes were not retrieved — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/35381/37536)

### Inferences
- With 24-30 letters, each letter appears once. Per-writer LoRA or DreamBooth therefore learns "this image goes with this letter", not "this writer's construction rules". That matches the project's measurement, and the general DreamBooth finding suggests it is not specific to the CNN.
- If adaptation is used, the usual regularizers are low rank, few steps, a small learning rate and prior preservation using rendered fonts as the class prior. Hold out a few of the writer's letters for early stopping. These are standard DreamBooth/LoRA practices with no font-specific evidence found. Adapting only the style or reference path, not the content or decoder path, is the analogue of the FontAdapter approach.
- An encoder-based, zero-tuning adapter (IP-Adapter or FontAdapter style) trained on many *handwritten* writers is the better-supported direction. Training only on Google Fonts is a likely contributor to the collapse toward typeset shapes.

### Gaps
- No textual-inversion results for glyph style were found.
- No ablation of LoRA rank or steps against unseen-glyph quality for fonts or handwriting was found.

## Q5. Do large pretrained T2I models (SD/SDXL/FLUX, 2025-2026) with LoRA give usable, consistent single glyphs? Glyph-specific models?

### Takeaway
The working community approach is to generate a whole **font atlas** in one image, so all letters share one sampling pass and stay consistent. It does not generate glyphs one by one. Ref2Font, a LoRA on FLUX.2 Klein 9B, turns an "Aa" sample into a 1280x1280 Latin/Cyrillic atlas plus a TTF script. Scene-text models (GlyphControl, AnyText, TextDiffuser) target text in images, and their small-text detail degrades. No independent benchmark of handwriting fidelity exists for any of these.

### Cited Findings
- Ref2Font (Civitai, 2026) is a LoRA for FLUX.2 Klein 9B. Input: an image of "Aa" (Latin) or "Аа" (Cyrillic). Output: a 1280x1280 atlas of A-Z, a-z, 0-9 and punctuation, driven by a fixed prompt that spells out the charset. An included script converts the grid to .ttf. V2 reports a more stable baseline, and a V3 exists — [Civitai](https://civitai.com/models/2361340/ref2font-flux2-klein-9b-fonts-lora); [comfy.icu](https://comfy.icu/models/2361340/Ref2Font-FLUX-2-Klein-9B-Fonts-LoRA/2655722/Ref2FontV1)
- 4lph4bet-next_v040 is an SD 1.5 fine-tune that generates an 88-character grid in a consistent style. Post-processing goes through Calligraphr with upscaling to 4096x4096. The amount of training data is not stated — [Hugging Face](https://huggingface.co/414design/4lph4bet-next_v040)
- fofr/flux-handwriting (trigger "HWRIT") generates handwritten *text samples*, neat or messy, not consistent glyph sets — [Hugging Face](https://huggingface.co/fofr/flux-handwriting)
- A type-design forum account of an earlier AI alphabet experiment (AlphabetMagic) says that generating letters one at a time gives no overall consistency across the alphabet — [TypeDrawers](https://typedrawers.com/discussion/4665/alphabetmagic-my-first-ai-experiment/p12)
- GlyphControl reports better OCR accuracy than DeepFloyd IF, SDXL and SD on SimpleBench and CreativeBench — [review](https://liner.com/review/glyphcontrol-glyph-conditional-control-for-visual-text-generation)
- The ControlText authors, building on AnyText, report that glyph-based conditioning loses fine font detail when text is very small, thin or long — [arXiv 2502.10999](https://arxiv.org/pdf/2502.10999)
- GlyphDraw2 says ControlNet has weak control over minute detail — [arXiv 2407.02252](https://arxiv.org/html/2407.02252)
- AnyText2 finds TextDiffuser-2 weaker in text accuracy, especially for multiple lines — [arXiv 2411.15245](https://arxiv.org/pdf/2411.15245)
- FontAdapter transfers font style from non-English reference glyphs, though it was trained only on English text. It is built for words in scenes, not isolated glyph rasters — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)

### Inferences
- A 9B-parameter FLUX.2 Klein needs about 18 GB for bf16 weights alone, so 24 GB is tight and 12 GB likely needs quantization or offload. This is not measured. Running inference with a published LoRA is feasible. Training a new LoRA of this size on 12-24 GB would need quantized training (QLoRA-style).
- Ref2Font is conditioned on only "Aa". With ~30 letters available, the remaining letters could be inpainted into an atlas template with the user's real letters placed in their cells, so the model sees more of the hand. This is untested but follows directly from the atlas design.
- These models were pretrained on typeset text, so I would expect the same pull toward textbook letterforms. Nothing measured on idiosyncratic hands was found.
- The scene-text models (GlyphControl, AnyText, TextDiffuser) need a glyph image of the target letter as a condition, usually a rendered font. That reproduces the project's content-glyph setup, so I would expect them to be poorly suited to idiosyncratic shapes.

### Gaps
- I found no paper or benchmark on SDXL, FLUX or SD3 plus LoRA producing per-letter handwriting glyphs from 10-30 samples.
- Ref2Font's license, training data and failure rates were not retrieved, and the FLUX.2 Klein license was not checked.

## Q6. Known failure modes

### Takeaway
The reported failures are slow multi-step sampling, structural errors on intricate characters or uncommon styles, trouble with irregular or decorative styles, and loss of fine detail in small glyphs. Large T2I models add inconsistency across separately generated letters and imperfect character accuracy.

### Cited Findings
- Slow sampling compared with GANs: Diff-Font and FontDiffuser both list it as a limitation — [ar5iv Diff-Font](https://ar5iv.labs.arxiv.org/html/2212.05895); [ar5iv FontDiffuser](https://ar5iv.labs.arxiv.org/html/2312.12142). HFH-Font addresses it with 1-step distillation — [arXiv](https://arxiv.org/abs/2410.06488)
- Wrong or broken structure: stroke/component conditioning reduces but does not eliminate errors, and extremely intricate characters or uncommon styles still fail — [ar5iv Diff-Font](https://ar5iv.labs.arxiv.org/html/2212.05895)
- Irregular styles: IF-Font struggles with fancy or irregular styles, decorations, and extremely flat or narrow layouts — [mlanthology](https://mlanthology.org/neurips/2024/chen2024neurips-iffont)
- Metric/perception mismatch: adding FontDiffuser's modules made FID worse (8.12→8.54) while SSIM, LPIPS and L1 improved — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12142)
- Character accuracy in T2I font adapters is imperfect: FontAdapter has WordAcc 0.53. Its font metrics capture overall structure, not fine detail. IP-Adapter-Instruct adds graphical artifacts — [arXiv 2506.05843](https://arxiv.org/html/2506.05843v1)
- Inconsistency across letters when each is generated separately — [TypeDrawers](https://typedrawers.com/discussion/4665/alphabetmagic-my-first-ai-experiment/p12)
- Thin or small strokes lose detail under glyph conditioning — [arXiv 2502.10999](https://arxiv.org/pdf/2502.10999)
- DreamBooth memorization at 10-30 images — [arXiv 2510.09475](https://arxiv.org/pdf/2510.09475)

### Inferences
- For a 75-glyph font, inconsistent style across independently sampled glyphs is a real risk with diffusion's stochastic sampling. Possible mitigations: share the initial noise or seed across glyphs, use deterministic DDIM/DPM-Solver sampling, or generate an atlas jointly.
- FID or SSIM gains do not show that idiosyncratic shapes are captured. A writer-specific held-out-letter metric, like the project's existing evaluation on the ten photographed writers, is needed.

### Gaps
- I found no published rate of wrong-character outputs for few-shot diffusion font models, and no study of stroke-topology errors such as wrong stroke count.
