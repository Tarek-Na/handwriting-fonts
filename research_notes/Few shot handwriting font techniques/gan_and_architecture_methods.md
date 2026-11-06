# Non-diffusion few-shot font generation (GAN / encoder-decoder / transformer / VQ), 2019-2026: techniques against style collapse

Scope note: 21 tool calls. CVF open-access PDFs returned HTTP 403, so CVPR/ICCV/WACV full texts were read through arXiv, ar5iv or abstracts where possible. Unless a line says "inference", it is what the cited paper reports. Every comparison number is the authors' own; no independent replication was found for any method.

## 1. Component and local-style methods: mechanism, Latin applicability, results, code

### Takeaway
Since 2021 the field has moved from one pooled style vector to per-location or per-component style, obtained by attending from content features to reference features (FS-Font 2022, VQ-Font 2023, DA-Font 2025) or by warping content features spatially (DG-Font 2021, NTF 2023). Nearly every method is trained and evaluated on Chinese at 128x128. The only Latin numbers found for a recent local-style GAN are in DRG-Font (April 2026), which ran itself and DA-Font on 811 Latin fonts at 64x64.

### Cited Findings
**Shared codebase (clovaai/fewshot-font-generation)**
- It holds FUNIT (ICCV'19, adapted to fonts), DM-Font (ECCV'20), LF-Font (AAAI'21) and MX-Font (ICCV'21), under the MIT licence. The FUNIT directory and `base/modules/modules.py` come from NVlabs/FUNIT, which has its own terms. "Few-shot" there means "less than 10 glyph images" with no test-time fine-tuning. DM-Font and LF-Font need component labels. MX-Font needs them only for training. FUNIT needs none. DM-Font "was proposed for complete compositional scripts (e.g., Korean)". Datasets can be TTF files or ImageFolder-style rendered images, the latter meant for cases like handwriting with no font file. — [clovaai repo](https://github.com/clovaai/fewshot-font-generation)
- LF-Font typically uses 8 reference glyphs. — [search summary of LF-Font AAAI 2021](https://cdn.aaai.org/ojs/16340/16340-13-19834-1-2-20210518.pdf)

**MX-Font (ICCV 2021, arXiv 2 Apr 2021)**
- Multiple "experts" each extract style features for a different local concept, such as a left-side sub-glyph. The features are not explicitly conditioned on component labels. Labels act only as weak supervision: components are assigned to experts by graph matching with the Hungarian algorithm. An independence loss and a content-style adversarial loss enforce disentanglement. — [arXiv 2104.00887](https://arxiv.org/abs/2104.00887)
- It claims cross-lingual generation (Chinese to Korean). It argues prior methods "fail to capture diverse local styles" and cannot generalise to unseen components or scripts. — [arXiv 2104.00887](https://arxiv.org/abs/2104.00887)
- DRG-Font (2026) describes MX-Font's independence term as HSIC. — [DRG-Font](https://arxiv.org/html/2604.13797v1)

**MX-Font++ (arXiv 4 Mar 2025, Bilibili)**
- It adds Heterogeneous Aggregation Experts, which aggregate across channel and spatial dimensions, plus a content-style homogeneity loss. The authors argue MX-Font's weak feature extractor left content and style insufficiently decoupled. They claim gains on Chinese and cross-lingual generation, and code and data are on GitHub. I found no peer-reviewed venue. — [arXiv 2503.02799](https://arxiv.org/abs/2503.02799)

**DG-Font (CVPR 2021, arXiv 7 Apr 2021)**
- It is unsupervised (no paired data). Its Feature Deformation Skip Connection (FDSC) predicts displacement maps and applies deformable convolution to the content encoder's low-level features. A mixer combines them into the output, and three deformable conv layers in the content encoder learn style-invariant features. Code: github.com/ecnuycxie/DG-Font. — [arXiv 2104.03064](https://arxiv.org/abs/2104.03064)
- DA-Font (2025) says DG-Font "misses intricate local details, leaving incomplete components". — [DA-Font](https://arxiv.org/html/2509.16632v1)

**FS-Font, "Learning Fine-Grained Local Styles" (CVPR 2022, arXiv 20 May 2022)**
- Its Style Aggregation Module (SAM) is 8-head cross-attention: content features are the queries, reference features the keys and values. The output is concatenated with the content features before decoding, so each content location gets its own style. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- Reference Selection uses a Chinese component-decomposition table. A greedy search picks the references that share the most components with the target, with ties broken by matching structural composition. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- The Self-Reconstruction branch feeds the ground-truth glyph itself as the reference, with shared weights and L1 plus adversarial losses. Its stated purpose is to make the attention's correspondence easier to learn and to recover detail. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- Data: 407 Chinese fonts, 3,396 characters, 128x128. Training: batch 32, 500K iterations, Adam (G 2e-4, D 8e-4). The reference count k is ambiguous in the text ("33-shot", possibly a typo for 3). Only Chinese is evaluated. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- Results, unseen fonts and unseen characters (L1/RMSE/SSIM/LPIPS/user-style %):

  | Method | L1 | RMSE | SSIM | LPIPS | User style % |
  |---|---|---|---|---|---|
  | FS-Font | 0.106 | 0.283 | 0.642 | 0.1627 | 67.1 |
  | LF-Font | 0.138 | 0.334 | 0.577 | 0.2018 | 13.3 |
  | MX-Font | 0.153 | 0.352 | 0.573 | 0.2317 | 11.2 |
  | DG-Font | 0.141 | 0.341 | 0.573 | 0.2151 | 3.1 |
  | AGIS-Net | 0.114 | 0.302 | 0.623 | 0.1877 | 4.1 |

  — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- No code link is given on the arXiv page. — [arXiv 2205.09965](https://arxiv.org/abs/2205.09965)

**XMP-Font (CVPR 2022, arXiv 2204.05084)**
- It uses self-supervised cross-modality pre-training: a transformer encoder is conditioned jointly on the glyph image and its stroke labels, then used without fine-tuning. It needs one reference and reports a 28% lower bad-case rate than the second-best method. Stroke labels are CJK-specific. — [CVF page](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_XMP-Font_Self-Supervised_Cross-Modality_Pre-Training_for_Few-Shot_Font_Generation_CVPR_2022_paper.html)

**CG-GAN, "Look Closer to Supervise Better" (CVPR 2022, arXiv 2205.00146)**
- The authors argue a coarse discriminator cannot supervise a font generator. Their Component-Aware Module supervises the generator at the component level. They report one-shot Chinese results and state that the method applies to handwritten word synthesis, with research questions on writer-relevant and writer-irrelevant handwriting generation. I did not retrieve the handwriting numbers. — [search summary, arXiv 2205.00146](https://emergentmind.com/papers/2205.00146)

**CF-Font (CVPR 2023, arXiv 24 Mar 2023)**
- Its Content Fusion Module projects a character's content feature onto the space spanned by the content features of several basis fonts, instead of using a single content font. Its Projected Character Loss treats 1-D projections of the glyph as distributions and compares them, which the authors say "focuses more on global character shape than L1/L2". Iterative Style-vector Refinement adjusts the style vector. Data: 300 fonts with about 6.5k characters each. Code: github.com/wangchi95/CF-Font. — [arXiv 2303.14017](https://arxiv.org/abs/2303.14017)

**NTF, Neural Transformation Fields (CVPR 2023)**
- Inspired by NeRF, NTF models generation as a continuous transformation from the source glyph to the target, through the "creation and dissipation of font pixels". Intermediate results are sampled along the estimated path and accumulated by a rendering formula. It has four parts: style estimator, structure encoder, transformation field and discriminator. The authors' motivation is that prior work's focus on stronger style representations lacks "modeling spatial transformation". Code: github.com/fubinfb/NTF. — [CVPR 2023 poster](https://cvpr2023.thecvf.com/virtual/2023/poster/21627)

**VQ-Font (ICCV 2023, arXiv 2 Sep 2023)**
- A frozen VQ-VAE-pretrained content encoder yields component codes. The Local Style Aggregator is 3 transformer layers with 8 heads: codebook entries are the queries, per-reference style features the keys and values. The stylised codes are mapped back to the content features by nearest-code lookup. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- The Global Style Aggregator weights each reference by the per-channel normalised cross-correlation between its content features and the target's, softmax-normalised. This is soft reference selection by structural similarity that needs no component labels. A multi-task discriminator judges realism and classifies style and content. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- The codebook has 100 entries for Chinese. The authors report that 15 codes sufficed for English (Latin), and the paper claims generalisation across scripts. Latin data details were not found. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827); [arXiv 2309.00827](https://arxiv.org/abs/2309.00827v2)
- Setup: 3 references by default (ablated from 1 to 8, with diminishing returns), 128x128, 370 training fonts. Code: github.com/awei669/VQ-Font. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- Results, unseen fonts and unseen characters (SSIM/RMSE/LPIPS/FID/user %):

  | Method | SSIM | RMSE | LPIPS | FID | User % |
  |---|---|---|---|---|---|
  | VQ-Font | 0.566 | 0.390 | 0.282 | 110.1 | 54.3 |
  | MX-Font | 0.478 | 0.447 | 0.333 | 114.8 | 15.2 |
  | DG-Font | 0.493 | 0.397 | 0.312 | 136.7 | 10.8 |
  | FS-Font (as re-run by VQ-Font) | 0.418 | 0.474 | 0.377 | 173.7 | 19.5 |

  This re-run of FS-Font is far worse than FS-Font's own table, so cross-paper numbers are not comparable. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- Stated limitations: quality drops with one reference and with highly decorative fonts. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)

**DA-Font (ACM MM 2025, arXiv Sep 2025)**
- A GAN (hinge loss, multi-task discriminator) with a pretrained content encoder, a content alignment module and the Dual-Attention Hybrid Module (DAHM). Its component attention uses the codebook as queries and reference style as keys and values, followed by graph feature propagation. Its relation attention uses content features as queries, the original codebook as keys and the stylised codebook as values, with a SoftPool and conv "Local Feature Refiner". — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- New shape losses: a corner consistency loss (Shi-Tomasi corners matched bidirectionally between output and ground truth) and an elastic mesh feature loss (local patch features at mesh points). It also uses style contrastive and feature-matching L1 losses. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- Setup: 575 Chinese fonts, 128x128, 4 references (gains beyond 4 marginal), 600K iterations at batch 8. Code: github.com/wrchen2001/DA-Font. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- Results, unseen fonts and unseen characters: SSIM 0.7352 vs 0.6966 (MX-Font); LPIPS 0.1699 vs 0.2014 (CF-Font); FID 54.19 vs 58.26 (VQ-Font). It also beats the diffusion models FontDiffuser and IF-Font on these metrics. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)

**DRG-Font (arXiv 15 Apr 2026)**
- A GAN encoder-decoder with a PatchGAN hinge loss. A frozen SD-v2 VAE encoder is used only for a latent reconstruction loss. Its Reference Selection module picks 1 reference from 10 candidates per target with a Stroke Matching Comparator: skeletonise, split into strokes, compare stroke length, mean curvature and an 8-bin orientation histogram by cosine similarity. Circle Loss is applied separately to style and content embeddings. The style head uses channel mean and variance statistics. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
- Only a project page is given (rejoycs.github.io/drg-font); no code repository was found. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)

**Transformer-only designs**
- TransFont is a pure transformer; its authors argue ViT generates glyph images better than CNNs. It is evaluated on Chinese. — [search summary; Font transformer, CVIU 2024](https://www.sciencedirect.com/science/article/abs/pii/S1077314224001243)
- FontTransformer (arXiv 2210.06301) uses stacked transformers for high-resolution Chinese glyphs. — [arXiv 2210.06301](https://arxiv.org/pdf/2210.06301)
- A one-shot multilingual ViT (arXiv 15 Dec 2024) uses MAE pre-training plus a Retrieval-Augmented Guidance module that retrieves style references. It claims "logographic and alphabetic scripts" but gives no Latin numbers on the abstract page and links no code. — [arXiv 2412.11342](https://arxiv.org/abs/2412.11342)

### Inferences
- The methods that report the largest gains in user-judged style consistency (FS-Font 67% vs ≤13%; VQ-Font 54% vs ≤20%) all route style through dense attention from content positions to reference positions, not through a pooled vector. The project already has this attention, but only at 8x8 with 4x4 reference keys. At that resolution a 128-px glyph cell covers about 16x16 to 32x32 pixels, plausibly too coarse to carry a terminal, loop or serif shape.
- The project measured that swapping the content font changes output by less than 0.5% and zeroing the skips by less than 1%. Its decoder is therefore not using spatially specific content or style information. In FS-Font the attended style is concatenated with content and drives decoding, and the self-reconstruction branch was its single most valuable component (ablation below). That branch directly trains the attention to copy local appearance from a reference, which a model that draws an average letter has never been forced to do.
- Component-label methods (DM-Font, LF-Font, XMP-Font, CG-GAN, FS-Font's reference selection) do not transfer to Latin as published, because Latin letters have no shared radical table. Label-free variants do: MX-Font's weakly supervised experts, VQ-Font's learned codebook (15 codes for English) and DRG-Font's stroke-descriptor selection.

### Gaps
- I could not retrieve full texts for DM-Font, LF-Font, XMP-Font, CG-GAN or NTF (CVF returned 403), so their exact numbers, reference counts and training costs are missing here.
- No "Fontformer" paper matching the brief was found. The closest are TransFont / "Font transformer for few-shot font generation" (CVIU 2024) and FontTransformer (2022).
- MX-Font++ quantitative tables and its code URL were not retrieved.

## 2. Which techniques move strokes (shape) rather than texture or weight, and with what evidence

### Takeaway
The direct evidence for shape transfer is ablations in which removing local attention, reference selection or self-reconstruction worsens structure-sensitive metrics (SSIM, LPIPS, FID) and user-judged style. DA-Font adds losses aimed explicitly at geometry: corners and elastic mesh patches. Deformation methods (DG-Font, NTF, CF-Font's basis fusion) are motivated by geometric change, but the evidence retrieved for them is only an aggregate claim that they beat the state of the art.

### Cited Findings
- FS-Font ablation (unseen fonts and characters). Each component helps, and self-reconstruction matters most:

  | Variant | L1 | SSIM | LPIPS |
  |---|---|---|---|
  | Full model | 0.106 | 0.642 | 0.1627 |
  | w/o SAM (averaged reference features instead of attention) | 0.113 | 0.621 | 0.1822 |
  | w/o self-reconstruction | 0.127 | 0.596 | 0.1859 |
  | w/o reference selection | 0.114 | 0.624 | 0.1798 |
  | All three removed | 0.131 | 0.584 | 0.2189 |

  — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- VQ-Font ablation: the full model scores FID 110.1, SSIM 0.566 and user preference 72.4%. Without local style attention it scores FID 145.0, SSIM 0.496 and 7.3%. Without similarity-weighted global style it scores FID 132.1 and 20.3%. The authors conclude local style contributes more. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- Shape-targeted losses:
  - CF-Font's projected character loss is said to emphasise "global character shape" over L1/L2. — [arXiv 2303.14017](https://arxiv.org/abs/2303.14017)
  - DA-Font's corner consistency loss matches detected corner points; its elastic mesh loss compares local patches. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- Offset and warp methods:
  - DG-Font predicts displacement maps and applies deformable convolution on the skip connections. — [arXiv 2104.03064](https://arxiv.org/abs/2104.03064)
  - NTF explicitly models spatial transformation from source to target glyph. — [CVPR 2023 poster](https://cvpr2023.thecvf.com/virtual/2023/poster/21627)
- Reference selection by structural similarity:
  - FS-Font uses shared components. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
  - VQ-Font uses content-feature cross-correlation weights. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
  - DRG-Font uses skeleton stroke descriptors, which is script-agnostic and was evaluated on Latin. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
- Contrastive and classifier losses:
  - VQ-Font and DA-Font use multi-task discriminators that classify style and content. — [VQ-Font](https://ar5iv.labs.arxiv.org/html/2309.00827); [DA-Font](https://arxiv.org/html/2509.16632v1)
  - DA-Font adds a style contrastive loss. — [DA-Font](https://arxiv.org/html/2509.16632v1)
  - DRG-Font uses Circle Loss on separate style and content embeddings. — [DRG-Font](https://arxiv.org/html/2604.13797v1)
  - CG-GAN argues a coarse discriminator is insufficient and supervises at component level. — [CG-GAN summary](https://emergentmind.com/papers/2205.00146)
- A handwriting-domain analogue: Handwriting Transformers (ICCV 2021) use self-attention over style examples plus encoder-decoder attention that gathers a style representation per query character, aiming at global and local style. — [arXiv 2104.03964](https://arxiv.org/abs/2104.03964)
- VATr (CVPR 2023) represents content as dense vectors from rendered GNU Unifont symbol images. — [VATr CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Pippi_Handwritten_Text_Generation_From_Visual_Archetypes_CVPR_2023_paper.pdf)

### Inferences
- No retrieved paper isolates shape from texture with a dedicated metric such as skeleton or contour distance. Evidence of shape transfer is indirect, through SSIM, LPIPS, FID and user style preference. Inference: the project should measure shape directly, for example by comparing skeletons or contours of output and ground truth on held-out letters. The project's own content-swap and skip-zeroing tests are already more diagnostic than anything these papers report.
- Corner matching (DA-Font) and projection-distribution losses (CF-Font) are cheap, architecture-independent additions that directly penalise averaged geometry. They are a natural complement to the L1 and Dice losses the project found favour average shapes.
- Dense attention alone does not force copying. FS-Font's self-reconstruction branch (feed the target as its own reference) is what teaches attention to transfer local detail, and it is the largest single ablation drop. It is cheap: one extra forward pass with shared weights.

### Gaps
- I found no paper that reports multi-scale cross-attention (attention at 16x16 and 32x32 rather than only at the bottleneck) as an ablated design choice for fonts. The SAM's feature-map resolution was not stated on the page retrieved.
- I found no deformable-skip ablation numbers for DG-Font.

## 3. Few-shot Latin and handwritten-style generation specifically

### Takeaway
Latin few-shot work is sparse and older: MC-GAN (2018, 26 letters), AGIS-Net (2019, artistic English) and FTransGAN (2021, English to Chinese). The only recent non-diffusion paper with Latin numbers found is DRG-Font (2026): 811 Latin fonts, 52 characters, 64x64, one selected reference from 10, beating DA-Font slightly. No paper found reports few-shot results on photographed handwriting for Latin glyph sets.

### Cited Findings
- **MC-GAN** generates all 26 Latin letters (cited in a 2020 metric-learning paper). — [arXiv 2011.02206](https://arxiv.org/pdf/2011.02206)
- **AGIS-Net (SIGGRAPH Asia 2019 / ToG 38(6))**
  - It uses two encoders (content, style) and two cooperating decoders that generate shape and texture images, plus a local texture refinement loss. It is evaluated on English and Chinese artistic glyph datasets and targets artistic fonts, not handwriting. Code and data: hologerry.github.io/AGIS-Net. — [arXiv 1910.04987](https://arxiv.org/abs/1910.04987)
  - On FS-Font's Chinese benchmark AGIS-Net was the strongest pre-2022 baseline (UFUC SSIM 0.623 vs FS-Font 0.642). — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- **FTransGAN (WACV 2021, Kyushu University)**
  - It extracts style from a few observed English letters and applies it to Chinese characters, using a Context-aware Attention Network and a Layer Attention Network that capture local and global style. It needs only a forward pass at test time. — [CVF page](https://openaccess.thecvf.com/content/WACV2021/html/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.html)
  - A secondary (blog) summary says the number of style images must be fixed during training and that highly artistic fonts perform poorly. — [blog summary](https://dlaiml.tistory.com/entry/Few-shot-Font-Style-Transfer-between-Different-Languages)
  - GAS-NeXt builds on FTransGAN. — [search result](https://link.springer.com/article/10.1007/s10489-022-04375-6)
- **DRG-Font (2026), Latin results**
  - Data: 811 fonts with 52 characters (A-Z, a-z), 783 for training and 28 unseen for testing, at 64x64. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
  - Unseen English results:

    | Method | L1 | RMSE | SSIM | LPIPS |
    |---|---|---|---|---|
    | DRG-Font | 0.072 | 0.237 | 0.739 | 0.108 |
    | DA-Font (re-run by DRG-Font) | 0.074 | 0.243 | 0.713 | 0.111 |

    Users preferred DRG-Font in 53.42% of English cases. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
- **VQ-Font** reports 15 codebook entries sufficing for English. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- **clovaai** supports training from rendered images, and its README names handwriting as the use case for that format. — [clovaai repo](https://github.com/clovaai/fewshot-font-generation)
- **FS-Font's broader-impact statement** says the method could imitate handwriting, though a human expert can still distinguish the result. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)

### Inferences
- Reference counts in the literature are 1 to 8 (VQ-Font 3, DA-Font 4, LF-Font about 8, DRG-Font 1 selected from 10). The project's roughly 30 references is far more than any benchmark uses. The bottleneck is therefore unlikely to be the amount of reference data; it is more likely that the pooled code and coarse attention cannot use it. Per-target selection or weighting (VQ-Font's similarity weights, DRG-Font's stroke matching), which picks the 3 to 5 most structurally related references for each target letter, is directly applicable. For example, 'n' would draw on 'h', 'm' and 'u', and 'q' on 'g', 'd' and 'a'.
- Google Fonts handwriting-category fonts are still digital and clean. None of the papers found evaluates domain shift to photographed handwriting, so reported Latin results probably overstate what the project will see on its ten held-out writers.

### Gaps
- I did not retrieve Attribute2Font (SIGGRAPH 2020) or MC-GAN primary sources.
- No Google-Fonts-based few-shot Latin benchmark with handwriting-category breakdowns was found.
- I did not retrieve CG-GAN's handwriting metrics.

## 4. Training cost and model size: one consumer GPU at 128 px?

### Takeaway
Parameter counts are almost never reported. Iteration budgets are 500K to 600K steps at batch 8 to 48, and DRG-Font trained on a single RTX 4080 (16 GB) at 64 px. Inference: all of these GANs are feasible on one consumer GPU at 128 px, but it takes days per run.

### Cited Findings
- FS-Font: batch 32, 500K iterations, 128x128; GPU not stated. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- VQ-Font: codebook pre-training at batch 256 for 50K iterations, then the full model at batch 48 for 500K iterations, 128x128; GPU not stated. — [ar5iv 2309.00827](https://ar5iv.labs.arxiv.org/html/2309.00827)
- DA-Font: stage 1 at batch 64 for 50K iterations, then stage 2 at batch 8 for 600K iterations, 128x128. GPU and parameter count not reported. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- DRG-Font: 500 epochs, batch 64, single RTX 4080 16 GB, 64x64. Training time and parameter count not reported. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
- clovaai's README gives no iterations or GPU requirements and points to per-model docs. — [clovaai repo](https://github.com/clovaai/fewshot-font-generation)

### Inferences
- Batch sizes of 8 to 48 at 128 px with these CNN/GAN backbones fit comfortably in 12 to 24 GB. Half a million iterations is the dominant cost. The project can borrow modules (multi-scale cross-attention, self-reconstruction, shape losses) and fine-tune its existing 32M model instead of training these models from scratch.

### Gaps
- No parameter counts were found for FS-Font, VQ-Font, DA-Font, DRG-Font, CF-Font or NTF.
- No wall-clock training times were found.

## 5. Reported weaknesses of AdaIN / pooled-style designs, and how papers quantify style capture

### Takeaway
The local-style papers explicitly argue that global or component-wise disentanglement misses style distributed across spatial locations. The cleanest quantification is FS-Font's ablation that replaces attention with averaged reference features. Style capture is measured with L1, RMSE, SSIM, LPIPS, FID and user studies; no paper found uses a shape-specific metric.

### Cited Findings
- FS-Font: most existing methods "explicitly disentangle content and style of reference glyphs globally or component-wisely". Style lies in local details, and "even a single character can contain different styles distributed over spatial locations". — [arXiv 2205.09965](https://arxiv.org/abs/2205.09965)
- Replacing FS-Font's attention with averaged features raises LPIPS from 0.1627 to 0.1822 and lowers SSIM from 0.642 to 0.621. — [ar5iv 2205.09965](https://ar5iv.labs.arxiv.org/html/2205.09965)
- NTF argues that the field's focus on stronger style representations is sub-optimal without modelling spatial transformation. — [CVPR 2023 poster](https://cvpr2023.thecvf.com/virtual/2023/poster/21627)
- MX-Font says earlier methods "fail to capture diverse local styles". — [arXiv 2104.00887](https://arxiv.org/abs/2104.00887)
- DA-Font critiques specific methods: MX-Font and LF-Font give blurry outputs; DG-Font has incomplete components; FUNIT is structurally incomplete unless the content and target fonts are similar. — [arXiv 2509.16632](https://arxiv.org/html/2509.16632v1)
- DRG-Font still uses channel mean and variance (AdaIN-style statistics) in its style head and does not analyse this as a failure mode. — [arXiv 2604.13797](https://arxiv.org/html/2604.13797v1)
- Metrics in use: L1, RMSE, SSIM, LPIPS, FID and user studies of character correctness and style consistency (FS-Font, VQ-Font, DA-Font, DRG-Font). — [FS-Font](https://ar5iv.labs.arxiv.org/html/2205.09965); [VQ-Font](https://ar5iv.labs.arxiv.org/html/2309.00827); [DA-Font](https://arxiv.org/html/2509.16632v1); [DRG-Font](https://arxiv.org/html/2604.13797v1)

### Inferences
- The project's symptom (it keeps weight and slant but loses idiosyncratic shapes) matches what global channel statistics can encode: thickness, contrast and slant-like global tendencies, but not where a loop or terminal sits. That is consistent with the papers' argument, but no retrieved paper demonstrates this mechanism on Latin handwriting.
- Pixel metrics such as SSIM and L1 reward the average letter. User style-preference studies are where local-style methods separate most strongly from the rest (FS-Font 67% vs 13%), which suggests the project needs a style-specific evaluation, not just reconstruction error.

### Gaps
- I found no paper that quantifies "style collapse" (the output ignoring content or skips) with ablations like the project's content-swap and skip-zeroing tests.
- I found no paper that reports per-character shape metrics (skeleton or Chamfer distance) for few-shot fonts.
