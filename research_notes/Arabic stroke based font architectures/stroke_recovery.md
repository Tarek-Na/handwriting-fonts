# Stroke / trajectory recovery from static handwriting images (2018-2026)

## Learned methods: what exists, architecture, data, metrics, code/weights, runtime

### Takeaway
InkSight (Google, 2024; TMLR 2025) is the only learned offline-to-online method found that ships usable weights under a permissive licence (Apache-2.0 code and Small-p weights). It works on word-sized crops of photographed handwriting, but Arabic is not among its documented languages. The academic line (Bhunia 2018, Nguyen 2020, TRACE 2021, PEN-Net 2022, diffusion 2026, ordered stroke instances 2026) is mostly paper-only or code-without-weights, and trained on Chinese, Japanese, Indic or Latin online data. No modern learned method for Arabic was found.

### Cited Findings
**InkSight (Mitrevski, Rak, Schnitzler, Li, Maksai, Berent, Musat; Google)**
- Architecture: a ViT encoder (ViT-B/16 for the Small models, ViT-L/16 for Large-i) feeding an mT5 encoder-decoder. The vocabulary is single characters plus 2N+3 ink tokens (N=224). The ViT is frozen; the mT5 part is trained from scratch — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Paired offline/online data is not needed. Training is a multi-task mix of derendering and recognition: derendering learns from synthetic renders of online ink with heavy augmentation (stroke width, colour, noise, cluttered backgrounds), and recognition learns from real OCR word images — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Training data:
  - Public: about 2.7M ink samples (DeepWriting, VNOnDB, SCUT-COUCH) and about 295K OCR samples (RIMES, HierText, IMGUR5K, ICDAR'15, IAM).
  - In-house: about 16M ink and about 500K OCR samples. The in-house data is mostly Mandarin (37%) and Japanese (23%), with every other language under 5% each.
  - Arabic is not mentioned. Source: [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Results:
  - Human evaluation (HierText, 200 samples, 16 raters): 87% of outputs were valid tracings and 67% looked human-made.
  - Character-level F1 on IAM / IMGUR5K / HierText: Small-p 0.65 / 0.47 / 0.53; Large-i 0.66 / 0.51 / 0.61. Human traces score 0.64 on HierText, so this metric has a ceiling.
  - An online recogniser reading the output ink scores 0.37–0.46 on HierText, against 0.74 for human tracings. Source: [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Training a recogniser on derendered ink (IAMOnDB test CER, 9.2M-parameter Transformer with CTC):
  - Real IAMOnDB only: 6.1.
  - Derendered IAM only: 7.8–8.4.
  - Real plus derendered: 4.5–4.6.
  - Derendered ink is usable training data and works best when added to real ink — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Input and scope:
  - Images are scaled and padded to 224×224, with an output context of 1024 tokens.
  - It works on word-level crops; full pages need OCR word boxes first.
  - Listed failure cases: extreme stroke width or style, untrained scripts (Korean is the example), complex many-stroke drawings, empty-ink outputs, and missing strokes or extra loops — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Runtime:
  - Greedy decoding at batch 32 runs at about 47 tokens/s on a T4, 114 on a Titan RTX and 139 on a V100.
  - Training a 340M model took about 33 h on 64 TPU v5e chips — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- Releases:
  - Small-p weights (about 340M parameters, public data only) for CPU/GPU on Hugging Face, plus a TPU version.
  - Inference Colab, a Hugging Face Space demo, a Gradio playground, and a dataset of model outputs and expert traces.
  - Code is Apache-2.0 and runs on TensorFlow 2.20. Supported OCR includes docTR and Tesseract.
  - Small-i and Large-i are not released. Source: [GitHub google-research/inksight](https://github.com/google-research/inksight)
- The Hugging Face model card for Small-p lists `license: apache-2.0`. One third-party page lists CC BY-NC-SA 4.0, which probably refers to the paper text — [HF model card](https://huggingface.co/Derendering/InkSight-Small-p/blob/main/README.md?code=true); [ecosyste.ms listing](https://awesome.ecosyste.ms/projects/github.com%2Fgoogle-research%2Finksight)

**Bhunia et al. 2018 (ICPR 2018)**
- A ConvLSTM encoder over the character image and an LSTM decoder that emits pen-tip coordinates. Trained on Tamil, Telugu and Devanagari isolated characters (LIPI Toolkit).
- The abstract gives no numbers. A Google Drive "code link" is listed — [arXiv 1801.07211](https://arxiv.org/abs/1801.07211)

**Nguyen, Nakamura, Nguyen, Nakagawa 2020 (Japanese kanji; submitted to ICPR 2020)**
- A CNN encoder and an attention LSTM decoder that generates coordinate sequences.
- Evaluated by visual checks and by recognition: online recognition of the recovered trajectories improves character recognition. No code link was found — [arXiv 2009.04284](https://arxiv.org/abs/2009.04284)

**TRACE (Archibald, Poggemann, Chan, Martinez; ICDAR 2021, BYU)**
- A CRNN trained end-to-end on whole text lines (IAM-OnDB), with no pre- or post-processing. The output trajectory is differentiable and can serve as a loss. Recovered trajectories were used for handwriting synthesis — [arXiv 2105.11559](https://arxiv.org/abs/2105.11559v1); [BYU thesis](https://scholarsarchive.byu.edu/etd/9299)
- Papers With Code lists "no code implementations" — [PwC author page](https://paperswithcode.com/author/taylor-archibald)

**PEN-Net (Chen et al., ACCV 2022)**
- A double-stream parsing encoder and a global tracing decoder, aimed at complex glyphs and long trajectories. Evaluated on Chinese, Japanese and Indic data.
- Proposes two metrics:
  - AIoU, for glyph fidelity regardless of stroke width.
  - LDTW, a length-independent DTW. Plain DTW favours trajectories with fewer points. Source: [arXiv 2210.15879](https://arxiv.org/pdf/2210.15879)
- The repository has PyTorch training code and configs. No pretrained weights or licence are shown, and data is CASIA (obtained externally) — [GitHub ChenZhounan/PEN-Net](https://github.com/ChenZhounan/PEN-Net)

**Diffusion trajectory recovery (Nagamatsu, Toyota, Uchida; arXiv July 2026)**
- Recovery as image-conditioned denoising diffusion, trained on CASIA-OLHWDB 1.0–1.1. The authors claim large DTW/LDTW and AIoU gains over PEN-Net and Cross-VAE.
- A model trained on Chinese recovers "reasonable stroke orders for Latin letters to some extent".
- The abstract gives no numbers and no code link — [arXiv 2607.03422](https://arxiv.org/abs/2607.03422)

**Ordered stroke instance prediction (Wang, Zhang, Yin, Liu; arXiv Sept 2026)**
- Two stages: autoregressive prediction of ordered stroke instances, then the motion within each stroke.
- Trained on Chinese, with cross-language tests on English and Tamil.
- The authors claim ordered prediction beats ordering strokes after the fact, and that trajectory sampling density strongly affects measured scores. No code was found — [arXiv 2609.02251](https://arxiv.org/abs/2609.02251)

**Temple University dissertation (2024)**
- Diffusion conditioned on a style template, trained with a DTW loss, for unconstrained documents — [Temple scholarshare](https://scholarshare.temple.edu/server/api/core/bitstreams/8216525e-6f5a-4cb9-90c0-58c199e5f1dc/content)

### Inferences
- For phone photos, InkSight Small-p is the only option that runs without training. It was built for word crops with real-photo augmentation. Whether it can handle isolated Arabic letters is unknown: Arabic is absent from its documented data, and the authors themselves list unseen scripts as a failure mode.
- The academic models all need a large online dataset in the target script to train on. For Arabic that means LMCA, or the online datasets covered in other notes, plus the owner's own pen captures.

### Gaps
- Sumi et al. (Cross-VAE, about 2019) was not fetched. It appears here only as the baseline named by Nagamatsu 2026.
- No numerical DTW/AIoU results were retrieved for PEN-Net, the 2026 diffusion paper or the 2026 stroke-instance paper; only their abstracts were read.
- The licence of the InkSight dataset and the licence of Bhunia's Drive code were not checked.
- No Arabic run of InkSight was found anywhere, so how it does on Arabic letters is untested.

## Arabic-specific stroke recovery

### Takeaway
The Arabic work comes from the REGIM/Sfax group (Elbaati, Rabhi, Boubaker, Alimi), using the LMCA dual on/off database. Their methods are a genetic-algorithm search over skeleton segments (2009) and later learned autoencoder plus BGRU models (2021–2023). No released code or weights were found. Detached dots and delayed strokes are a known difficulty, and no modern code handles them for Arabic.

### Cited Findings
- **Abuhaiba et al.:** three classical algorithms for offline Arabic: polygonal approximation of the smoothed image, a heuristic to find the character's start point, and graph traversal to recover temporal order — [search summary of Arabic recovery literature; arXiv 1801.07211 related work](https://arxiv.org/pdf/1801.07211)
- **Elbaati et al. (2009):**
  - Method: segments the skeleton into three stroke types, then uses a genetic algorithm to find the best order through them.
  - Data: LMCA (dual on/off Arabic), IRONOFF (Latin) and IFN/ENIT.
  - A companion paper uses the restored order for Arabic recognition. Source: [AICCSA / IEEE CS digital library](https://doi.ieeecomputersociety.org/10.1109/AICCSA.2005.1387116)
- **Rabhi, Elbaati, Boubaker, Pal, Alimi (Multimedia Tools and Applications, online August 2023):**
  - Recovers pen up/down, velocity and temporal order from offline words.
  - Architecture: a stacked denoising autoencoder and a BGRU encoder, then a BGRU decoder with temporal attention.
  - Evaluated on Latin and Arabic characters and words, Arabic from LMCA. Only the abstract was available: it says "highly competitive results" with no figures — [DOI 10.1007/s11042-023-16499-z](https://doi.org/10.1007/s11042-023-16499-z); [ISI digital commons](https://digitalcommons.isical.ac.in/journal-articles/4913)
- A predecessor in Memetic Computing (2021) used CNN-BGRU with attention for multilingual temporal order and velocity recovery — [aminer record](https://www.aminer.org/pub/648db2e7d68f896efaf59eba/multi-lingual-handwriting-recovery-framework-based-on-convolutional-denoising-autoencoder-with-attention)
- Across the literature, the hard parts are choosing the start point and pairing incoming with outgoing paths at junctions. Connected Arabic script makes both harder — [arXiv 1801.07211](https://arxiv.org/pdf/1801.07211)

### Inferences
- Order between strokes is less of a problem for single isolated Arabic letters than for words. A letter is typically one main body plus 0–3 dots or a hamza or madda, and the conventional order is body first, then the detached marks. That order can be set by rule from connected components (largest component first, then marks by position) rather than learned.
- Delayed strokes inside a word (for example the kaf bar) do not arise for isolated letters, except for a few letters with separate marks.

### Gaps
- No released Arabic trajectory-recovery code or weights were found.
- No paper reports DTW/AIoU numbers for Arabic.

## Classical skeleton and graph approaches and their failure cases

### Takeaway
The classical recipe is: skeletonise, build a graph of endpoints, junctions and edges, then choose a traversal. "Smoothest Euler path" with direction context is the standard formulation. The known failures are junction ambiguity (combinatorial blow-up), retraced segments (one ink line drawn twice), loops with no start point, and skeleton artefacts such as spurs and split junctions.

### Cited Findings
- Jaeger et al.'s shortest-path search over skeleton graphs blew up combinatorially on images with ambiguous zones such as junctions — [arXiv 1905.06749 related work](https://arxiv.org/pdf/1905.06749)
- One formulation treats recovery as finding the smoothest path that covers every edge of the graph. Smoothness between edges comes from direction context, the path is solved as an optimal Euler path, and double-traced lines are handled with maximum weighted matching — [search summary; typeset.io survey PDF](https://typeset.io/pdf/techniques-for-static-handwriting-trajectory-recovery-a-4t0al45bdg.pdf)
- The offline maths-expression pipeline breaks the skeleton into junctions and segments, merges segments into strokes, then orders strokes by recursive projection and topological sort — [arXiv 1905.06749](https://arxiv.org/pdf/1905.06749)
- A 2016 paper on loops in skeletal graphs names the typical failures: noisy skeleton artefacts, no identifiable start node, misidentified nodes, and high-degree nodes — [typeset.io PDF](https://typeset.io/pdf/extraction-of-dynamic-trajectory-on-multi-stroke-static-3bt4867uxb.pdf)
- Learned work exists largely because finding the start point and the in/out paths at junctions is hard without temporal information — [arXiv 1801.07211](https://arxiv.org/pdf/1801.07211)

### Inferences
- For isolated letters the graphs are small: usually fewer than about 6 edges and 0–3 junctions. Brute-force enumeration over all Euler-path or edge-ordering options is cheap, so the combinatorial blow-up does not apply. What remains is choosing well, which a scoring function (smoothness plus conventions) or a small learned ranker can do.

### Gaps
- No quantitative head-to-head between classical and learned methods on the same benchmark was found.
- The 2016 loop paper and the Nguyen & Blumenstein survey were seen only as abstracts.

## How good is it, and does training on recovered strokes work?

### Takeaway
Recovered ink is good enough to help recognition: with InkSight, derendered IAM alone gives CER 8.2 against 6.1 for real ink, and adding it to real ink drops CER to 4.5. But recovered ink is measurably less natural than real ink: an online recogniser reads it at 0.37–0.46 accuracy against 0.74 for human tracings. TRACE used recovered strokes for synthesis, but no numbers on generator quality were found.

### Cited Findings
- On IAMOnDB test CER:
  - Real ink only: 6.1.
  - Derendered only: 7.8–8.4.
  - Real plus derendered: 4.5–4.6.
  - Source: [arXiv 2402.05804v4, App. I](https://arxiv.org/html/2402.05804v4)
- On HierText, an online recogniser reads InkSight ink at 0.37–0.46 accuracy against 0.74 for human-traced ink. 67% of outputs were judged "human-like" — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- When prompted with text that is not in the image, InkSight does not produce matching ink. It derenders; it does not generate — [arXiv 2402.05804v4](https://arxiv.org/html/2402.05804v4)
- TRACE's trajectories were used for handwriting synthesis — [BYU thesis](https://scholarsarchive.byu.edu/etd/9299)
- Plain DTW is biased by point count. Trajectory sampling density strongly changes reported scores, so cross-paper numbers are not comparable — [arXiv 2210.15879](https://arxiv.org/pdf/2210.15879); [arXiv 2609.02251](https://arxiv.org/abs/2609.02251)
- Nguyen et al. report that recognising recovered kanji trajectories improves character recognition — [arXiv 2009.04284](https://arxiv.org/abs/2009.04284)

### Inferences
- (a) Few-shot style input: recovered strokes from a photo are plausible as a style reference if the generator's encoder mainly reads shape and slant. The risk is that wrong direction or order teaches the generator a wrong ductus. Errors in direction and order matter more for the stroke sequence than for how the result looks rendered.
- (b) Training a generator on offline-only data: the evidence covers recognition, not generation. Mixing recovered ink with real online ink helped recognition, while recovered ink alone was worse. By analogy, recovered ink is better used to augment a real online Arabic set than to replace it. This is an inference; no generation study measured it.

### Gaps
- No study measured how good a stroke-generator trained on recovered strokes is.
- No numbers for the 2026 methods were retrieved.

## Practical recommendation for a centreline pipeline that already skeletonises isolated glyphs

### Takeaway
For single isolated letters, the simplest robust approach is a deterministic graph-traversal method on the existing skeleton:
1. Prune the skeleton.
2. Build a graph of endpoints, junctions and edges.
3. Enumerate all Euler-path orderings of the small graph, allowing retracing of edges that must be drawn twice.
4. Score each by angle smoothness at junctions plus writing-convention priors: start at the rightmost or topmost endpoint for Arabic, write right-to-left, prefer fewer pen lifts.
5. Order connected components main body first, then dots and marks.

This can be checked against the owner's own online captures, and later swapped for a small learned ranker trained on rendered online Arabic. InkSight Small-p is worth a quick trial on the photos as a comparison, but its Arabic support is unproven.

### Cited Findings
- The smoothest-Euler-path formulation with direction context, and matching for double-traced edges, is the established classical method — [typeset.io survey PDF](https://typeset.io/pdf/techniques-for-static-handwriting-trajectory-recovery-a-4t0al45bdg.pdf)
- Splitting into segments, merging into strokes, then ordering strokes by projection and topological sort works for offline maths symbols — [arXiv 1905.06749](https://arxiv.org/pdf/1905.06749)
- Junction ambiguity drives the combinatorial cost of search-based traversal — [arXiv 1905.06749](https://arxiv.org/pdf/1905.06749)
- InkSight Small-p is Apache-2.0, runs on CPU/GPU and has a Colab notebook, so testing it costs almost nothing — [GitHub google-research/inksight](https://github.com/google-research/inksight)
- Learned recovery trained on one script transfers stroke-order tendencies to another only "to some extent" (Chinese to Latin) — [arXiv 2607.03422](https://arxiv.org/abs/2607.03422)

### Inferences
- Why rules suffice for isolated letters:
  - Few junctions make exhaustive scoring tractable.
  - Arabic letter ductus is highly conventional, and a per-letter table of start point and direction for 28 letters × up to 4 forms is small enough to write by hand.
  - Dots and marks are separate components with a fixed convention: drawn after the body.
- The stubborn failures for Arabic isolated letters will be:
  - closed loops (ha, mim, waw, fa/qaf heads, sad/dad bowls), where the start point and direction around the loop are ambiguous;
  - retraced tails (for example the hook back in some ra and waw forms);
  - junctions where the skeleton splits into two nearby nodes.

  A per-letter prior resolves the first two: "loops go counter-clockwise from the top" or "start where the reference ductus starts". The skeleton pruning step should merge nearby junction nodes.
- Since the letter identity is known (sheet slots), a template-guided approach is the most robust option. Match the skeleton graph to a reference stroke template for that letter, from the owner's online captures or a font's ductus, and copy its order and direction. This turns the problem into graph matching rather than blind recovery.
- To evaluate, use LDTW and AIoU, or at minimum direction and order agreement, against ground truth from tablet captures. Do not rely on plain DTW.

### Gaps
- No published system does template-guided stroke ordering for isolated Arabic letters with released code. This recommendation is an engineering inference, not a measured result.
