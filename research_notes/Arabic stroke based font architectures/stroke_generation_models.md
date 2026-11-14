# Stroke/trajectory-based (online) handwriting generation models, 2020-2026, as candidates for few-shot handwriting-to-font

Verification legend: [V-paper] = read in the paper text (ar5iv/arXiv HTML) this session; [V-repo] = read on the GitHub README this session; [V-search] = from search-result snippets or abstracts only; [Unverified] = from background knowledge, not checked this session (these are kept in Gaps/Inferences).

## SDT (Dai et al., CVPR 2023): architecture, data, references, metrics, licence, compute, weights, script coverage

### Takeaway
SDT is the strongest documented open-source candidate. It is an MIT-licensed, character-level online generator. It takes **K=15 offline style images (64x64)** plus a content glyph image and outputs a pen trajectory as a 20-component bivariate GMM over offsets plus 3 pen states. It was trained on one RTX 3090 and has released Chinese weights. It has English configs (52 Latin letters) but nothing for Arabic, and its style was weakest on English. The writer-ID classifier reached only 20.57% on English, against 99.98% on Chinese.

### Cited Findings
- [V-paper] Style encoder: ResNet18 followed by a 2-layer Transformer encoder. The inputs are **K = 15 style reference images resized to 64x64**. Two heads split the features into a writer-wise representation and a character-wise (glyph) representation. — [SDT paper, ar5iv](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Two contrastive losses. **WriterNCE** treats characters from the same writer as positives (tau = 0.07, MLP projection). **GlyphNCE** treats two patches sampled from the same character as positives. The patch sampling ratio alpha = 0.25 was best (94.50 Style Score, against 91.54 at alpha = 1.0). — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Content encoder: ResNet18 plus a Transformer encoder, c = 512. The content reference is an *image* of the character: an "average Chinese font" for Chinese, printed fonts for Japanese and English, and learned character embeddings for Indic (Tamil). — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Decoder: a multi-layer Transformer decoder (512-d, 8 heads) that fuses style and content through attention. It is autoregressive and emits 6R+3 = 123 logits per step: **R = 20 bivariate Gaussian mixtures** over the pen offsets (du, dv), plus 3 pen states (down, up, end). Loss: L = L_wri + L_gly + L_pre + 2·L_cls. Concatenation instead of attention fusion dropped the Style Score to 78.12. — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Datasets:
  - Chinese: CASIA-OLHWDB 1.0-1.2 for training (~3.7M characters, 1,020 writers). ICDAR-2013 for testing (60 writers x 3,755 characters). Trajectories were RDP-simplified (eps = 2), average length 50 points.
  - Japanese: TUAT HANDS (~3M characters, 271 writers, 216 train / 55 test).
  - Indic (Tamil): 156 classes, 169 writers.
  - English: 53,248 training characters from 1,020 writers, 3,120 test characters from 60 writers, **52 classes (a-z, A-Z)**, average length 30 points.
  - Source: [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Chinese results (Table 1, Style / Content / DTW / user preference):

  | Method | Style | Content | DTW | User preference |
  |---|---|---|---|---|
  | SDT | 94.50 | 97.04 | 0.8789 | 56.67% |
  | WriteLikeYou-v2 | 72.37 | 96.44 | 0.9289 | 13.07% |
  | WriteLikeYou-v1 | 71.09 | – | – | – |
  | DeepImitator | 50.67 | – | – | – |
  | FontRNN | 46.14 | – | – | – |
  | Drawing | 35.83 | – | – | – |

  The user study had 50 volunteers and 1,500 responses. — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Other scripts:
  - Japanese: Content 91.31, DTW 1.1289, Style 41.85, against WLU-v2 at 85.61, 1.2066, 32.88.
  - Indic: Content 97.22, DTW 0.7075, against WLU-v2 at 13.19, 4.5130.
  - **English: Content 85.52, DTW 1.6048, against WLU-v2 at 84.41, 1.6215. No English Style Score is reported.** The writer identifier used for the Style Score reached only **20.57% on English**, against 72.54% on Indic and 99.98% on Chinese. English content errors came from O/o-type case confusions.
  - Source: [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Compute: a **single RTX 3090 (24 GB)**. The content encoder was pre-trained for 138k iterations at batch 256, then the full model was trained for 148k iterations at batch 128 (Adam, lr 2e-4, gradient clipping at 5). **Wall-clock time is not reported.** — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] Stated weakness: some Indic outputs do not match the reference slant or aspect ratio. Font generation is mentioned only as future work. — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-repo] **MIT licence.** Training configs exist for `CHINESE_CASIA.yml`, `Japanese_TUATHANDS.yml` and `English_CASIA.yml`. The released checkpoints are the trained SDT, the content encoder, the Content Score model and the Style Score model, on Google Drive and Baidu. The README does not say which script the released SDT checkpoint covers, but the test commands centre on Chinese. Indic and Arabic are not mentioned. An offline-to-offline extension for Chinese is also in the repo. — [github.com/dailenson/SDT](https://github.com/dailenson/SDT)
- [V-search] Third-party line-level benchmark on CASIA-OLHWDB 2.x (DiffInk paper, Table 1), with SDT adapted to lines through OLHWG's layout module: Style 50.51%, DTW 1.270. SDT is slow (3.35 characters/s) and shows **stitching artefacts** at character boundaries. — [DiffInk arXiv 2509.23624](https://arxiv.org/html/2509.23624v1)

### Inferences
- The style input is offline images, and the output is isolated characters. That fits a "photograph a few letters, get trajectories per glyph" font pipeline better than any line-level model. However, SDT's style images during training are *rendered from online trajectories* (thin, uniform strokes). Phone photos of pen ink would be a domain shift, so binarisation and skeleton re-rendering would probably be needed. This is inferred, not tested in the paper.
- Arabic would need retraining, with (a) an online Arabic character dataset with writer IDs and (b) a content-image source (a printed Arabic font). Contextual letter forms (isolated, initial, medial, final) would each need their own class. Given the 52-class English result, the low English style identifiability suggests Latin and Arabic letters, which have few strokes, carry less writer signal per glyph than Chinese characters.
- Training on one 24 GB GPU is documented. Time is unknown. About 286k total iterations on a 3090 is plausibly a few days (inferred).

### Gaps
- The exact number of references used at test time if fewer than 15 are available. The paper fixes K = 15. No sweep over K was found.
- Whether the released checkpoint includes English weights. The repo table does not say.
- Training wall-clock time.

## Follow-ups and other trajectory models (One-DM, DiffBrush, stroke diffusion, DeepCalliFont, Kotani DSD, OLHWG, DiffInk, CHWmaster, TrInk, WLU, FontRNN, DeepWriting, Graves)

### Takeaway
From the SDT group, **One-DM (ECCV 2024) and DiffBrush (ICCV 2025) are offline image generators, not trajectory models**. The active online-trajectory line since 2023 is mostly from Chinese groups working on Chinese:
- DiffWriter (2023)
- OLHWG (ICLR 2025): 1D U-Net glyph diffusion plus a layout module, with code released
- DiffInk (2025, listed as ICLR 2026 by one source): latent diffusion transformer over lines, code not yet released
- CHWmaster (2024)
- TrInk (2025)

For Latin, the main few-shot trajectory works are Kotani et al. DSD (ECCV 2020, non-commercial licence, weights currently unavailable) and Luhman 2020 (stroke diffusion with offline-image style). No neural Arabic online-generation model was found.

### Cited Findings
- [V-repo/V-search] One-DM is a one-shot *handwritten text image* diffusion model (ECCV 2024). DiffBrush generates full text lines as images (ICCV 2025). Both are listed in the SDT repo as follow-ups. — [SDT repo](https://github.com/dailenson/SDT); [One-DM paper](https://arxiv.org/pdf/2409.04004)
- [V-search] The One-DM paper notes that offline methods can reproduce stroke width and ink colour, which online methods cannot. — [One-DM ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf)
- [V-search] DiffusionPen (ECCV 2024) is a 5-shot latent-diffusion *image* model (offline), not a trajectory model. — [DiffusionPen ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11492.pdf)
- [V-search] **Luhman & Luhman 2020**, "Diffusion models for Handwriting Generation" (arXiv 2011.06704, preprint):
  - A diffusion model that outputs online pen-stroke sequences.
  - It needs no recognition, writer or adversarial losses and no auxiliary networks.
  - **Writer style comes from offline images**, so the user does not have to write online at sampling time.
  - It was shown on the IAM (English) data.
  - Code: tcl9876/Diffusion-Handwriting-Generation. The licence was not checked.
  - Source: [arXiv 2011.06704](https://arxiv.org/abs/2011.06704v1)
- [V-repo] **Kotani, Tellex & Tompkin, "Generating Handwriting via Decoupled Style Descriptors" (ECCV 2020)**:
  - English online handwriting with an MDN output.
  - Supports writer and character interpolation. Reference samples improve quality.
  - Dataset **BRUSH: 27,649 online samples from 170 writers**, with per-point character labels.
  - **Licence: Brown copyright notice, non-commercial. BRUSH is research-only.**
  - Pretrained weights are marked "[2025-12-22 Not currently available.]"
  - The README notes there is no supervised character loss, so characters can run on.
  - Source: [github.com/brownvc/decoupled-style-descriptors](https://github.com/brownvc/decoupled-style-descriptors)
- [V-search/V-repo] **OLHWG, "Decoupling Layout from Glyph in Online Chinese Handwriting Generation"** (Ren, Zhang, Chen, ICLR 2025):
  - Architecture: character embedding dictionary, multi-scale calligraphy style encoder and a **1D U-Net diffusion denoiser** over glyph trajectories, plus a layout generator that does in-context learning from style references.
  - Code is released (`train_ddpm.py`, `style_classifier.py`, `generate_line.py`), with data on Hugging Face (`Immortalman12/OLHWD`). There is no licence and no weights on the page.
  - The README says "Due to graduation, some files were lost."
  - Chinese only.
  - Sources: [MLAnthology](https://mlanthology.org/iclr/2025/ren2025iclr-decoupling); [arXiv 2410.02309](https://arxiv.org/html/2410.02309v3); [github.com/singularityrms/OLHWG](https://github.com/singularityrms/OLHWG)
- [V-paper] **DiffInk** (Pan, He, Cheng, Shi, Jin; SCUT; arXiv 2509.23624, Sept 2025):
  - **InkVAE**: 1D convolutional sequence VAE, 8x temporal compression, 20-component GMM decoder over (x, y) plus 3 pen states, regularised by an OCR CTC loss and a writer-classification loss.
  - **InkDiT**: 16-layer DiT that predicts x0, with 5-step DDIM sampling.
  - The style reference is an **online** trajectory: a reference line, or fewer than 10 characters in one-shot mode.
  - Data: CASIA-OLHWDB 2.0-2.2 (500 writers, 400 train / 100 test) plus about 67k synthetic lines. IAM English appears only in appendix figures.
  - **Compute: 4x A6000. About 16 h for the VAE and about 3 days for the DiT.**
  - Line-level results:

    | Method | Style % | DTW | Speed (characters/s) |
    |---|---|---|---|
    | DiffInk | 77.38 | 1.049 | 58.47 |
    | OLHWG | 44.74 | 1.326 | 0.07 |
    | SDT | 50.51 | 1.270 | 3.35 |
    | WLU | 35.71 | 1.540 | – |

  - **Code: "will be made publicly available"**, with no link.
  - Source: [DiffInk arXiv HTML](https://arxiv.org/html/2509.23624v1)
- [V-search] Related online works named in DiffInk's related-work section:
  - **CHWmaster** (Li et al., 2024): sliding-window RNN for few-shot personalised Chinese handwriting.
  - **DiffWriter** (Ren et al., 2023): diffusion for character-level synthesis.
  - **TrInk** (Jin et al., 2025): Transformer ink generation.
  - **WLU / Write Like You** (Tang & Lian, Computer Graphics Forum 2021): metric-based meta-learning for personalised cursive Chinese.
  - **Liu et al. 2024**: writer/character style disentangled by components.
  - Source: [DiffInk arXiv HTML](https://arxiv.org/html/2509.23624v1)
- [V-search] Search results describe one model (Li et al. 2024) together with Graves 2013 as autoregressive trajectory-point generation. — [DiffInk arXiv](https://arxiv.org/pdf/2509.23624v3)
- [V-search/V-repo] **DeepCalliFont** (Liu & Lian, AAAI 2024):
  - Few-shot Chinese *calligraphy font* synthesis with two branches, image synthesis and writing-sequence generation. They are kept consistent by dual-modality representation learning, a feature-recombination module and a **rasterisation loss**.
  - Pre-trained on large single-modality data.
  - Repo: only an example dataset (PKU Disk). **No licence, no weights and no reference count stated.**
  - Sources: [arXiv 2312.10314](https://arxiv.org/pdf/2312.10314); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28168); [repo](https://github.com/lsflyt-pku/deepcallifont)
- [V-paper] FontRNN, DeepImitator, WriteLikeYou-v1/v2 and "Drawing" are all baselines in SDT's Table 1. All scored well below SDT on Chinese style (46-72 against 94.5). — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-search] Arabic: no recent neural Arabic online-*generation* model was found. Prior Arabic online synthesis is the Beta-elliptic / velocity-profile family (Alimi group) and patents describing superposed beta velocity profiles. A survey remark notes that movement-simulation needs tablet-captured online data, while shape-simulation suits offline data. — [US10140262B2](https://patents.google.com/patent/US10140262); [Pattern Recognition 48 (2015)](https://faculty.kfupm.edu.sa/ee/malek/research/PR-48-3-J15.pdf)

### Inferences
- For a font, the useful pieces of the trajectory line are (i) SDT's offline-image style plus character-level trajectory output, and (ii) the diffusion trend:
  - OLHWG / DiffWriter: 1D U-Net over point sequences.
  - DiffInk: latent DiT over a sequence VAE.
  
  The diffusion models outperform SDT in third-party tables, but they are Chinese-only, line-oriented, and DiffInk has no code.
- Every trajectory model outputs a centreline only. Stroke width and ink texture are lost (as One-DM points out). That is acceptable or even desirable for a single-line or stroked font, but a filled outline font needs a stroking or width model on top.
- A model's training data must be *online* with writer IDs. Inference can be offline-image-conditioned only for SDT and Luhman 2020. DiffInk and probably OLHWG need online references (OLHWG not confirmed).

### Gaps
- [Unverified, not checked this session] DeepWriting (Aksan et al., CHI 2018; conditional VRNN on IAM-OnDB/Deepwriting dataset, separates content and style), Graves 2013 (LSTM + MDN, IAM-OnDB, style by priming), FontRNN (Tang et al., Pacific Graphics 2019, Chinese font via RNN with monotonic attention), and StrokeGAN (stroke-label-conditioned image GAN, not trajectories). The details are from memory and should be confirmed before citing.
- Handwriting Transformers (HWT, Bhunia et al., ICCV 2021) is an **offline image** model. It is included only to distinguish it from the trajectory models. Not re-verified.
- Licences for OLHWG, DeepCalliFont and Luhman's repo were not stated or not checked. Pretrained weights for OLHWG and DeepCalliFont were not found.
- "Write like a calligrapher" was not located.
- The DiffInk truncated appendix covers paragraph-level and other-language extensions. It was not read.

## Evidence that trajectory models imitate idiosyncratic letter shape better than image models

### Takeaway
No direct head-to-head (trajectory vs. image model, same writers, human or classifier judging shape) was found. The evidence is within-family only: SDT beats other trajectory models on Chinese style-classifier accuracy and a user study. For Latin letters, even the evaluation classifier can barely identify writers from single online characters (20.57%).

### Cited Findings
- [V-paper] SDT reports a Style Score of 94.50 and 56.67% user preference on Chinese, but all its baselines are trajectory or sequence methods (FontRNN, DeepImitator, WLU, Drawing). No offline image generator was compared. — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] The English writer identifier reached 20.57% accuracy, so per-character Latin style is weakly identifiable even for the evaluator. — [SDT paper](https://ar5iv.labs.arxiv.org/html/2303.14736)
- [V-paper] DiffInk's style metric is also a writer classifier on online data. Its best line-level Style is 77.38%. No image-model comparison is reported in the visible text. — [DiffInk](https://arxiv.org/html/2509.23624v1)
- [V-search] One-DM (an image model) argues that offline generation captures stroke width and ink colour, which online methods cannot. These are appearance attributes rather than skeleton shape. — [One-DM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf)

### Inferences
- The likely claimed advantage of trajectory models is that they represent shape as a skeleton, free of ink weight and texture. That matters for a project already using a weight-fair, centreline-based score. It is plausible, but there is no published human study isolating it.
- Rendered-image style inputs in SDT mean "shape" is learned from thin rendered strokes. Real photos would need skeletonisation to match.

### Gaps
- No human study or style-classifier comparison between trajectory and image generators on the same writers was found.
- No study of few-shot Latin or Arabic letter-shape fidelity for trajectory models with fewer than 15 references was found.
