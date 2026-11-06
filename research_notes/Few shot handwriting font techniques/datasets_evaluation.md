# Real handwriting datasets and evaluation practice for a Latin few-shot handwriting font generator

Researched 2026-11-06. "Verified" means I read it in the primary document (NIST user guides, the EMNIST paper, the HWD paper, the HWT paper, DeepWriting, the CASIA page). Everything else is marked as coming from search summaries or secondary pages. All URLs were reached on 2026-11-06. The FKI/IAM site (fki.tic.heia-fr.ch) refused the connection that day (ECONNREFUSED), so IAM facts come from search snippets of its pages.

## Q1. Which datasets have many writers each writing many or all individual Latin characters? (SD19 in detail, then the rest)

### Takeaway
NIST SD19 is still the only large offline dataset in which thousands of writers each write the full a–z and A–Z set once in isolated boxes, with checked labels and writer IDs. Its by_write labels do not need image matching. The 1st-edition release (still downloadable) ships a `.cls` label file for every `.mis` file, including by_write, and the `.mit` files in by_field/by_class point back to exact by_write indices. No other Latin dataset found offers more than ~1 full alphabet per writer at scale. EMNIST drops writer IDs and re-centres and rescales every glyph. Kaggle "A–Z" sets have no writer IDs. CASIA-HWDB (Chinese) includes the 52 Latin letters for 1,020 writers.

### Cited Findings

**NIST SD19, verified from both user guides**
- 3,669 HSF full-page forms (the 1st-edition guide abstract says 3,699; the 2nd-edition guide says 3,669) and 814,255 segmented characters. Each character "occupies a 128x128 pixel raster", in 62 classes (0–9, A–Z, a–z). 2nd Edition: Sept 2016, all images converted to PNG (3,992,357 PNG files in 5 zips). 1st Edition: March 1995, IHEAD binary format. — [SD19 2nd ed. User's Guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf); [SD19 1st ed. User's Guide](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- Forms were scanned at 11.8 dots/mm (300 dpi), binary. Each has 34 fields: name/date, city/state, 28 digit fields, **one lowercase field (field 31), one uppercase field (field 32)**, and a free-text U.S. Constitution preamble paragraph (field 33). — [1st ed. guide](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- There are 100 form templates. They differ only in "a different random ordering of characters intended for the digit, upper and lower case fields". The example reference file gives field 31 = `gyxlakpdsbtzirumwfqjenhocv` and field 32 = `ZXSBNGECMYWQTKFLUOHPIRVDJA`. So **each writer was asked to print each lowercase and each uppercase letter exactly once** in the boxed fields. Reference files (`hsf_page/truerefs/ref_XX`) hold the intended text. They are in the 1st edition and are **not** included in the 2nd edition. — [1st ed. guide](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf); [2nd ed. guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf)
- Totals by field: digits 402,953; uppers 81,463; lowers 81,096; Constitution 248,743. The Constitution field exists only for hsf_0, hsf_1 and hsf_2 (roughly 1,500 writers). Per-partition uppers/lowers: hsf_0 10,790/10,968; hsf_4 11,941/12,000; hsf_7 12,092/11,578. hsf_5 (83,609 chars) is held back by NIST. — [1st ed. guide, Tables 3–4](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- Writers with at least one checked character per field: hsf_0 upper 488 / lower 490 of 500; hsf_4 500/500; hsf_6 499/498; hsf_7 500/499. hsf_8 (70 writers) has forms only. Its character fields were "completed but not processed". — [1st ed. guide, Table 4](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- Writer origin: hsf_4 are Bethesda high-school students; the others are Census Bureau employees (field staff nationwide for hsf_0–3, Suitland MD for hsf_6–8). NIST earmarks hsf_4 as the standard test set. — [2nd ed. guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf); [1st ed. guide](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- Labels: characters were automatically classified, then "manually checked... twice, each time by a different person", with residual error "about 0.1%". The checkers deliberately kept ambiguous shapes "that could plausibly have been the intended class". — [1st ed. guide](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- **by_write labels exist in the 1st edition.** "Each writer directory contains files for each field type; digit, upper, lower and... constitution alphas. All images are in the .mis format and are accompanied by a .cls class reference file." and "All .mis files on Special Database 19 have an associated .cls file." The `.mit` files in the other hierarchies give "a pathname to the 'parental' .mis file and a zero-oriented index into that file". — [1st ed. guide §4.2, §6.3–6.4](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)
- The 2nd edition names PNGs after the `.mis` file plus a zero-based index (e.g. `d0000_14.mis` with 235 chars becomes `d0000_14_00000.png` … `d0000_14_00234.png`). The 2nd-edition guide does not mention `.cls` files for by_write. — [2nd ed. guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf)
- The NIST page lists both the 2nd Edition zips (by_class, by_field, by_merge, by_write, hsf_page) and **`1stEdition1995.zip`**, each with an MD5. The page was updated 9 Jan 2025 and shows no price. DOI 10.18434/T4H01C. — [NIST SD19 page](https://www.nist.gov/srd/nist-special-database-19)
- The 1st-edition software includes `normmis`, which "performs size and orientation normalization of the images in .mis files", and a PNG converter whose output is "typically sized 128 width and 128 height". — [1st ed. guide §7](https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf)

**EMNIST, verified from the paper (arXiv 1702.05373, 2017)**
- Conversion: Gaussian blur, then a fitted bounding box is extracted, "centered in a square frame with lengths equal to the largest dimension, with the aspect ratio... preserved", padded by 2 px, and bicubically downsampled to 28×28 8-bit grayscale. So **absolute size, position and baseline are discarded**. — [EMNIST paper](https://arxiv.org/pdf/1702.05373v1)
- Only the By_Class and By_Merge hierarchies were used. By Author is noted as a future opportunity ("beyond the scope of this work"). Released splits: ByClass and ByMerge (814,255), Balanced (131,600, 47 classes), Letters (103,600, 37 → 26 classes). Train and test were re-drawn randomly, mixing students and Census staff. — [EMNIST paper](https://arxiv.org/pdf/1702.05373v1)
- The EMNIST paper states SD19 has "over 500 writers". That contradicts the NIST guides (3,669 forms, writers numbered up to 4169). — [EMNIST paper](https://arxiv.org/pdf/1702.05373v1) vs [2nd ed. guide](https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf)
- FEMNIST (LEAF) re-derives EMNIST from SD19 **partitioned by writer**. The TensorFlow Federated version splits each writer's examples across train and test and excludes writers with fewer than 2 examples. Hugging Face `flwrlabs/femnist` keeps everything in one train split with writer IDs. This is from search summaries; the LEAF code was not inspected. — [HF flwrlabs/femnist](https://huggingface.co/datasets/flwrlabs/femnist); [TFF emnist.load_data](https://tensorflow.google.cn/federated/api_docs/python/tff/simulation/datasets/emnist/load_data?authuser=4)

**Other datasets**
- **Kaggle "A–Z Handwritten Alphabets in .csv" (Sachin Patel, 2018):** ~372,450 images, 26 uppercase classes, reported as derived from NIST SD19 and other sources. No writer IDs were found in any source. Secondary sources only. — [baselight mirror](https://baselight.app/u/kaggle/dataset/sachinpatel21_az_handwritten_alphabets_in_csv_format); [hyper.ai](https://hyper.ai/en/datasets/32507)
- **IAM Handwriting DB 3.0:** 657 writers, 1,539 scanned pages. Segmented at form, line, sentence and word level; no character level was found. Registration is required. "Freely available for non-commercial research purposes". Line count is 13,353 per Papers with Code vs 9,862 in the official split table. — [IAM page](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (unreachable 2026-11-06; facts from search snippet); [Voxel51 mirror](https://docs.voxel51.com/dataset_zoo/datasets_hf/iam_handwriting_finevision.html)
- **CVL Database (TU Wien, ICDAR 2013):** 310 writers (later papers say 311). Texts: 7 texts (27 writers) or 5 texts (284 writers), with 1 German and 6 English texts. Word bounding boxes are in XML. License CC BY-NC 3.0, non-commercial research. Download is on the TU Wien page. — [CVL page](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/); [ICDAR paper](https://cvl.tuwien.ac.at/wp-content/uploads/2014/12/kleber-icdar132.pdf)
- **CASIA-HWDB / OLHWDB (Chinese, for comparison):** 1,020 writers (Anoto pen on paper, so both online and offline). Isolated-character sets DB1.0–1.2 hold ~3.9M samples in 7,356 classes (7,185 Chinese + 171 symbols). **The symbols include the 52 English letters and 10 digits.** Train:test writers are 4:1. Access requires an application form ("Conditions of Use"). — [CASIA page](https://nlpr.ia.ac.cn/databases/handwriting/home.html)
- **Omniglot:** 1,623 characters from 50 alphabets, each drawn by 20 people on Mechanical Turk. Images plus [x,y,t] strokes. Background split has 30 alphabets, evaluation split 20. MIT license. Not Latin. — [brendenlake/omniglot via ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fbrendenlake%2Fomniglot); [TFDS omniglot](https://www.tensorflow.org/datasets/catalog/omniglot)
- **UNIPEN Train-R01/V07 (Dec 1999):** online data with isolated digits, isolated uppercase, isolated lowercase and mixed-case categories. Hosted on Zenodo. — [Zenodo record 1195803](https://zenodo.org/records/1195803)
- **IRONOFF (LORIA):** dual online and offline database of isolated characters, digits and words in French and English, in UNIPEN format. Reported at ~700 writers (secondary source). Has a sub-corpus of isolated lowercase and uppercase letters. — [search summary; LipiTk resources](https://lipitk.sourceforge.net/resources.htm)
- **English online set, 1,020 writers × 52 letters (53,248 chars):** described in "Disentangling Writer and Character Styles for Handwriting Generation" (SDT, CVPR 2023). The dataset's name and origin were not confirmed. It may be the Latin part of CASIA-OLHWDB, which would match the 1,020-writer figure; that is an inference. — [arXiv 2303.14736](https://arxiv.org/pdf/2303.14736)
- **Ukrainian (Cyrillic) word dataset, 2026:** 126K samples from 308 writers. Not Latin. — [arXiv 2605.27487](https://arxiv.org/html/2605.27487)
- A 2025 font paper states that no public dataset of the essential parts of each font exists. This concerns printed fonts. — [Few-Part-Shot Font Generation, arXiv 2509.10006](https://arxiv.org/html/2509.10006)

### Inferences
- From SD19 one should expect about one sample of each of the 52 letters per writer (uppers 81,463 / about 3,600 writers ≈ 22–23 of 26 surviving segmentation and checking), plus roughly 100+ digits. About 1,500 writers (hsf_0–2) add Constitution-box letters: many e/t/o/s and few rare letters.
- **Practical label route:** download `1stEdition1995.zip` and read `by_write/hsf_*/f*/u*.cls` and `l*.cls` alongside the `.mis` entries. Because the 2nd-edition PNG index equals the `.mis` entry index, these labels transfer one-to-one to the 2nd-edition by_write PNGs. This replaces image matching against by_field. It should be confirmed on one writer before relying on it.
- **Position and size inside the 128×128 raster: not documented in either guide.** The existence of `normmis` (a size-normalisation tool) implies the stored rasters are *not* size-normalised. That would mean relative scale between a writer's letters is probably kept. Whether vertical offset relative to a shared baseline is kept (glyph placed by its box position vs centred) is not stated. Check empirically: compare bounding-box centroids of "g/y/p" vs "x/o" within one writer. Note also that each letter was written in a separate box of a random-order field, so there is no true line baseline to recover except from the hsf_page form images.
- SD19 is boxed, hand-printed block capitals and lowercase by US Census staff and students in 1990–1995. Writer idiosyncrasy is present but constrained by box-printing. It will not show cursive joins.
- Only SD19, CASIA (52 letters), the unnamed 1,020-writer English online set, UNIPEN and IRONOFF give per-writer full alphabets. IAM, CVL, BRUSH and DeepWriting are text and give uneven per-letter coverage.

### Gaps
- No licence text was found for SD19. The NIST page shows only a generic copyright link. NIST SRD data is distributed free, but I did not verify terms of reuse or redistribution of derived images.
- Exact placement rule of characters in the 128×128 raster (centred vs box-relative) was not found in the guides.
- UNIPEN and IRONOFF writer counts, per-writer letter coverage and 2026 licence/availability were not verified from primary pages. IRONOFF's ~700-writer figure is from a secondary source.
- RIMES (French, mail) was not researched in this pass. As far as I know it is word- and line-level with no per-writer alphabets; this is unverified.
- No 2023–2026 public Latin dataset of per-writer full alphabets or "handwriting fonts" was found in two searches.
- The current licence of the Kaggle A–Z set, and whether it contains NIST-derived writer splits, is unknown.

## Q2. Can word-level datasets (IAM, CVL) yield labelled per-character crops?

### Takeaway
Not directly. IAM and CVL give word (and line) boxes only. Character crops need forced alignment (HMM or CTC alignment from an HTR model) or a commercial segmenter. Published work reports that HMM alignment often misplaces boundaries. DeepWriting is the one Latin dataset that ships character-level segmentation, and it is online.

### Cited Findings
- CVL stores per-word bounding boxes in XML with writer IDs. No character boxes are mentioned. — [CVL ICDAR 2013 paper](https://cvl.tuwien.ac.at/wp-content/uploads/2014/12/kleber-icdar132.pdf)
- No public character-level bounding-box annotation for IAM was found. The Keras IAM example uses word boxes. — [search summary; Keras example](https://embed.cocalc.com/github/keras-team/keras-io/blob/master/examples/vision/md/handwriting_recognition.md)
- CTC alignment can match transcriptions to line images by dynamic programming over CTC outputs. This was used to improve annotations of the Bullinger letters (2025). — [arXiv 2508.07904 summary](https://fugumt.com/fugumt/paper_check/2508.07904v1_enmode)
- HMM forced alignment can segment handwriting into characters without ground-truth boundaries, but "often fails to segment the basic units to their actual boundaries". — [arXiv 2307.00664](https://arxiv.org/html/2307.00664v1)
- DeepWriting segmented its IAM-OnDB portion to character level with a commercial tool (MyScript), then manually cleaned the results. — [DeepWriting, arXiv 1801.08379](https://ar5iv.labs.arxiv.org/html/1801.08379)

### Inferences
- For cursive IAM/CVL words, character crops will cut through ligatures, and CTC peaks are not boundaries. Crops would be noisy and not comparable to the isolated 128×128 glyph format. For a glyph-level font generator, IAM/CVL are better used as a style or writer-ID signal (e.g. a style encoder or evaluator) than as glyph targets.

### Gaps
- No published, downloadable IAM character-segmentation ground truth was found.

## Q3. Which online (pen-trajectory) datasets can be rendered to glyphs with a chosen pen width, and which give full alphabets per writer?

### Takeaway
Trajectories render cleanly at any pen width and baseline. But BRUSH, IAM-OnDB and DeepWriting are word or line text, so letter coverage per writer is uneven. Isolated-letter online sets with per-writer alphabets are UNIPEN (1a/1b/1c categories), IRONOFF, CASIA-OLHWDB (52 Latin letters, 1,020 writers) and Omniglot (non-Latin). Licences are mostly non-commercial.

### Cited Findings
- **BRUSH (Kotani, Tellex, Tompkin, ECCV 2020):** 27,649 online samples from 170 writers. Points are labelled with intended characters, so strokes map to characters. 488 common words were written by all writers plus 3,668 rarer words. "May only be used for non-commercial research purposes". On GitHub as a zip. — [DSD GitHub](https://github.com/brownvc/decoupled-style-descriptors); [Papers with Code BRUSH](https://paperswithcode.com/dataset/brush); [DSD slides](https://dsd.cs.brown.edu/files/presentation.pdf)
- **IAM-OnDB:** handwritten lines from 221 writers on a smart whiteboard (infrared pen tracking), transcribing LOB-corpus text. — [IAM-OnDB page](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database); [Graves 2013](https://arxiv.org/pdf/1308.0850)
- **DeepWriting (ETH, 2018):** 294 authors (200 from IAM-OnDB after removing 21, plus 94 new on iPad Pro). 85,181 words, 406,956 characters, median 1,349 characters per writer, character-level segmentation. "Data and code are publicly available". No licence stated in the paper. — [DeepWriting, arXiv 1801.08379](https://ar5iv.labs.arxiv.org/html/1801.08379)
- UNIPEN, IRONOFF, CASIA-OLHWDB and Omniglot: see Q1.

### Inferences
- DeepWriting's median of ~1,349 characters per writer means common letters appear many times per writer, while q/z/x and capitals are sparse. Rendering at a project-chosen pen width lets one match SD19 stroke weight. Whiteboard samples are low-precision per the authors.
- Online data can preserve the true baseline and x-height within a line. That is a property SD19 boxes cannot give.

### Gaps
- Per-writer letter-coverage statistics for BRUSH and IAM-OnDB were not found.
- The 2026 download status and licence of DeepWriting and IRONOFF were not verified.

## Q4. How are few-shot font and handwriting generators evaluated, and which metrics track "same hand"?

### Takeaway
Papers report FID/KID, plus L1/SSIM/LPIPS for paired font glyphs. For handwriting, two style-specific measures are emerging. HWD (BMVC 2023) is a writer-level feature distance; it separates same and different authors far better than FID and is stable at about 100 samples. Writer-classifier accuracy and retrieval (HTGstyle, mAP/top-1) is the other. No paper found validated any of these against human "same hand" judgements. Generic perceptual metrics such as LPIPS fall short of human 2AFC agreement even on natural images.

### Cited Findings
- **HWD (Pippi, Quattrini, Cascianelli, Cucchiara; BMVC 2023; arXiv 2310.20316, 31 Oct 2023):** a VGG16 pretrained to classify calligraphic fonts on "Font2", which renders 10,400 words in 10,400 fonts (more than 100M samples). Images are resized to height 32; per-column 512-d features; per writer, Euclidean distance between mean real and mean generated feature vectors; dataset HWD is the mean over writers. — [HWD paper](https://arxiv.org/html/2310.20316v1); code [aimagelab/HWD](https://github.com/aimagelab/HWD)
- HWD vs FID for same- vs different-author discrimination on IAM: FID Overlap 27.1% / EER 13.6%, HWD 0.7% / 0.3%. FID needs about 2,000 samples to plateau, HWD about 100. HWD rises roughly linearly with shear, erosion and dilation. **There is no user study or human-correlation analysis.** It is trained on English. — [HWD paper](https://arxiv.org/html/2310.20316v1)
- **Writer-ID accuracy:** WordStylist trains a ResNet18 writer classifier on real IAM. Generated samples score 70.67% vs under 5% for GAN baselines. — [WordStylist, arXiv 2303.16576](https://arxiv.org/pdf/2303.16576)
- **HTGstyle:** accuracy of a style classifier trained on real data, tested on generated data for the same writers. — [Rethinking HTG Evaluation, arXiv 2409.02683](https://arxiv.org/html/2409.02683v1)
- Learning-free retrieval (leave-one-sample-out mAP and top-1 over writer embeddings) is used for paragraph imitation. — [arXiv 2409.00786](https://arxiv.org/pdf/2409.00786)
- BoundInk (2026) reports writer-classifier top-1 rising from 35.8 to 61.7 against DeepWriting. — [arXiv 2604.02103](https://arxiv.org/pdf/2604.02103)
- DSD (BRUSH) reports writer identification on held-out writers from a single word as evidence of style capture. — [DSD project](https://dsd.cs.brown.edu/)
- HWT reports FID averaged across writers, plus a geometric score (GS). — [HWT](https://ar5iv.labs.arxiv.org/html/2104.03964)
- **LPIPS vs humans (BAPPS 2AFC):** human cross-rater agreement 73.9%. LPIPS-Alex 69.8% and LPIPS-VGG 68.1% on the full validation set. These are natural-image patches, not glyphs. — [Zhang et al. 2018](https://arxiv.org/pdf/1801.03924.pdf); [Attacking Perceptual Similarity Metrics](https://arxiv.org/pdf/2305.08840)

### Inferences
- For "same hand", a writer-embedding metric (HWD-style mean-feature distance, or retrieval of the true writer among distractors) is the closest automatic proxy. The HWD backbone works on word images at height 32, not 128×128 glyphs, so it would need retraining or adaptation for glyph sets. A glyph-level writer classifier trained on SD19 writers, not on the 10 test writers, could serve the same role.
- clDice measures stroke topology against a single ground truth. It does not test whether a glyph is plausibly by the same writer, so it will not necessarily predict gate-3 outcomes.

### Gaps
- No study was found that correlates FID, HWD, LPIPS or writer-ID scores with human "same writer" judgements at glyph level.

## Q5. How are human two-alternative / real-vs-fake studies run in font and handwriting papers, with what sample sizes, and what are the pitfalls?

### Takeaway
Handwriting-generation papers use (a) a *plausibility* test, where each sample is labelled real or forged after seeing the writer's real handwriting, and (b) a *preference* test, picking which method best mimics the writer. Reported plausibility accuracy sits at about 48–54%, near chance, with 28–100 participants. Font papers mostly use preference votes among methods with 20–64 participants. A strict "same hand?" 2AFC on glyphs was not found. HYPE gives the most rigorous real/fake protocol: 50% real, scored as error rate, bootstrapped.

### Cited Findings
- **HWT (ICCV 2021):** 100 participants. Preference study: 10 questions each (1,000 responses), with a real paragraph plus HWT, Davis et al. and GANwriting outputs in random order; HWT chosen 81%, GANwriting 10%, Davis 9%. Plausibility study: 60 responses each (6,000). Participants saw the writer's real handwriting, then six samples per trial, three of them genuine, and labelled each real or forged, with no time limit. Accuracy 48.1%. Confusion: real→real 24.9%, real→fake 25.1%, fake→real 26.8%, fake→fake 23.2%. Generated samples came from unseen IAM test writers. — [HWT, ar5iv 2104.03964](https://ar5iv.labs.arxiv.org/html/2104.03964); [HWT supplementary](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Bhunia_Handwriting_Transformers_ICCV_2021_supplemental.pdf)
- **One-DM (2024):** participants were first shown 30 real samples, then classified images (mixed real training and generated) as authentic or synthetic. Accuracy 49.05%. A preference study was also run. — [One-DM, arXiv 2409.04004](https://arxiv.org/html/2409.04004v1)
- **CONSTANT (WACV 2026 supplement):** 28 participants, 1,680 responses. In each group of 6 images, pick the 3 real ones after seeing examples from the same writers. Accuracy 53.8%. — [CONSTANT supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Le_CONSTANT_Towards_High-Quality_WACV_2026_supplemental.pdf)
- **Font papers:** Diff-Font had 64 participants pick the best method on style and content. MX-Font showed 5 methods in random order per query and said it trusts the user study more than metrics. MX-Font++ used 20 volunteers. FontDiffuser and DA-Font report a "User" percentage column. A 2025 survey describes the typical format as choosing the most preferred among methods. — [Diff-Font](https://arxiv.org/pdf/2212.05895); [MX-Font](https://arxiv.org/pdf/2104.00887); [MX-Font++](https://arxiv.org/pdf/2503.02799); [FontDiffuser](https://arxiv.org/pdf/2312.12142); [DA-Font](https://arxiv.org/pdf/2509.16632); [survey](https://arxiv.org/html/2508.06900v1)
- **HYPE (NeurIPS 2019):** a standardised real/fake protocol on Mechanical Turk with 50% real images. HYPE-∞ (no time limit) reports the error rate, where 50% means indistinguishable. HYPE-Time adapts exposure time. Rankings were checked with bootstrap. A reviewer noted it measures realism, not diversity. — [HYPE](https://arxiv.org/pdf/1904.01121); [NeurIPS reviews](https://papers.neurips.cc/paper_files/paper/2019/file/65699726a3c601b9f31bf04019c8593c-Reviews.html)

### Inferences
Pitfalls that bear on gate 3:
- **Chance-level accuracy is ambiguous.** Near-50% can mean fakes are good or that raters cannot tell the writers apart at all. HWT's confusion matrix shows real samples called fake 50% of the time. Include a control: real glyphs from a *different* writer versus the target writer. Without it, raters' baseline ability is unknown.
- **"≥70% judged same hand" needs a defined denominator and baseline.** Report the rate at which genuine held-out glyphs of the same writer are judged "same hand" (ceiling) and the rate for another writer's glyphs (floor). Also report a binomial confidence interval. For example, 100 judgements at 70% give a 95% CI of about ±9 percentage points (normal approximation), so several hundred judgements per condition are needed to separate 70% from 60%.
- Randomise left/right position and order, hide which side is real, and impose no or fixed time limits (HYPE-∞ style). Use several raters and report inter-rater agreement (BAPPS shows humans agree only about 74% even on easy perceptual 2AFC).
- Single isolated glyphs carry less writer information than words or paragraphs, which HWT, One-DM and CONSTANT all used. Expect lower discriminability, which inflates "plausible" rates.
- Rater familiarity with the writer matters. One-DM showed 30 real samples first and HWT showed the real paragraph. Fix and report what reference set raters see.
- Keep the 10 photographed writers' images out of any public form, consistent with the project rule against publishing real handwriting.

### Gaps
- No paper found ran a glyph-level "same hand" 2AFC with reported sample-size justification.
- The DA-Font and FontDiffuser user-study protocol details were not read (snippets only).
- No source was found on online-form tooling specific to these papers (Google Forms vs MTurk) beyond HYPE's MTurk.
