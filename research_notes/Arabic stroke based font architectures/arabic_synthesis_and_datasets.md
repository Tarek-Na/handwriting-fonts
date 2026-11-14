# Arabic handwriting synthesis/generation and Arabic handwriting datasets (2015-2026)

Scope note: about 25 search/fetch calls. Several primary PDFs (Elarian 2015 Pattern Recognition, the IJACSA 2025 calligraphy-generation review, the DiffusionRec PDF) were downloaded but could not be read: no PDF text tools on this machine, and a TLS error on the NUST host. Claims about them come from search snippets and abstracts, and are marked as such. Anything I know from background but could not verify in this session is in the Gaps sections, not in Cited Findings.

## Papers and systems on Arabic (and same-script) handwriting / font generation

### Takeaway
Arabic handwriting generation splits into two lines of work:
- **Classical concatenation synthesis (2013-2015).** Saabni & El-Sana; the KFUPM work of Elarian et al. plus its patents; Dinges et al. These systems glue per-writer positional-form glyphs together. They explicitly model connection points and extensions (kashida-like joins). They are the closest prior art to "write each positional form once, get a font".
- **Image-based deep generators (2021-2025).** ScrabbleGAN-style augmentation, cDCGAN for isolated characters, Arabic-One-DM diffusion, and DiffusionRec for Urdu Nastaliq. These output word images, not strokes or fonts.

I found no published deep-learning model that generates Arabic *online strokes* in a writer's style. I also found no few-shot font-generation paper evaluated on Arabic contextual forms.

### Cited Findings
**Classical / concatenation synthesis**
- **Saabni & El-Sana (IJDAR 2013), "Comprehensive synthetic Arabic database for on/off-line script recognition research".** Generates an on-demand synthetic database for online and offline recognition. Each word gets multiple shapes meant to represent multiple personal handwriting styles. Offline images are made from online strokes by dilation, "to resemble pen's path". A later survey summarises the method as using two concatenation models, Extended-Glyph and Synthetic-Extension. — [CRIS IUCC record](https://cris.iucc.ac.il/en/publications/comprehensive-synthetic-arabic-database-for-onoff-line-script-rec-3/); [Springer PDF mirror](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s10032-012-0189-5.pdf); [PeerJ CS 861 related-work summary](https://peerj.com/articles/cs-861)
- **Elarian, Ahmad, Awaida, Al-Khatib, Zidouri, "An Arabic handwriting synthesis system", Pattern Recognition 48 (2015) 849-861** (DOI 10.1016/j.patcog.2014.09.013). Synthesises Arabic words from segmented character shapes using two concatenation models, Extended-Glyphs connection and Synthetic-Extensions connection. Training a recogniser on the collected plus synthesised data improved results on the IFN/ENIT benchmark. I could not read the exact numbers. — [KFUPM PURE](https://pure.kfupm.edu.sa/en/publications/an-arabic-handwriting-synthesis-system/); [author PDF](https://faculty.kfupm.edu.sa/ee/malek/research/PR-48-3-J15.pdf)
- **Elarian's 2014 KFUPM PhD thesis, "Arabic Handwritten: Analysis and Synthesis".** Segmentation yields two sets of character shapes: strictly segmented ones and *extended* ones. The extended shapes are placed side by side so their extensions meet. This is directly relevant to joining glyphs in a personal font. — [KFUPM eprints](https://eprints.kfupm.edu.sa/id/eprint/139204/)
- **KFUPM / Elarian-family USPTO patents:**
  - US 10055659, "Systems and associated methods for Arabic handwriting using synthetic extension concatenation and synthesis"
  - US 10083362, "Arabic handwriting synthesis system and method"
  - US 10140262, "Systems and associated methods for Arabic handwriting synthesis and dataset design"
  - US 10185882, "Method for synthesizing Arabic handwritten text"
  - US 10242276, "Arabic script analysis with connection points"
  - US 10467330, "System for analysis and reproduction of text data"

  They describe choosing character samples by connection-point features and width statistics. One reports connection-point location error rates of 1.64% (right side) and 2.12% (left side). The patents name personalised fonts, recogniser training and CAPTCHA generation as uses. They say Arabic synthesis is hard because of script complexity and scarce data. — [US10083362](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10083362); [US10055659](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10055659); [US10185882](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10185882); [US10140262 (Google Patents)](https://patents.google.com/patent/US10140262); [US10242276](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10242276)
- **Dinges, Al-Hamadi, Elzobi et al., "ASM Based Synthesis of Handwritten Arabic Text Pages", The Scientific World Journal 2015 (open access).**
  - Character synthesis uses Active Shape Models trained on 28,046 *online* samples.
  - Statistics from IESK-arDB are used to simulate baseline, slant and skew.
  - Words and pages are composed from ASM representations, smoothed by B-spline interpolation, and rendered with writing speed and pen characteristics taken into account.
  - The synthetic data was used to validate a segmentation method.

  — [PMC4534626](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4534626/); [earlier MC07 paper](https://www.csmc.uni-hamburg.de/publications/mc/files/articles/mc07-dinges-al-hamadi-elzobi-el-etriby.pdf)
- **Laslo Dinges, PhD dissertation (Magdeburg, 2019, in German).** Synthesis of Arabic handwriting for recognition support. It was used to augment IESK-arDB. — [opendata.uni-halle.de PDF](https://opendata.uni-halle.de/bitstream/1981185920/32647/1/Dinges_Laslo_Dissertation_2019.pdf)

**Deep generative (image-based)**
- **Eltay, Zidouri, Ahmad et al., PeerJ Computer Science 2022 (cs-861), "Generative adversarial network based adaptive data augmentation for handwritten Arabic text recognition".** Uses a ScrabbleGAN-based architecture, with a text-recognition auxiliary network, to generate handwritten Arabic word images. The GAN is forked from Fogel et al.'s ScrabbleGAN, and output is shown on IFN/ENIT text. The goal is recognition augmentation, not writer imitation. — [PeerJ](https://peerj.com/articles/cs-861/); [ResearchGate](https://www.researchgate.net/publication/358125571_Generative_adversarial_network_based_adaptive_data_augmentation_for_handwritten_Arabic_text_recognition)
- **"Conditional deep convolutional generative adversarial networks for isolated handwritten Arabic character generation"** (cDCGAN). I only saw it cited in a reference list. A separate cited paper on DCGANs for Urdu/Arabic handwriting in resource-scarce languages was also seen only as a reference. — [systematic review arXiv 2507.06275](https://arxiv.org/pdf/2507.06275)
- **Abushahla, Panopio, Elfattal, Zualkernan, "One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation"** (IEEE venue linked from the README; year not confirmed, citation key says 2025).
  - Adapts One-DM (arXiv 2409.04004, a one-shot diffusion mimicker built on Stable Diffusion v1-5) to Arabic. It generates word images in a writer's style from one reference sample.
  - **Training data:** a merge of IFN/ENIT, AlexU-Word, AHAWP words and IESK-ArDB, giving 62,437 word images, 1,412 unique words and 1,417 writers. The README is internally inconsistent about whether it is 3 or 4 datasets.
  - **Content glyphs:** the content input uses GNU Unifont 16×16 glyphs rendered in isolated/initial/medial/final form. A heuristic `shape_arabic_text` uses a non-joining-letter list (ا د ر و …), with a dummy س to force joining.
  - **Diacritics and dots:** harakat, hamza and shadda can be rendered. The README does not say how dots (i'jam) are handled.
  - **Release:** code is released (train, fine-tune, test, evaluation scripts). The README states no quantitative results, no clear pretrained-weight download, and no licence.

  — [GitHub 7abushahla/Arabic-One-DM](https://github.com/7abushahla/Arabic-One-DM); [One-DM arXiv 2409.04004](https://arxiv.org/pdf/2409.04004)
- **Arabic calligraphy generation review (IJACSA vol. 16 no. 3, 2025, "Handwritten Arabic Calligraphy Generation").** A review of 19 studies. It reports that GANs are the main deep-learning approach for Arabic calligraphy generation, and that research on realistic *handwritten* calligraphy is lacking. Only the snippet was readable. — [IJACSA PDF](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)
- **Nastaliq GAN art project.** A paper notes that existing Arabic-script research is "mostly restricted to style transfer and single letter generation". It is an artistic GAN exploration, not a benchmark. — seen in the [few-shot font search results](https://arxiv.org/pdf/2412.11342) (the snippet is attributed there; I did not confirm the exact paper).

**Urdu / Persian (same script)**
- **DiffusionRec (NUST TUKL lab), "Recognition-Guided Diffusion for Content-Aware Urdu Handwriting Generation".** Claims to be the first diffusion framework for Urdu Nastaliq handwriting generation, using recognition-guided latent diffusion for word images. It says earlier Arabic-script methods were mostly GAN-based and that no standard benchmark exists for diffusion-based Urdu word generation. It cites a prior GAN for context-aware Urdu *ligature* generation. Venue and year are not confirmed; the PDF failed with a TLS error. — [NUST PDF](https://tukl.seecs.nust.edu.pk/members/projects/conference/DiffusionRec-Recognition-Guided-Diffusion-for-Content-Aware-Urdu-Handwriting-Generation.pdf)
- **"Urdu Handwritten Ligature Generation Using Generative Adversarial Networks"** (GAN). Seen only as a reference. — [arXiv 2507.06275 references](https://arxiv.org/pdf/2507.06275)
- **"Diffusion-Enhanced NAT-BART Vision Language Transformer…" (Hussain, Hassan et al., TU Dublin).** Uses conditional diffusion to synthesise Urdu word images for recognition, and introduces a real Urdu word dataset. — [TU Dublin ARROW](https://arrow.tudublin.ie/engschelero/26/)
- **WGANs for Urdu character recognition augmentation (Information Research Communications 2025).** — [PDF](https://jourdata.s3.us-west-2.amazonaws.com/irc/InfoResCom-2-1-152.pdf)

**Few-shot font generation and Arabic**
- **"One-Shot Multilingual Font Generation via ViT" (arXiv 2412.11342).** Lists generalising to other writing systems "such as Arabic" as future work, so it was not tested. — [arXiv 2412.11342](https://arxiv.org/pdf/2412.11342)
- Mainstream few-shot font methods are built for Latin or CJK component structure: LF-Font (8 reference glyphs), DM-Font, MX-Font, GAS-NeXt cross-lingual, and DA-Font 2025. None of the search results showed an Arabic evaluation. — [LF-Font](https://arxiv.org/pdf/2009.11042); [DM-Font](https://arxiv.org/pdf/2005.10510); [MX-Font](https://arxiv.org/pdf/2112.11895); [GAS-NeXt](https://arxiv.org/pdf/2212.02886); [DA-Font](https://arxiv.org/pdf/2509.16632)

**Stroke models (generic, not Arabic)**
- **Graves's RNN+MDN handwriting synthesis (arXiv 1308.0850).** Trained on IAM-OnDB (English). — [arXiv 1308.0850](https://arxiv.org/pdf/1308.0850)
- **TrInk (2025).** A transformer encoder-decoder with a 20-component GMM head. — [arXiv 2508.21098](https://arxiv.org/html/2508.21098)
- **DiffInk (2025).** A latent diffusion transformer for text-to-online handwriting (glyph- and style-aware). I did not check whether it covers Arabic. — [arXiv 2509.23624](https://arxiv.org/pdf/2509.23624)
- **Diffusion models for online handwriting (2020).** — [arXiv 2011.06704](https://arxiv.org/pdf/2011.06704)
- No Arabic-specific RNN, transformer or diffusion *stroke* generator showed up in the searches.

### Inferences
- The classical KFUPM / Saabni line already formalises what a personal Arabic font needs: per-position glyphs plus explicit left and right connection points, joined by either "extended glyphs" (the writer's own kashida tail is kept) or "synthetic extensions" (a generated joining stroke). A stroke-based few-shot font system can reuse that decomposition directly.
- All the deep Arabic generators found work on image-level whole words (IFN/ENIT, KHATT and similar). None outputs per-glyph vectors with OpenType positional forms. Arabic-One-DM handles positional forms only on the *content* side, through a heuristic shaper over Unifont glyphs.
- Because no Arabic stroke-generation model exists, the online datasets below are the main way to train or pretrain a stroke model. Of these, Online-KHATT (lines), AltecOnDB (sentences, >1000 writers) and Calliar (calligraphy) are the largest.

### Gaps
- No exact numbers for Elarian 2015, the IJACSA review or DiffusionRec: the PDFs could not be parsed or fetched here.
- From background knowledge, not checked this session: Arabic Pix2Pix / CycleGAN calligraphy works, Persian handwriting GANs, "Arabic-Nougat"-style work, and Calliar-based generation follow-ups. I found no citable source for any Calliar-trained *generator*.
- Patent holders and assignees (KFUPM vs. others) were not checked on each patent.
- Several ICDAR/ICFHR 2023-2025 Arabic HTG papers may exist under other names (e.g. "Arabic HTG with transformers"). My searches did not surface them.

## Datasets: contents, sizes, formats, licences, access

### Takeaway
For few-shot *style* learning, the most useful finds are:
- **AHAWP** (offline, but 82 writers × all 65 positional forms × 10 repeats, CC BY, on Mendeley)
- **Online-KHATT** (online InkML, 623 writers; host is down)
- **ADAB** (online InkML; IEEE DataPort, login needed)
- **Calliar** (online JSON strokes, MIT licence, on GitHub; calligraphy, not everyday handwriting)
- **Hijja / ACRC** (children's Arabic letters with positional forms; whether it is online is disputed)
- **HMBD** (offline positional forms)

Confirmed downloadable today:
- Calliar (GitHub)
- AHAWP (Mendeley)
- ADAB (IEEE DataPort with free login)
- AHCD (Kaggle; listing seen, contents not verified)
- HMBD (GitHub; listing seen)

### Cited Findings
**Online (stroke / trajectory) datasets**
- **Calliar (Alyafeai, Al-shaibani, Ghaleb, Al-Wajih; arXiv 2106.10745; Neural Computing & Applications 2022, DOI 10.1007/s00521-022-07537-2).**
  - Online Arabic calligraphy: 2,500 sentences with more than 40,000 strokes, in styles including Diwani, Thuluth and Farisi.
  - Annotated at stroke, character, word and sentence level. Composite characters are decomposed into primitive strokes.
  - **Format:** 2,500 JSON files, each a list of strokes with a character label and a point list. There is also an RDP-simplified `dataset.npz` (8.6 MB).
  - **Split:** train 2,000 samples (6,065 words, 24,722 characters, 36,561 strokes); valid 250 (738 words, 2,946 characters, 4,410 strokes); test 250 (753 words, 3,052 characters, 4,601 strokes).
  - **Licence:** MIT.
  - **Writer count:** not stated.
  - Rare characters are under-represented, which the authors list as future work.
  - **Access:** DOWNLOADABLE (GitHub).

  — [GitHub ARBML/Calliar](https://github.com/ARBML/Calliar); [arXiv](https://arxiv.org/pdf/2106.10745)
- **Online-KHATT (KFUPM; Open Cybernetics & Systemics Journal 2018, DOI 10.2174/1874110X01812010042).**
  - 10,040 lines of Arabic text by 623 writers, written on Android and Windows devices with pens.
  - Uses the same source texts as offline KHATT. Split 70/15/15.
  - Ground truth at line level, in text, InkML and XML formats.
  - Stated as freely available at onlinekhatt.ideas2serve.net. **That host no longer resolves (DNS ENOTFOUND, checked 2026-11-14), so it is NOT confirmed downloadable today.**

  — [Bentham abstract](https://benthamopen.com/ABSTRACT/TOCSJ-12-42); [KFUPM PURE](https://pure.kfupm.edu.sa/en/publications/online-khatt-an-open-vocabulary-database-for-arabic-online-text-p/)
- **ADAB.**
  - Online Arabic handwritten words from 937 Tunisian town and village names.
  - Each file holds a postcode label (Numeric Character References), an InkML file with the trajectory, and a plot image.
  - Used in the ICDAR 2021 writer-identification competition.
  - **Access:** DOWNLOADABLE from IEEE DataPort (open access, free login), DOI 10.21227/wpf8-dk19.

  — [IEEE DataPort ADAB](https://ieee-dataport.org/open-access/adab-database)
- **AltecOnDB (arXiv 1412.7626).**
  - Large-vocabulary online Arabic database with more than 1,000 writers, covering about 98% of possible PAWs.
  - Complete sentences including digits and punctuation, with sentence-, word- and character-level data.
  - The abstract criticises ADAB's 937-word lexicon as small.
  - Device, format and licence are not stated. **No download location found; described only in the paper.**

  — [arXiv 1412.7626](https://arxiv.org/abs/1412.7626)
- **CHAKEL-DB (2020).**
  - Online database of handwritten Arabic characters with diacritics, collected on a tablet. Each point of the pen trajectory is recorded.
  - Covers the 28 letters plus digits.
  - Snippets give 55 writers and 2,205 characters (17,095 traces?). The table was garbled.
  - **Access not confirmed.**

  — [SciTePress PDF](https://www.scitepress.org/Papers/2020/88043/88043.pdf)
- **OHASD.**
  - Online sentence database: 48 writers, 154 paragraphs, 19,467 characters.
  - **Access not confirmed.**

  — [arXiv 2005.02171 (cites it)](https://arxiv.org/pdf/2005.02171)
- **AHWDB1/AHWDB2.**
  - Online and offline text-dependent writer-ID data: 200 writers × 10 trials of 1-2 words, tablet-captured.
  - **Access not confirmed.**

  — [ResearchGate](https://www.researchgate.net/publication/333783530_New_OnlineOffline_text-dependent_Arabic_Handwriting_dataset_for_Writer_Authentication_and_Identification)
- **ASAR 2021 ACRC / "Hijja".**
  - One summary describes 47,434 characters by 591 children aged 7-12, written on a Wacom UltraPad A4 tablet. Each character appears in up to 3 forms by position in the word.
  - **Conflict:** the original Hijja dataset (Altwaijry & Al-Turaiki 2020) is, as far as I know, distributed as *offline* 32×32 images. Check whether ACRC released trajectories. I did not fetch the original.

  — [ACRC ResearchGate](https://www.researchgate.net/publication/354360116_ASAR_2021_Competition_on_Online_Arabic_Character_Recognition_ACRC); [arXiv 2211.02119](https://arxiv.org/pdf/2211.02119)
- **IESK-arDB (Magdeburg).**
  - Word database of 300 different words by 20 writers. Ground truth stores trajectories for online applications.
  - A sample is downloadable without registration from the group's site, as stated in the 2015 paper. **Current status not checked.**

  — [PMC4534626](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4534626/)
- **Quranic online word database.** An online database of handwritten Quranic words exists. — [ResearchGate](https://www.researchgate.net/publication/262336469_Online_database_of_Quranic_handwritten_words)

**Offline (image) datasets**
- **AHAWP (Majid Ali Khan; Data in Brief 2022; Mendeley Data DOI 10.17632/2h76672znt, v2 dated 21 Feb 2022).**
  - **Contents:** 82 writers. Each writes all **65 Arabic letter variants (isolated, initial, medial and final forms)** and 10 words that together contain every letter, each 10 times, plus 3 fixed paragraphs.
  - **Totals:** 53,199 letter images, 8,144 word images, 241 paragraph images.
  - **Collection:** paper forms scanned at 300 dpi (2480×3507 JPEG raw forms kept). Crops are grayscale with height normalised to 128 px.
  - **Organisation:** per-user and per-letter folders, with filenames `userid_alphabetName_variationName_index`.
  - **Licence:** the article is CC BY 4.0; consent allows public research release.
  - **Access:** DOWNLOADABLE (Mendeley).
  - **Best "same writer, all positional forms" set found.**

  — [PMC8866147](https://pmc.ncbi.nlm.nih.gov/articles/PMC8866147/); [Mendeley](https://data.mendeley.com/datasets/2h76672znt/2)
- **HMBD v1.** Arabic handwritten characters in isolated, beginning, middle and end positions, plus digits. GitHub repository exists. A later study expanded it to 114 positional classes (snippet). — [GitHub HossamBalaha/HMBD-v1](https://github.com/HossamBalaha/HMBD-v1)
- **AHCD.** Arabic Handwritten Characters Dataset on Kaggle (listing seen). — [Kaggle mloey1/ahcd1](https://www.kaggle.com/datasets/mloey1/ahcd1)
- **IFN/ENIT.** 26,459 samples of 946 Tunisian town and village names by 411 writers. — [ResearchGate](https://www.researchgate.net/publication/309760586_Ifnenit-database_of_handwritten_arabic_words_Proceedings_of_CIFED_vol)
- **KHATT.**
  - Offline unconstrained text by 1,000 writers from various countries.
  - A Hugging Face copy exists (`benhachem/KHATT`). Its licence/redistribution status was not checked.

  — [ResearchGate](https://www.researchgate.net/publication/259092955_KHATT_An_open_Arabic_offline_handwritten_text_database); [HF](https://huggingface.co/datasets/benhachem/KHATT)
- **AHTID/MW.** Offline text lines covering all positional forms, 53 writers. — [ResearchGate](https://www.researchgate.net/publication/261156375_A_Database_for_Arabic_Handwritten_Text_Image_Recognition_and_Writer_Identification)
- **Muharaf.** Historical handwritten Arabic manuscripts (arXiv 2406.09630). It notes that many letters have 4 positional shapes. — [arXiv](https://arxiv.org/html/2406.09630)
- **AlexU-Word and IESK-ArDB.** Used (with IFN/ENIT and AHAWP) in Arabic-One-DM's 1,417-writer merged training set. — [Arabic-One-DM](https://github.com/7abushahla/Arabic-One-DM)

### Inferences
- **Same writer, many positional forms:** AHAWP is the only confirmed public set where each of 82 writers produces every positional form (10× each), but it is offline. It could be the few-shot style *test* or fine-tune source after skeletonisation. HMBD and Hijja/ACRC also have positional forms per writer, but their per-writer structure is unverified.
- **Online data with natural joins:** for stroke-level pretraining with natural context (joins, dots, kashida), the line- and sentence-level online sets are the candidates. Online-KHATT (InkML, 623 writers) and AltecOnDB are the main ones. Neither is confirmed obtainable today: the Online-KHATT host is dead and AltecOnDB has no link. ADAB (InkML words) and Calliar (JSON strokes) are the confirmed-downloadable online options.

### Gaps
- **Not verified this session** (from background knowledge only):
  - LMCA (an online letters dataset from Tunisia, REGIM)
  - OnHW (the Fraunhofer OnHW sets are Latin, not Arabic, as far as I know)
  - HACDB (offline, 66 shapes, 50 writers?)
  - MADBase (offline digits, 700 writers)
  - Urdu/Persian *online* datasets (e.g. the Urdu online character sets, Persian "PHOND"?)

  I found no source for these, so they need checking.
- **Licences unknown** for Online-KHATT, AltecOnDB, CHAKEL-DB, OHASD, HMBD, AHCD, IFN/ENIT and KHATT. The access procedures for IFN/ENIT (historically a request/licence form) and KHATT (historically registration at khatt.ideas2serve.net, the same domain family as the dead Online-KHATT host) were not confirmed.
- **Writer count unknown:** Calliar's writer/calligrapher count is not stated in the repo.

## Commercial / consumer products making an Arabic font from handwriting

### Takeaway
I found no evidence that any consumer handwriting-to-font product supports Arabic contextual (positional) forms.

### Cited Findings
- **Calligraphr.** Exports standard TTF/OTF. Pro automatically replaces suitable character sequences with user-specified ligatures. The features page does not mention Arabic or isolated/initial/medial/final forms. — [Calligraphr features](https://www.calligraphr.com/features)
- **What a proper Arabic font needs:** separate initial, medial, final and isolated glyphs, chosen by the OpenType contextual-analysis engine from the neighbouring characters. — [Glyphs Arabic tutorial](https://glyphsapp.com/learn/arabic); [Microsoft OpenType Arabic](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- **Patents name the use:** the KFUPM synthesis patents list personalised fonts as an application. — [US10083362](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10083362)

### Inferences
- A product that collects all 4 forms per letter (about 100+ glyphs plus lam-alef ligatures) and writes a GSUB init/medi/fina/isol table would apparently be novel among consumer tools. A few-shot generator that fills in missing forms would be even more so.

### Gaps
- I did not test Calligraphr's Arabic template or check other tools (MyScriptFont, FontForge-based services, Arabic-specific apps). Whether any of them emit Arabic GSUB features is unknown.

## Reported difficulties specific to Arabic generation

### Takeaway
Across sources, the reported difficulties are:
- context-dependent letter shapes (up to 4 positional forms)
- cursive connection points and extensions between glyphs
- diacritics and dots
- baseline, slant and skew variation
- scarce datasets, especially online ones

### Cited Findings
- **Positional forms:** many Arabic letters have four shapes (isolated, initial, medial, final), which makes the task harder. — [Muharaf arXiv 2406.09630](https://arxiv.org/html/2406.09630)
- **One-shot generation:** Arabic-One-DM names cursive structure, contextual letter forms and diacritics as the adaptation challenges. — [Arabic-One-DM](https://github.com/7abushahla/Arabic-One-DM)
- **Glyph joining:** concatenation systems need connection-point detection and extension modelling (extended glyphs vs. synthetic extensions). Connection-point localisation errors were 1.64% / 2.12% (right / left). — [US10083362 / family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10083362); [Elarian thesis](https://eprints.kfupm.edu.sa/id/eprint/139204/)
- **Page layout:** baseline, slant and skew must be simulated separately for realistic page synthesis. — [Dinges et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4534626/)
- **Data scarcity:** no robust standard comprehensive online/offline Arabic database existed (2013), and Arabic synthesis is harder than Latin. — [Saabni & El-Sana record](https://cris.iucc.ac.il/en/publications/comprehensive-synthetic-arabic-database-for-onoff-line-script-rec-3/); [US10083362](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10083362)
- **Urdu benchmark:** no standard benchmark exists for Urdu diffusion word generation, which makes comparison infeasible. — [DiffusionRec](https://tukl.seecs.nust.edu.pk/members/projects/conference/DiffusionRec-Recognition-Guided-Diffusion-for-Content-Aware-Urdu-Handwriting-Generation.pdf)
- **Calligraphy research gap:** realistic handwritten calligraphy generation is under-researched. — [IJACSA 2025](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)
- **Character imbalance:** rare characters are under-represented in Calliar. — [Calliar GitHub](https://github.com/ARBML/Calliar)

### Inferences
- **Detached dots:** dots are separate strokes, often written after the main body (delayed strokes). A stroke model must therefore either learn delayed-stroke ordering or treat dots as separately placed components. Calliar's primitive-stroke decomposition already separates them.
- **Lam-alef and kashida:** these need explicit handling. Lam-alef is a mandatory ligature. Kashida corresponds to the "extended glyph" or "synthetic extension" in the classical systems.

### Gaps
- No source found that quantifies dot-placement or lam-alef errors in generated Arabic. Vertical stacking/overlap of letters (common in Nastaliq and in handwriting) was not specifically documented in what I read.
