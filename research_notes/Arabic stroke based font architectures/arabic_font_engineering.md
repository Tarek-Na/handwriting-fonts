# Arabic handwriting font engineering: from generated strokes/glyph images to a working OpenType font (2015-2026)

Scope note: research done 2026-11-14. Sources are specs (Microsoft OpenType, Unicode UCD, AFDKO feature-file spec), tool docs/source (skia-pathops, uharfbuzz, Glyphs), and practitioner threads (TypeDrawers, HarfBuzz list). The project's own code (`src/hfont/fontbuild/arabic.py`, `build.py`) was read for the "does it extend" question. Calligraphr's Arabic support could not be verified (see Gaps).

## 1. Stroke/single-line fonts: turning centreline glyphs into fillable outlines

### Takeaway
OpenType outline fonts (glyf/CFF) only render filled closed contours, so a centreline (open-path) glyph must be expanded into a closed outline before it goes in the font; the practical Python route is `skia-pathops` (`Path.stroke(...)` then `simplify`) feeding a fontTools pen. The only OpenType mechanism that keeps true open strokes is the SVG table (as Relief SingleLine does), which is not supported in most text renderers and is not suitable for a general-use Arabic text font; COLR does not support open contours.

### Cited Findings
- "True single line OpenType fonts don't exist, as OpenType fonts are based on the principle of closed contours"; FontCreator expands stroked open paths to outlines on export, and a stroked open path gets two caps — [FontCreator manual: single-line fonts](https://www.high-logic.com/fontcreator/manual15/fc_single-line-fonts.html); [FontCreator: designing with strokes](https://www.high-logic.com/font-editor/fontcreator/tutorials/designing-with-strokes)
- FontCreator notes OpenType-COLR "does not support open contours" — [FontCreator: single-line fonts tutorial](https://www.high-logic.com/font-editor/fontcreator/tutorials/how-to-make-single-line-fonts-for-engraving-and-cutting)
- Relief SingleLine Regular is an OpenType-SVG font whose glyphs stay open paths, working in Adobe apps (CC 2019+); a companion "Relief SingleLine Outline" is a classic closed-contour OTF — [befonts: Relief SingleLine](https://befonts.com/relief-singleline-font.html); [TypeDrawers thread](https://typedrawers.com/discussion/5026/exporting-a-single-line-open-contour-svg-font-from-fontlab)
- The OpenType SVG table holds SVG descriptions for some or all glyphs and defines `context-stroke` so a document's stroke can be inherited — [Microsoft OpenType spec: SVG table](https://learn.microsoft.com/typography/opentype/spec/svg)
- Users reported artefacts in Illustrator with single-line SVG glyphs and plotters outlining letters twice — [TypeDrawers](https://typedrawers.com/discussion/comment/65722)
- `vpype-singlestroke` (Python) converts single-line open-contour SVG fonts to closed contours for TTF generation by offsetting parallel paths on each side and joining with end caps — [PyPI vpype-singlestroke](https://pypi.org/project/vpype-singlestroke/)
- `skia-pathops` (fontTools org) exposes `Path.stroke(width, cap: LineCap, join: LineJoin, miter_limit, dash_array=None, dash_offset=0.0)`; `Path.simplify(fix_winding, keep_starting_points, clockwise)` (also module-level `simplify`); boolean `op(one, two, operator)` / `OpBuilder` (union via `kUnion_SkPathOp`); `Path.getPen(allow_open_paths=True)` returns a fontTools-compatible `PathPen`, and `Path.draw(pen)` replays into any fontTools pen — [skia-pathops source `_pathops.pyx`](https://raw.githubusercontent.com/fonttools/skia-pathops/main/src/python/pathops/_pathops.pyx); README: boolean ops intersection/union/difference/xor — [skia-pathops](https://github.com/fonttools/skia-pathops)
- I found no stroke-expansion pen in fontTools itself documented in search results; fontTools is mentioned in forums only for glyph copying/testing — [DrawBot forum](https://forum.drawbot.com/post/1083) (absence of evidence, see Gaps)

### Inferences
- Recommended pipeline for a stroke generator: centreline polyline/Bezier (font units) -> `pathops.Path` via `getPen(allow_open_paths=True)` -> `path.stroke(w, ROUND cap, ROUND join, 4)` -> `path.simplify(fix_winding=True, clockwise=False)` (CFF wants counter-clockwise outer contours; TrueType wants clockwise — choose per output format) -> `path.draw(T2CharStringPen | TTGlyphPen)` -> `FontBuilder`. Round caps/joins best imitate a ballpoint/felt pen.
- Variable stroke width (pressure) is not available from `Path.stroke` (single width). Options: (a) stamp circles of radius r(t) along the path and union them (simple, robust, many points), or (b) compute left/right offset curves with per-point normals scaled by w(t)/2 and close them with caps, then `simplify` to remove self-overlap.
- A weight axis "built from centrelines" (variable font) needs point-compatible masters; skia's stroker and `simplify` do not guarantee identical point structure across widths, so a variable font requires a custom offsetter with fixed topology (option b without `simplify`, keeping overlaps — overlapping contours are allowed in glyf variable fonts with the OVERLAP_SIMPLE flag). This is inference, not verified in a source here.
- Arabic is especially well suited to the stroke approach because joins are baseline strokes: if every connecting stroke is expanded with the same width, the join thickness matches automatically (cf. Glyphs advice that the joining stroke height equals the stroke width, section 5).

### Gaps
- No authoritative doc found on a fontTools-native stroker or on building variable fonts from centrelines; verify OVERLAP_SIMPLE rendering in target apps.
- `LineCap`/`LineJoin` enum member names in skia-pathops were not seen in the fetched excerpt (check `pathops.LineCap.ROUND_CAP` etc. in an interpreter).

## 2. Arabic OpenType shaping: features, lookups, joining classes, minimum glyph set

### Takeaway
The shaper (HarfBuzz/Uniscribe/DirectWrite/CoreText) computes each letter's joining position from Unicode joining data; the font only supplies GSUB single substitutions under `isol/fina/medi/init`, a required lam-alef ligature under `rlig`, and optionally `calt`, `liga`, `curs`, `kern`, `mark`, `mkmk`. Basic Arabic (36 letters U+0621-063A, U+0641-064A) has 24 dual-joining, 11 right-joining, 1 non-joining letter, giving about 119 positional letter glyphs + 8 lam-alef glyphs as a working minimum, before digits/punctuation.

### Cited Findings
- Shaping stages (Uniscribe): (1) analyse characters for contextual shape, (2) GSUB, (3) GPOS. Standard order: `ccmp, isol, fina, medi, init, rlig, rclt, calt` (language forms, always applied), `liga` (on by default), `dlig`, `cswh` (off by default), `mset`; then GPOS `curs, kern, mark, mkmk` (always applied) — [Microsoft: Developing OpenType Fonts for Arabic Script](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- The same order with `stch` (Syriac stretching) before language forms and `locl` first among them is documented as the general Arabic shaping model (HarfBuzz-aligned) — [n8willis opentype-shaping-documents: Arabic](https://github.com/n8willis/opentype-shaping-documents/blob/master/opentype-shaping-arabic-general.md)
- `isol/fina/medi/init` are GSUB LookupType 1; `rlig` is LookupType 4 for ligatures that must always form, e.g. lam-alef; "Ligatures with more components must be stored ahead of those with fewer"; for Win9x back-compat also put rlig items in `liga` — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- `rclt` is for substitutions required for script correctness; `calt` (LookupType 6) is always applied for Arabic; put `rclt` items in `calt` too for Windows 7/8.1 — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- `curs` = GPOS LookupType 3, "positions cursive characters so that the exit point of the current character matches with entry point of the following character"; `mark` = LookupType 4 (or 5 for ligatures), `mkmk` = LookupType 6 — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- Recommended extra code points: dotted circle U+25CC (fallback base for invalid marks), ZWNJ U+200C, ZWJ U+200D, LRM U+200E, RLM U+200F; ZWNJ prevents a cursive connection — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- Include `dflt` language system; script tag `arab`, language tags e.g. `ARA `, `FAR `, `URD ` — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- Joining types: D dual, R right (joins to preceding only), L, U non-joining, T transparent (marks), C join-causing (tatweel U+0640 and ZWJ) — [n8willis shaping docs](https://github.com/n8willis/opentype-shaping-documents/blob/master/opentype-shaping-arabic-general.md); exact values from [Unicode ArabicShaping.txt](https://www.unicode.org/Public/UCD/latest/ucd/ArabicShaping.txt):
  - R (right-joining, so only isolated + final forms): U+0622 آ, 0623 أ, 0624 ؤ, 0625 إ, 0627 ا, 0629 ة, 062F د, 0630 ذ, 0631 ر, 0632 ز, 0648 و (also 0671 ٱ, 0698 ژ)
  - U: U+0621 ء hamza only
  - C: U+0640 tatweel
  - D: everything else in U+0626, 0628, 062A-062E, 0633-063A, 0641-0647, 0649, 064A (ئ ب ت ث ج ح خ س ش ص ض ط ظ ع غ ف ق ك ل م ن ه ى ي); Persian 067E پ, 0686 چ, 06A9 ک, 06AF گ, 06CC ی are D
- Google Fonts' `GF_Arabic_Core` glyph set has ~383 names: 55 Arabic base letters, 127 positional forms, 5 lam-alef ligature glyphs (e.g. `lam_alef-ar`, `lam_alefMadda-ar`, `lam_alefWasla-ar`), 32 digits (Western, Arabic-Indic, Farsi/Urdu), 2 number separators, 42 marks (fatha, shadda, hamzaabove, dot marks), 68 punctuation/symbols incl. `dottedCircle`, plus 52 Latin letters (counts approximate, from the list) — [googlefonts/glyphsets GF_Arabic_Core.txt](https://raw.githubusercontent.com/googlefonts/glyphsets/main/data/results/txt/nice-names/GF_Arabic_Core.txt)
- HarfBuzz has a fallback Arabic shaper that synthesises lookups from Unicode presentation forms when the font lacks GSUB joining features (Arabic script only) — [HarfBuzz hb-ot-shaper-arabic.cc](https://flutter.googlesource.com/third_party/harfbuzz/+/HEAD/src/hb-ot-shaper-arabic.cc); [fallback header](https://arai.searchfox.org/firefox-main/source/gfx/harfbuzz/src/hb-ot-shaper-arabic-fallback.hh)
- Cursive-attachment `RightToLeft` lookup flag only affects curs: when set, the last glyph of the chain sits on the baseline; HarfBuzz swaps parent/child when the flag is absent; a 2015 HarfBuzz fix handled mixes of RTL and non-RTL curs lookups (Khaled Hosny report, Behdad fix) — [HarfBuzz list 2015](https://lists.freedesktop.org/archives/harfbuzz/2015-August/005021.html); [Glyphs forum: RightToLeft flag](https://forum.glyphsapp.com/t/righttoleft-lookup-flag/15678); [HarfBuzz CursivePosFormat1](https://flutter.googlesource.com/third_party/harfbuzz/+/HEAD/src/OT/Layout/GPOS/CursivePosFormat1.hh)
- July 2020 test (Simon Cozens): with `lookupflag IgnoreMarks RightToLeft` in curs, HarfBuzz and Uniscribe (via WINE) left an unattached nukta mark alone but CoreText moved it; Apple fixed it in macOS 10.15 / iOS 13 — [TypeDrawers: Cursive positioning in Uniscribe, CoreText, HarfBuzz](https://typedrawers.com/discussion/comment/48159)

Minimum viable glyph arithmetic (derived from ArabicShaping.txt above, Basic Arabic letters only):

| Class | Letters | Forms each | Glyphs |
|---|---|---|---|
| D (dual) | 24 (incl. ئ ى ي) | isol, init, medi, fina | 96 |
| R (right) | 11 (ا آ أ إ ؤ ة د ذ ر ز و) | isol, fina | 22 |
| U | 1 (ء) | isol | 1 |
| lam-alef ligatures | 4 (لا لآ لأ لإ) | isol, fina | 8 |
| tatweel | 1 | — | 1 |
| Arabic-Indic digits U+0660-0669 | 10 | — | 10 |
| punctuation ، ؛ ؟ ٪ (U+060C, 061B, 061F, 066A) + space, ZWNJ, ZWJ, LRM, RLM, U+25CC | ~10 | — | ~10 |
| **Total** | | | **~148** |

Example feature file (compiles with `fontTools.feaLib`; names follow the project's `base.init` scheme, shown with Glyphs-style names here):

```fea
languagesystem DFLT dflt;
languagesystem arab dflt;

@MARKS = [fatha-ar damma-ar kasra-ar shadda-ar sukun-ar hamzaabove-ar dotabove-ar dotbelow-ar];
table GDEF { GlyphClassDef , , @MARKS, ; } GDEF;   # bases, ligs, marks, components

feature isol { sub beh-ar by beh-ar; } isol;        # usually identity; can be omitted
feature fina { lookupflag IgnoreMarks; sub beh-ar by beh-ar.fina; sub alef-ar by alef-ar.fina; } fina;
feature medi { lookupflag IgnoreMarks; sub beh-ar by beh-ar.medi; } medi;
feature init { lookupflag IgnoreMarks; sub beh-ar by beh-ar.init; } init;

feature rlig {
  lookupflag IgnoreMarks;
  sub lam-ar.init alef-ar.fina by lam_alef-ar;          # word-initial / isolated ligature
  sub lam-ar.medi alef-ar.fina by lam_alef-ar.fina;     # joined from the right
  sub lam-ar.init alefHamzaabove-ar.fina by lam_alefHamzaabove-ar;
  # ... madda, hamza-below likewise
} rlig;

feature curs {                                    # GPOS type 3: entry anchor, exit anchor
  lookupflag RightToLeft IgnoreMarks;
  pos cursive beh-ar.init <anchor NULL>      <anchor 0 100>;
  pos cursive beh-ar.medi <anchor 520 100>   <anchor 0 100>;
  pos cursive beh-ar.fina <anchor 520 100>   <anchor NULL>;
  pos cursive alef-ar.fina <anchor 180 100>  <anchor NULL>;
} curs;

markClass [fatha-ar dotabove-ar]  <anchor 0 650> @TOP;
markClass [kasra-ar dotbelow-ar]  <anchor 0 -150> @BOTTOM;
feature mark {
  pos base [beh-ar beh-ar.init beh-ar.medi beh-ar.fina] <anchor 260 600> mark @TOP <anchor 260 -120> mark @BOTTOM;
} mark;
```
Syntax for `pos cursive <glyph> <entry> <exit>` with `<anchor NULL>`, `markClass`, `pos base`, `lookupflag`, ligature `sub a b by c;` and chaining `sub x a' by a.alt;` is from the [AFDKO OpenType Feature File Specification](https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html) (§4.d, §4.f, §5.d, §5.f, §6.c, §6.d).

### Inferences
- Because lam is D and every alef variant is R, a lam followed by alef is always shaped as (lam.init|lam.medi) + alef.fina, so the two rlig rules per alef above cover every case.
- If joins are drawn to a fixed baseline height and sidebearings are zero at the join (exit at x=0, entry at x=advance), `curs` is not strictly required in a naskh-like handwriting font: glyphs abut correctly with plain advances. `curs` becomes necessary when generated glyphs' join points vary in x/y, which is the likely case for a generative model (section 5).
- Dots as separate marks (dotless "rasm" skeletons + dot marks via `mark`, or composites) reduce the number of shapes the generator must produce: the 28 letters collapse onto roughly 17-19 skeleton groups (cf. Glyphs' `teh-ar = behDotless-ar + two dots`). For a handwriting generator, baking dots into each glyph is simpler and avoids GPOS mark anchors; separate marks give consistent dots but look more mechanical. Trade-off, not sourced.
- Ship `rlig` contents in `liga` as well only if old Windows matters; not needed for HarfBuzz/CoreText/DirectWrite.

### Gaps
- Exact current behaviour differences between HarfBuzz, DirectWrite and CoreText for `curs` in 2024-2026 not found; the 2020 CoreText bug was reported fixed.
- No authoritative "minimum Arabic glyph set" specification exists; the table above is a derivation; GF_Arabic_Core is the nearest published standard.

## 3. Handwriting fonts: contextual alternates, randomness, and Calligraphr

### Takeaway
The robust, renderer-independent way to make a handwriting font less mechanical is deterministic `calt` cycling (chained contextual substitutions picking `.alt1/.alt2` based on preceding glyphs); `rand` exists and HarfBuzz implements it (since 2018, AlternateSubstFormat1 only), but application support is thin and repeats happen. I could not confirm whether Calligraphr supports Arabic shaping.

### Cited Findings
- HarfBuzz only uses `rand` if the font has a `rand` feature with an AlternateSubst (format 1) subtable; a 2018 commit made randomisation the default value for `rand`; most fonts instead do pseudo-randomness via contextual substitutions — [HarfBuzz list 2020: Alternate/random glyphs](https://lists.freedesktop.org/archives/harfbuzz/2020-February/007475.html); [HarfBuzz list 2018: Changes to 'rand'](https://lists.freedesktop.org/archives/harfbuzz/2018-September/006752.html)
- Users note that with few alternates randomisation can place the same alternate side by side — [HarfBuzz list 2020](https://lists.freedesktop.org/archives/harfbuzz/2020-February/007481.html)
- A TypeDrawers thread reported no mainstream layout engine supporting `rand` at the time, with later replies naming PagePlus and FontCreator — [TypeDrawers: Any app that supports rand?](https://typedrawers.com/discussion/comment/65512)
- `calt` is always applied for Arabic (cannot be switched off by Uniscribe) and is the place for "alternative forms of connections ... to give a better calligraphic presentation"; `cswh` for swashes (off by default) — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- No search result documented Calligraphr's Arabic/RTL/positional-form support — searches: [results for "Calligraphr Arabic"](https://glyphsapp.com/learn/arabic) returned only general Arabic-font tutorials; the help page URL tried returned 404.

### Inferences
- Arabic `calt` alternation must run *after* `init/medi/fina` (it does, by feature order), so alternates are per positional form: `beh-ar.medi.alt1`, etc. Each alternate must keep the same entry/exit join point (or carry its own `curs` anchors), otherwise alternation breaks joins.
- A simple cycling scheme in .fea: `sub @MEDI' lookup TO_ALT1;` with backtrack `@MEDI_DEFAULT`, i.e. "if the previous letter used the default form, use alt1" — produces default/alt1/default... alternation without randomness and works in every engine.

### Gaps
- Calligraphr Arabic support: unverified. The owner should check the product's own templates (it may export isolated forms only, which would give unconnected Arabic — the failure mode seen in [DrawBot forum](https://forum.drawbot.com/post/100)).

## 4. Python tooling: build, compile features, test with HarfBuzz

### Takeaway
Everything needed is in fontTools + uharfbuzz: `fontTools.fontBuilder.FontBuilder` for tables, `fontTools.feaLib.builder.addOpenTypeFeaturesFromString` for GSUB/GPOS/GDEF from .fea text, `skia-pathops` for stroke expansion/overlap removal, and `uharfbuzz` to shape test strings and assert glyph names/positions. ufo2ft/fontmake are optional (useful if glyphs are written as UFO with anchors, from which ufo2ft generates `mark`/`curs` automatically).

### Cited Findings
- uharfbuzz shaping API: `hb.Blob.from_file_path(path)`, `hb.Face(blob)`, `hb.Font(face)`, `buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()`, `hb.shape(font, buf, {"kern": True, "liga": True})`, then `buf.glyph_infos` (glyph id, cluster) and `buf.glyph_positions` (advances, offsets) — [uharfbuzz README](https://github.com/harfbuzz/uharfbuzz)
- Glyphs naming convention: `name-ar`, positional suffixes `.init/.medi/.fina`, unsuffixed = isolated; ligatures `lam_alef-ar`, positional ligatures named after their components' forms; Glyphs can auto-generate positional and ligature feature code — [Glyphs: Creating an Arabic font](https://glyphsapp.com/learn/arabic)
- The project already compiles Arabic GSUB with `addOpenTypeFeaturesFromString` (`init/medi/fina` with `lookupflag IgnoreMarks`, `rlig` lam-alef matching `lam.init|lam.medi + alef.fina`) — local: `c:\Users\User\Desktop\DeepLearningProject\src\hfont\fontbuild\arabic.py`; FontBuilder CFF build in `src\hfont\fontbuild\build.py`

### Inferences
- Test recipe (no GUI): shape `"بيت"`, `"لا"`, `"سلام"`, `"مع ‌ ZWNJ"` with uharfbuzz; map gids back with `ttfont.getGlyphOrder()`; assert names equal expected (`teh.fina`, `yeh.medi`, `beh.init` in visual order — note HarfBuzz returns RTL runs in visual order, i.e. reversed); assert that for each adjacent joined pair the exit point of one glyph plus its position equals the entry point of the next (checks the join geometry after `curs`). Render with `hb-view` or by drawing outlines into a raster for visual review.
- Add `fontbakery` (GF profile) or `ots-sanitize` as a structural check; add `glyphsets` to diff coverage against GF_Arabic_Core.

### Gaps
- ufo2ft's exact anchor-naming rules for automatic `curs` generation (`entry`/`exit` anchors) not verified in this session.

## 5. The joining problem: consistent baseline join points and what it demands of a generator

### Takeaway
Type designers fix the join to a single height and thickness (Glyphs' default joining stroke at y≈100 with stroke-width thickness, exit at x=0, entry at the right sidebearing) and add `curs` entry/exit anchors so the shaper snaps any residual mismatch. For a generator, this means either constraining every connecting stroke to end at a canonical (x, y, thickness, tangent) or emitting per-glyph entry/exit anchors from the generated strokes and letting `curs` align them; the tangent/thickness mismatch cannot be fixed by `curs` and must be handled in generation.

### Cited Findings
- Glyphs: joining line at default height 100 with overshoot 16, value matching the stroke width; connecting stroke ends must meet at the same height; use segment components (`_segment.strokebutt`) with `start`/`end` anchors on the baseline, butt ~10 units deep, "best to only add segment components to the entering strokes"; cursive `exit` anchor at x=0,y=0 and `entry` on the right sidebearing, aligned with sidebearings for old renderers — [Glyphs: Creating an Arabic font](https://glyphsapp.com/learn/arabic)
- `curs` matches the exit point of the current glyph with the entry point of the next; in Nastaliq it raises glyphs vertically so they connect — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- With `RightToLeft` set, the last glyph of a cursive chain sits on the baseline and earlier glyphs are shifted by the accumulated y offsets — [Glyphs forum: RightToLeft lookup flag](https://forum.glyphsapp.com/t/righttoleft-lookup-flag/15678); [HarfBuzz CursivePosFormat1](https://flutter.googlesource.com/third_party/harfbuzz/+/HEAD/src/OT/Layout/GPOS/CursivePosFormat1.hh)
- Project status: "cursive joins already fail in Latin (README, known issue 4). The Arabic export path itself is done and tested (gate 4)" — local `instruction.md` line 390

### Inferences
- Without `curs`, a join mismatch of dy between a glyph's exit and the next glyph's entry is a visible step; with `curs` the step disappears but y-errors accumulate along a word (Nastaliq-style drift), so a handwriting generator should still keep joins near a fixed height; clamp or retarget the last ~5-10% of each connecting stroke to the canonical join (x=0 or advance, y=h_join) with a fixed tangent (horizontal) before stroke expansion.
- With stroke-expanded centrelines, a uniform width at the endpoint plus a butt cap at the join and an overlap of a few units (Glyphs' "butt ~10 units deep" idea) hides seams; round caps at joins produce visible bulges where two caps overlap with anti-aliasing.
- Constraint on the generator: generate per positional form, conditioned on form (init/medi/fina/isol), with entry/exit as explicit boundary conditions; R-joining letters' final forms need an entry only; init forms an exit only.
- Kashida/tatweel: include a `tatweel` glyph that is a pure join segment (entry and exit at the canonical height, same thickness); this also serves as the join-test glyph.

### Gaps
- No published work found (2015-2026) specifically on ML-generated Arabic handwriting fonts with enforced join constraints; that would need a separate literature search.

## 6. Does the project's Latin pipeline (75 glyph images -> contour tracing -> OTF) extend to Arabic?

### Takeaway
Partially: the build/trace/FontBuilder/feaLib pieces carry over, and a tested GSUB module for positional forms and lam-alef already exists. What is new is (a) ~120-150 glyphs instead of 75, structured as positional forms, (b) join geometry (consistent baseline joins, ideally `curs` anchors), (c) RTL testing with a shaper, and (d) optionally marks/dots via GPOS. Contour tracing of glyph images makes (b) hard because join points come out of a raster; a centreline/stroke representation makes joins controllable.

### Cited Findings
- Existing `arabic.py` writes `languagesystem DFLT dflt; languagesystem arab dflt;`, `fina/medi/init` single substitutions discovered from `base.init/.medi/.fina` names with `lookupflag IgnoreMarks`, and `rlig` lam-alef (alef, alef-madda, alef-hamza-above, alef-hamza-below) for `lam.init` and `lam.medi`; missing forms fall back to isolated glyphs; only isolated forms in cmap — local `c:\Users\User\Desktop\DeepLearningProject\src\hfont\fontbuild\arabic.py`
- Shaper decides joining; the font supplies isolated forms in cmap and other forms through GSUB — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- Recommended extras (dotted circle, ZWJ/ZWNJ/LRM/RLM) — [Microsoft Arabic guide](https://learn.microsoft.com/en-us/typography/script-development/arabic)

### Inferences
- Not yet present in `arabic.py` (from reading it): no `curs`, `mark/mkmk`, GDEF classes, `calt` alternates, tatweel, dotted circle, ZWJ/ZWNJ/LRM/RLM glyphs, Arabic-Indic digits or Arabic punctuation. Of these, ZWNJ/ZWJ/LRM/RLM (zero-width, empty), U+25CC, tatweel, digits and ، ؛ ؟ are cheap and should be added first; `curs` is the main new engineering item.
- Contour-traced images: join points must be measured from the traced outline (where the stroke crosses the glyph's left/right edge near the baseline) and written as `curs` anchors; this is fragile with ink blobs. Stroke-based glyphs give exact endpoints, so anchors are known by construction — the main argument for a centreline generator in Arabic.
- Data collection: a template needs ~120 cells (positional forms) or the writer writes words and forms are segmented out; the latter matches natural handwriting but needs a segmenter. Hamza forms (أ إ آ ؤ ئ ء) and ة/ى must be on the sheet; digits ٠-٩ too.
- Sanity rule for tests: every Arabic sample word, shaped with uharfbuzz, should contain no glyph without a positional suffix where a joined form was expected, and join-point continuity should hold within ±N units.

### Gaps
- Whether the project's charset table (`src/hfont/charset.py`) already lists all 36 letters' joining classes was not checked in detail.
- Whether Latin "known issue 4" (failed cursive joins) has a root cause that also affects Arabic was not investigated.
