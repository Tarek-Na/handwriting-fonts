"""Arabic contextual shaping for exported fonts.

Latin needs no shaping tables, which is why Phase 1 cannot surface any of the
problems here, and why the brief asks for this to be validated separately and
early. An Arabic font works only if text engines can swap each letter for the
right positional form as it is typed, and that is entirely a matter of the GSUB
table this module writes.

The division of labour is worth being precise about, because it determines
what a font has to contain:

* the **shaper** (HarfBuzz, CoreText, DirectWrite) decides, from Unicode's
  joining data, whether each letter is isolated, initial, medial or final in
  its context. The font does not encode joining behaviour.
* the **font** supplies, under the ``isol``/``init``/``medi``/``fina`` features,
  a substitution from each base glyph to its positional form, and under
  ``rlig`` the mandatory ligatures — above all lam-alef, which Arabic requires
  and which is not optional typography.

So a font needs: the isolated form of each letter in its ``cmap``, every other
form reachable only through GSUB, and a consistent naming scheme linking them.
The names are exactly the ``GlyphSpec`` names from charset.py.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..charset import Form

_ALT = re.compile(r"^(.+)\.alt\d+$")

POSITIONAL_FEATURES = {
    Form.INIT: "init",
    Form.MEDI: "medi",
    Form.FINA: "fina",
}

#: Lam-alef ligatures: alef variant -> ligature base name. Arabic requires these
#: whenever lam is followed by alef; drawing the two letters separately is a
#: spelling error to a reader, not a stylistic choice.
LAM = "ل"
ALEF_VARIANTS = {
    "ا": "lam_alef",          # alef
    "آ": "lam_alefMadda",     # alef with madda above
    "أ": "lam_alefHamzaAbove",
    "إ": "lam_alefHamzaBelow",
}


def _split(name: str) -> tuple[str, str | None]:
    base, _, suffix = name.partition(".")
    return base, suffix or None


def arabic_feature_code(glyph_names: list[str], lam: str, alefs: dict[str, str]) -> str:
    """OpenType feature code for positional forms and lam-alef ligatures.

    ``glyph_names`` is every glyph in the font. Forms are discovered from the
    naming convention (``base``, ``base.init``, ``base.medi``, ``base.fina``),
    so a letter the model did not produce a form for simply gets no rule — the
    shaper then falls back to the isolated glyph, which is legible, rather than
    to a missing glyph, which is not.

    ``lam`` is the glyph name of isolated lam; ``alefs`` maps each alef glyph
    name to the base name of its ligature with lam.
    """
    names = set(glyph_names)
    by_feature: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name in sorted(names):
        base, suffix = _split(name)
        if suffix in ("init", "medi", "fina") and base in names:
            by_feature[suffix].append((base, name))

    lines = ["languagesystem DFLT dflt;", "languagesystem arab dflt;", ""]
    # Order in the file does not decide application order — the shaper applies
    # the joining features first and rlig after, by specification — but keeping
    # them in that order makes the file read the way it will run.
    for feature in ("fina", "medi", "init"):
        rules = by_feature.get(feature)
        if not rules:
            continue
        lines.append(f"feature {feature} {{")
        lines.append("    lookupflag IgnoreMarks;")
        lines.extend(f"    sub {src} by {dst};" for src, dst in rules)
        lines.append(f"}} {feature};")
        lines.append("")

    # rlig runs after the positional forms have been chosen, so it matches the
    # *already substituted* glyphs: lam in initial or medial position followed
    # by alef in final position. The initial case yields the free-standing
    # ligature; the medial case the ligature's final form, still joined to the
    # letter before it.
    rlig = []
    for alef, lig in sorted(alefs.items()):
        alef_fina = f"{alef}.fina"
        if alef_fina not in names:
            continue
        if f"{lam}.init" in names and lig in names:
            rlig.append(f"    sub {lam}.init {alef_fina} by {lig};")
        if f"{lam}.medi" in names and f"{lig}.fina" in names:
            rlig.append(f"    sub {lam}.medi {alef_fina} by {lig}.fina;")
    if rlig:
        lines.append("feature rlig {")
        lines.append("    lookupflag IgnoreMarks;")
        lines.extend(rlig)
        lines.append("} rlig;")
        lines.append("")

    lines.extend(_cycling_code(names))
    return "\n".join(lines) + "\n"


def _variant(name: str) -> int:
    m = _ALT.match(name)
    return int(name.rsplit("alt", 1)[1]) if m else 0


def _cycling_code(names: set[str]) -> list[str]:
    """``calt`` rules that cycle a letter's versions (``name``, ``name.alt1``, ...).

    A glyph with alternates takes the version after the one the letter
    before it shows (any letter; one without alternates counts as version
    0), so a word with three behs does not show three identical stamps.
    ``calt`` runs after the positional forms are chosen and is on by default
    in every major shaper, unlike ``rand``, which HarfBuzz did not apply.
    """
    versions: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        m = _ALT.match(name)
        if m and m.group(1) in names:
            versions[m.group(1)].append(name)
    if not versions:
        return []
    count = 1 + max(len(v) for v in versions.values())
    out, rules = [], []
    for k in range(count):
        nxt = (k + 1) % count
        if nxt == 0:
            continue                                   # back to the glyph itself: nothing to do
        pairs = [(base, f"{base}.alt{nxt}") for base in sorted(versions) if f"{base}.alt{nxt}" in names]
        after = sorted(n for n in names if _variant(n) == k and n not in (".notdef", "space"))
        if not pairs or not after:
            continue
        out.append(f"lookup cycle{nxt} {{")
        out.extend(f"    sub {src} by {dst};" for src, dst in pairs)
        out.append(f"}} cycle{nxt};")
        out.append(f"@after{k} = [{' '.join(after)}];")
        out.append(f"@takes{nxt} = [{' '.join(src for src, _ in pairs)}];")
        rules.append(f"    sub @after{k} @takes{nxt}' lookup cycle{nxt};")
    if not rules:
        return []
    return ["", *out, "", "feature calt {", "    lookupflag IgnoreMarks;", *rules, "} calt;"]


def add_arabic_shaping(ttfont, lam: str, alefs: dict[str, str]) -> str:
    """Compile Arabic GSUB into a built font. Returns the feature code used."""
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

    code = arabic_feature_code(ttfont.getGlyphOrder(), lam, alefs)
    addOpenTypeFeaturesFromString(ttfont, code)
    return code
