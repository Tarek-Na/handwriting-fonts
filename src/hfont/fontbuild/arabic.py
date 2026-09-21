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

from collections import defaultdict

from ..charset import Form

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
    return "\n".join(lines) + "\n"


def add_arabic_shaping(ttfont, lam: str, alefs: dict[str, str]) -> str:
    """Compile Arabic GSUB into a built font. Returns the feature code used."""
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

    code = arabic_feature_code(ttfont.getGlyphOrder(), lam, alefs)
    addOpenTypeFeaturesFromString(ttfont, code)
    return code
