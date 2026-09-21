"""Glyph inventory definitions.

A *glyph* here is the pair (character, contextual form). Latin only ever uses
``Form.ISOL``, but the type is carried through the whole pipeline from day one so
that Phase 2 (Arabic) adds entries to a table rather than changing signatures.

Naming follows the Adobe Glyph List where one exists, because that is what
``fontTools`` and every downstream text engine expect. Arabic contextual forms
use the conventional ``.init`` / ``.medi`` / ``.fina`` suffixes.
"""

from __future__ import annotations

import enum
import functools
import unicodedata
from dataclasses import dataclass


@functools.total_ordering
class Form(enum.Enum):
    """Contextual form of a glyph.

    Explicitly ordered. ``GlyphSpec`` is an orderable dataclass, so sorting a
    charset compares ``(char, form)`` tuples and reaches this type whenever two
    glyphs share a character. That never happens in Latin — one form per
    character — so an unordered enum here would go unnoticed until Arabic,
    where every letter has up to four forms and every ``sorted()`` in the
    pipeline would start raising.
    """

    ISOL = "isol"
    INIT = "init"
    MEDI = "medi"
    FINA = "fina"

    @property
    def suffix(self) -> str:
        """Glyph-name suffix used in the exported font."""
        return "" if self is Form.ISOL else f".{self.value}"

    @property
    def sort_index(self) -> int:
        return _FORM_ORDER[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Form):
            return NotImplemented
        return self.sort_index < other.sort_index


#: Presentation order: isolated first, then the joining forms left-to-right.
_FORM_ORDER = {"isol": 0, "init": 1, "medi": 2, "fina": 3}


# Adobe Glyph List names for the ASCII punctuation we care about. Anything not
# in here falls back to a ``uniXXXX`` name, which is valid but less readable.
_AGL: dict[str, str] = {
    " ": "space", "!": "exclam", '"': "quotedbl", "#": "numbersign",
    "$": "dollar", "%": "percent", "&": "ampersand", "'": "quotesingle",
    "(": "parenleft", ")": "parenright", "*": "asterisk", "+": "plus",
    ",": "comma", "-": "hyphen", ".": "period", "/": "slash",
    ":": "colon", ";": "semicolon", "<": "less", "=": "equal",
    ">": "greater", "?": "question", "@": "at", "[": "bracketleft",
    "\\": "backslash", "]": "bracketright", "^": "asciicircum",
    "_": "underscore", "`": "grave", "{": "braceleft", "|": "bar",
    "}": "braceright", "~": "asciitilde",
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "‘": "quoteleft", "’": "quoteright",
    "“": "quotedblleft", "”": "quotedblright",
    "–": "endash", "—": "emdash",
}


def agl_name(char: str) -> str:
    """Return the conventional glyph name for a single character."""
    if char in _AGL:
        return _AGL[char]
    if char.isascii() and char.isalpha():
        return char
    return f"uni{ord(char):04X}"


@dataclass(frozen=True, order=True)
class GlyphSpec:
    """One entry in the glyph inventory the model must be able to produce."""

    char: str
    form: Form = Form.ISOL

    @property
    def name(self) -> str:
        """Glyph name as it will appear in the exported font."""
        return agl_name(self.char) + self.form.suffix

    @property
    def codepoint(self) -> int:
        return ord(self.char)

    @property
    def key(self) -> str:
        """Stable identifier used as a dict key and in filenames."""
        return f"{self.codepoint:04X}.{self.form.value}"

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"

# Punctuation split by how reliably it is present in a free font. The core set is
# near-universal; the extended set is skipped per font when the glyph is missing.
PUNCT_CORE = ".,;:!?'\"()-/&"
PUNCT_EXT = "[]{}@#%*+=<>_~$\\|^`"


@dataclass(frozen=True)
class Charset:
    """An ordered, de-duplicated glyph inventory with a fixed index mapping.

    The index order is part of the trained model (it is the content-embedding
    row order), so ``glyphs`` must never be reordered for an existing checkpoint.
    """

    name: str
    glyphs: tuple[GlyphSpec, ...]

    def __post_init__(self) -> None:
        if len(set(self.glyphs)) != len(self.glyphs):
            raise ValueError(f"charset {self.name!r} contains duplicate glyphs")

    def __len__(self) -> int:
        return len(self.glyphs)

    def __iter__(self):
        return iter(self.glyphs)

    def index(self, glyph: GlyphSpec) -> int:
        return self._index[glyph]

    @property
    def _index(self) -> dict[GlyphSpec, int]:
        # Built lazily and cached on the instance; frozen dataclass needs the
        # object.__setattr__ dance.
        cached = self.__dict__.get("_index_cache")
        if cached is None:
            cached = {g: i for i, g in enumerate(self.glyphs)}
            object.__setattr__(self, "_index_cache", cached)
        return cached

    @property
    def chars(self) -> str:
        return "".join(dict.fromkeys(g.char for g in self.glyphs))

    def subset(self, name: str, chars: str) -> "Charset":
        """A sub-charset containing only the listed characters (isolated form)."""
        wanted = set(chars)
        return Charset(name, tuple(g for g in self.glyphs if g.char in wanted))

    def describe(self) -> str:  # pragma: no cover - reporting aid
        return f"{self.name}: {len(self)} glyphs"


def _latin(name: str, text: str) -> Charset:
    return Charset(name, tuple(GlyphSpec(c) for c in dict.fromkeys(text)))


#: Letters only. Useful for fast ablations where punctuation is noise.
LATIN_LETTERS = _latin("latin_letters", UPPER + LOWER)

#: The Phase 1 target inventory: what an exported font must contain.
LATIN_FULL = _latin("latin_full", UPPER + LOWER + DIGITS + PUNCT_CORE + PUNCT_EXT)

#: Letters, digits and reliable punctuation. The default training inventory.
LATIN_CORE = _latin("latin_core", UPPER + LOWER + DIGITS + PUNCT_CORE)


# --------------------------------------------------------------------------- #
# Seed sets: the glyphs the *user* actually handwrites.
# --------------------------------------------------------------------------- #
#
# Chosen to span the style dimensions a Latin typeface varies along, rather than
# to be alphabetically tidy. Each letter below is carrying a specific signal:
#
#   ascender shape        b k l h        descender shape      g y p
#   arch / shoulder       n m r          closed bowl          o e a
#   diagonal stress       v w x A        s-curve              s S
#   terminal treatment    t f c          cap proportions      H O E M
#   joint / spur detail   a g R G
#
# 24 glyphs is the default because it sits inside the brief's 20-30 budget while
# still covering both cases and the digit style, which is often independent.

SEED_24 = "AEGHMORS" "abegkmnorsty" "036" ","

#: Minimal set for the ablation "how few glyphs can we get away with?".
SEED_12 = "AHOR" "aeglnos"","

#: Comfortable upper end of the brief's range; adds cases that are easy to get
#: wrong when generated (dot placement, crossbar height, leg angles).
SEED_30 = SEED_24 + "BDNfijp"[:6]

LATIN_SEED_SETS: dict[str, str] = {
    "seed12": SEED_12,
    "seed24": SEED_24,
    "seed30": SEED_30,
}


def seed_charset(base: Charset, seed: str) -> Charset:
    """Resolve a seed-set name or literal string against ``base``."""
    chars = LATIN_SEED_SETS.get(seed, seed)
    missing = sorted(set(chars) - set(base.chars))
    if missing:
        raise ValueError(f"seed chars {missing} are not in charset {base.name!r}")
    return base.subset(f"{base.name}:{seed}", chars)


CHARSETS: dict[str, Charset] = {
    c.name: c for c in (LATIN_LETTERS, LATIN_CORE, LATIN_FULL)
}


def get_charset(name: str) -> Charset:
    try:
        return CHARSETS[name]
    except KeyError:
        raise KeyError(
            f"unknown charset {name!r}; available: {sorted(CHARSETS)}"
        ) from None


def unicode_name(char: str) -> str:  # pragma: no cover - reporting aid
    try:
        return unicodedata.name(char)
    except ValueError:
        return f"U+{ord(char):04X}"
