"""Font corpus discovery, indexing and splitting.

The corpus is described by a JSON manifest of :class:`FontEntry` records rather
than by a directory walk at training time, so that a run is reproducible from
one file and a split can be audited.

Two things here matter more than they look:

*Splitting is by family, never by file.* Roboto ships 18 weights. Putting
Roboto-Light in train and Roboto-Medium in validation would let the model see
essentially the target style during training and report a validation number that
means nothing. Families are hashed to a split, so every weight of a family lands
together.

*Per-family caps.* Those same 18 weights would otherwise make Roboto 18x more
influential than a one-weight display face, biasing the style prior toward the
handful of superfamilies that happen to be large.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..charset import Charset
from .render import FontRenderError, FontRenderer

log = logging.getLogger(__name__)

FONT_SUFFIXES = {".ttf", ".otf"}

#: Google Fonts' own taxonomy. HANDWRITING is the slice closest to the actual
#: inference-time target, so it is tracked separately everywhere.
CATEGORIES = ("SANS_SERIF", "SERIF", "DISPLAY", "HANDWRITING", "MONOSPACE", "UNKNOWN")

GOOGLE_FONTS_REPO = "https://github.com/google/fonts.git"


@dataclass
class FontEntry:
    font_id: str
    path: str
    family: str
    style: str
    category: str
    source: str
    coverage: float
    n_glyphs: int
    units_per_em: int
    split: str = "train"

    @property
    def is_handwriting(self) -> bool:
        return self.category == "HANDWRITING"


@dataclass
class Manifest:
    charset: str
    entries: list[FontEntry] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "charset": self.charset,
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            charset=payload["charset"],
            entries=[FontEntry(**e) for e in payload["entries"]],
        )

    def split_entries(self, split: str) -> list[FontEntry]:
        return [e for e in self.entries if e.split == split]

    def summary(self) -> str:
        by_split = Counter(e.split for e in self.entries)
        by_cat = Counter(e.category for e in self.entries)
        families = len({e.family for e in self.entries})
        lines = [
            f"{len(self.entries)} fonts / {families} families / charset={self.charset}",
            "  splits:     " + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())),
            "  categories: " + ", ".join(f"{k}={v}" for k, v in by_cat.most_common()),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Google Fonts metadata
# --------------------------------------------------------------------------- #

_CATEGORY_RE = re.compile(r'^\s*category:\s*"([A-Z_]+)"', re.MULTILINE)
_NAME_RE = re.compile(r'^\s*name:\s*"([^"]+)"', re.MULTILINE)


def read_google_metadata(directory: Path) -> tuple[str | None, str | None]:
    """Parse ``METADATA.pb`` for (family name, category).

    Parsed with regexes rather than a protobuf dependency: the two fields we
    need are top-level scalars with a stable spelling, and pulling in
    ``protobuf`` plus the Google Fonts schema to read them would be absurd.
    """
    meta = directory / "METADATA.pb"
    if not meta.is_file():
        return None, None
    try:
        text = meta.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    name_match = _NAME_RE.search(text)
    cat_match = _CATEGORY_RE.search(text)
    return (
        name_match.group(1) if name_match else None,
        cat_match.group(1) if cat_match else None,
    )


def discover_fonts(root: str | Path) -> list[Path]:
    """All font files under ``root``, skipping known-bad directories."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in FONT_SUFFIXES else []
    out: list[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in FONT_SUFFIXES or not path.is_file():
            continue
        parts = {p.lower() for p in path.parts}
        # google/fonts keeps pre-release and archived copies alongside the real
        # ones; including them duplicates families under near-identical names.
        if parts & {".git", "archive", "axisregistry", "tools", "catalog"}:
            continue
        out.append(path)
    return sorted(out)


def _split_for(family: str, val_frac: float, test_frac: float) -> str:
    """Deterministic family-level split from a hash of the family name."""
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], "big") / 2**64
    if x < test_frac:
        return "test"
    if x < test_frac + val_frac:
        return "val"
    return "train"


def _style_rank(style: str) -> tuple[int, str]:
    """Sort key preferring the most typical styles when capping a family."""
    s = style.lower()
    if s in ("regular", "normal", "book"):
        return (0, s)
    if s == "italic":
        return (1, s)
    if s in ("bold", "medium"):
        return (2, s)
    if "italic" in s:
        return (4, s)
    return (3, s)


def build_manifest(
    roots: list[str | Path],
    charset: Charset,
    *,
    min_coverage: float = 1.0,
    max_per_family: int = 4,
    val_frac: float = 0.08,
    test_frac: float = 0.08,
    source: str = "google-fonts",
    progress_every: int = 250,
) -> Manifest:
    """Open every candidate font, keep the usable ones, and assign splits.

    ``min_coverage`` of 1.0 keeps only fonts that can draw the *entire* charset.
    That is the right default for training: a font missing a few glyphs would
    otherwise contribute style references for characters it cannot supply
    ground truth for, and the reconstruction loss would be computed against a
    hole.
    """
    candidates: list[tuple[Path, str | None, str | None]] = []
    for root in roots:
        for path in discover_fonts(root):
            family, category = read_google_metadata(path.parent)
            candidates.append((path, family, category))

    log.info("found %d candidate font files", len(candidates))

    by_family: dict[str, list[FontEntry]] = defaultdict(list)
    n_failed = 0

    for i, (path, meta_family, meta_category) in enumerate(candidates):
        if progress_every and i and i % progress_every == 0:
            log.info("  indexed %d/%d (%d unusable)", i, len(candidates), n_failed)
        try:
            with FontRenderer(path) as renderer:
                available = renderer.available(charset)
                coverage = len(available) / max(len(charset), 1)
                if coverage < min_coverage:
                    n_failed += 1
                    continue
                # A font that parses but draws nothing is worse than one that
                # fails to open, because it fails silently later.
                renderer.measure(available)
                family = meta_family or renderer.family
                entry = FontEntry(
                    font_id=f"{path.parent.name}/{path.stem}",
                    path=str(path),
                    family=family,
                    style=renderer.style,
                    category=meta_category or "UNKNOWN",
                    source=source,
                    coverage=coverage,
                    n_glyphs=len(available),
                    units_per_em=renderer.units_per_em,
                )
        except FontRenderError as exc:
            log.debug("skip %s: %s", path.name, exc)
            n_failed += 1
            continue
        except Exception as exc:  # a corrupt font should not kill an index run
            log.debug("skip %s: unexpected %s: %s", path.name, type(exc).__name__, exc)
            n_failed += 1
            continue
        by_family[family].append(entry)

    entries: list[FontEntry] = []
    for family, group in sorted(by_family.items()):
        group.sort(key=lambda e: _style_rank(e.style))
        split = _split_for(family, val_frac, test_frac)
        for entry in group[:max_per_family]:
            entry.split = split
            entries.append(entry)

    log.info(
        "kept %d fonts from %d families (%d unusable)",
        len(entries), len(by_family), n_failed,
    )
    return Manifest(charset=charset.name, entries=entries)


def clone_google_fonts(dest: str | Path, depth: int = 1) -> Path:
    """Shallow-clone the Google Fonts repository (~1.2 GB).

    Returns the clone directory. Safe to call again: an existing clone is left
    alone rather than re-downloaded.
    """
    import subprocess

    dest = Path(dest)
    if (dest / "ofl").is_dir():
        log.info("google/fonts already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), "--single-branch", GOOGLE_FONTS_REPO, str(dest)]
    log.info("cloning google/fonts into %s (this takes a few minutes)", dest)
    subprocess.run(cmd, check=True)
    return dest
