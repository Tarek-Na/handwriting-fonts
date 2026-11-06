"""Get SD19's per-writer labels, check them, and describe the data.

The 2nd edition's by_write PNGs have no labels; the 1995 1st edition has a
``.cls`` file per field (see hfont.data.sd19). This script:

1. ``--fetch``: downloads only the by_write ``.cls`` files (about 10,700, ~1 MB)
   out of ``1stEdition1995.zip`` (406 MB) with HTTP range requests, into
   ``data/sd19/by_write_labels.json``. No need to download the whole zip.
2. Cross-checks those labels against ``by_field.zip``, which sorts the same
   images into class folders: an image's pixels hashed in by_write must land
   in the folder of its label. A partial by_field download is fine; whatever
   members are complete are used.
3. Prints coverage (how many writers have each letter) and where letters sit
   in the 128x128 box per class (are descenders lower? are sizes kept?).

    python scripts/sd19_labels.py --fetch
    python scripts/sd19_labels.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfont.data.sd19 import FIELDS, index_writers  # noqa: E402

FIRST_EDITION = "https://s3.amazonaws.com/nist-srd/SD19/1stEdition1995.zip"
SD19 = ROOT / "data/sd19"


# --- reading members of a remote zip by range requests ----------------------

def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    error: Exception | None = None
    for _ in range(5):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:  # network hiccup: retry
            error = e
    raise error  # type: ignore[misc]


class _RemoteFile(io.RawIOBase):
    """Seekable read-only view of a URL, enough for zipfile to read the directory."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        data = _range(self.url, self.pos, min(self.size, self.pos + len(buffer)) - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _member(url: str, info: zipfile.ZipInfo) -> bytes:
    blob = _range(url, info.header_offset, info.header_offset + 30 + 512 + info.compress_size)
    n, m = struct.unpack("<HH", blob[26:30])
    data = blob[30 + n + m : 30 + n + m + info.compress_size]
    return data if info.compress_type == 0 else zlib.decompress(data, -15)


def fetch_labels(out: Path) -> None:
    t = time.time()
    z = zipfile.ZipFile(io.BufferedReader(_RemoteFile(FIRST_EDITION), buffer_size=1 << 16))
    wanted = [i for i in z.infolist() if "/by_write/" in i.filename and i.filename.endswith(".cls")
              and Path(i.filename).name[0] in FIELDS]
    print(f"{len(wanted)} label files listed ({time.time() - t:.0f}s); fetching")
    labels = {}
    with ThreadPoolExecutor(32) as pool:
        for info, data in zip(wanted, pool.map(lambda i: _member(FIRST_EDITION, i), wanted)):
            labels[info.filename.split("/by_write/")[1]] = data.decode("ascii")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels), encoding="utf-8")
    print(f"wrote {len(labels)} label files to {out} ({time.time() - t:.0f}s)")


# --- checks -----------------------------------------------------------------

def _pixels(png: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(png)).convert("L")) < 128


def _hash(png: bytes) -> str:
    return hashlib.blake2b(np.packbits(_pixels(png)).tobytes(), digest_size=10).hexdigest()


def _complete_members(path: Path):
    """(name, bytes) of every complete member, read from local headers.

    Works on a truncated download, which has no central directory.
    """
    with open(path, "rb") as f:
        while True:
            head = f.read(30)
            if len(head) < 30 or head[:4] != b"PK\x03\x04":
                return
            method = struct.unpack("<H", head[8:10])[0]
            size = struct.unpack("<I", head[18:22])[0]
            n, m = struct.unpack("<HH", head[26:30])
            name = f.read(n).decode()
            f.read(m)
            data = f.read(size)
            if len(data) < size:
                return
            if name.endswith(".png"):
                yield name, data if method == 0 else zlib.decompress(data, -15)


def cross_check(writers, by_write: zipfile.ZipFile, by_field: Path) -> None:
    classes: dict[str, set[str]] = defaultdict(set)
    groups = set()
    for name, data in _complete_members(by_field):
        parts = name.split("/")  # by_field/hsf_N/{lower,upper,digit}/<hex>/<file>
        if len(parts) == 5 and parts[2] in ("lower", "upper", "digit"):
            classes[_hash(data)].add(chr(int(parts[3], 16)))
            groups.add(parts[1])
    agree = disagree = ambiguous = unseen = 0
    examples = []
    for writer in writers:
        if writer.writer_id.split("/")[0] not in groups:
            continue
        for ch, names in writer.members.items():
            for name in names:
                found = classes.get(_hash(by_write.read(name)))
                if not found:
                    unseen += 1
                elif len(found) > 1:
                    ambiguous += 1
                elif ch in found:
                    agree += 1
                else:
                    disagree += 1
                    if len(examples) < 8:
                        examples.append((name, ch, "".join(found)))
    total = agree + disagree
    print(f"\nlabel cross-check against by_field ({', '.join(sorted(groups))}):")
    print(f"  agree {agree} / {total} ({agree / max(total, 1):.2%}); "
          f"{ambiguous} images match several classes (skipped), {unseen} not in by_field")
    for name, ch, other in examples:
        print(f"  disagree: {name} labelled {ch!r}, by_field has {other!r}")


def coverage(writers) -> None:
    lower, upper, digits = (set(chr(c) for c in range(97, 123)),
                            set(chr(c) for c in range(65, 91)), set("0123456789"))
    print(f"\n{len(writers)} writers with labelled fields")
    for label, chars in (("a-z", lower), ("A-Z", upper), ("0-9", digits)):
        have = [len(w.chars() & chars) for w in writers]
        full = sum(h == len(chars) for h in have)
        print(f"  {label}: complete for {full} writers; median {np.median(have):.0f} of {len(chars)}")
    letters = [len(w.chars() & (lower | upper)) for w in writers]
    for n in (52, 48, 44, 40, 30):
        print(f"  writers with >= {n} of 52 letters: {sum(x >= n for x in letters)}")


def placement(writers, by_write: zipfile.ZipFile, n_writers: int = 300) -> None:
    centre, height = defaultdict(list), defaultdict(list)
    for writer in writers[:: max(1, len(writers) // n_writers)]:
        for ch, names in writer.members.items():
            ink = _pixels(by_write.read(names[0]))
            rows = np.nonzero(ink.any(axis=1))[0]
            if rows.size:
                centre[ch].append((rows[0] + rows[-1] + 1) / 2)
                height[ch].append(rows[-1] + 1 - rows[0])
    print("\nwhere letters sit in the 128x128 box (median over writers):")
    print("  char  centre-row  ink-height")
    for ch in "oxaenbdhklgpqyjfAOgQ":
        if centre[ch]:
            print(f"  {ch:>4}  {np.median(centre[ch]):10.1f}  {np.median(height[ch]):10.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the .cls labels first")
    ap.add_argument("--labels", default=str(SD19 / "by_write_labels.json"))
    ap.add_argument("--by-write", default=str(SD19 / "by_write.zip"))
    ap.add_argument("--by-field", default=str(SD19 / "by_field.zip"))
    args = ap.parse_args()
    labels_path = Path(args.labels)
    if args.fetch:
        fetch_labels(labels_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(args.by_write) as by_write:
        writers = index_writers(by_write, labels)
        coverage(writers)
        placement(writers, by_write)
        if Path(args.by_field).is_file():
            cross_check(writers, by_write, Path(args.by_field))
    counts = Counter(len(w.members) for w in writers)
    print(f"\ncharacters per writer: {sorted(counts.items())[-6:]} (top end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
