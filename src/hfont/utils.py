"""Small shared helpers: seeding, device selection, image grids, checkpoints."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Costs ~10-20% throughput; worth it when comparing two runs, not
        # otherwise.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} ({props.total_memory / 1e9:.1f} GB)"
    return device.type


def to_json(obj) -> dict:
    """Serialize nested dataclasses/configs for checkpoint metadata."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    return obj


def image_grid(
    images: torch.Tensor, n_cols: int, pad: int = 2, pad_value: float = 0.35
) -> np.ndarray:
    """Tile (N, 1, H, W) tensors in [-1, 1] into a single uint8 image.

    Output is ink-on-white, matching how the glyphs are actually read, rather
    than the raw signed tensor.
    """
    images = images.detach().float().cpu()
    n, _, h, w = images.shape
    n_cols = max(1, min(n_cols, n))
    n_rows = (n + n_cols - 1) // n_cols

    canvas = np.full(
        (n_rows * (h + pad) + pad, n_cols * (w + pad) + pad), pad_value, dtype=np.float32
    )
    ink = ((images + 1.0) * 0.5).clamp(0, 1).numpy()[:, 0]
    for i in range(n):
        r, c = divmod(i, n_cols)
        y = r * (h + pad) + pad
        x = c * (w + pad) + pad
        canvas[y : y + h, x : x + w] = 1.0 - ink[i]  # white paper, dark ink
    return (canvas * 255).astype(np.uint8)


def save_image_grid(path: str | Path, images: torch.Tensor, n_cols: int) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_grid(images, n_cols)).save(path)


class EMA:
    """Exponential moving average of model weights.

    Averaged weights are noticeably cleaner for this task than the raw ones:
    glyph edges stop jittering between steps, which matters because the raster
    is traced to curves and per-step jitter shows up as outline wobble.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().float(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights in, returning the originals for restoration."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}


def atomic_save(obj, path: str | Path) -> None:
    """Write a checkpoint via a temp file, then rename.

    Colab runtimes are killed without warning. A plain ``torch.save`` that is
    interrupted leaves a truncated file, which then fails to load on resume —
    silently costing the whole run. Renaming is atomic, so the destination is
    either the old checkpoint or the complete new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class CSVLogger:
    """Append-only metrics log. Survives restarts; trivial to plot later."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        if self.path.exists():
            first = self.path.read_text(encoding="utf-8").splitlines()
            if first:
                self._fields = first[0].split(",")

    def log(self, row: dict) -> None:
        if self._fields is None:
            self._fields = list(row)
            with self.path.open("w", encoding="utf-8") as fh:
                fh.write(",".join(self._fields) + "\n")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(",".join(_fmt(row.get(f, "")) for f in self._fields) + "\n")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(payload), indent=1), encoding="utf-8")
