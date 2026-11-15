"""Which Arabic letter form is in a box: the sheet reader's check.

A photographed sheet says where each form should be, but not that the
writer put it there: a box may hold the wrong letter (a skipped row shifts
everything after it), a form written for another (a medial beh in the
initial box), or a scribble. This classifier names the form in a box, so
the reader can flag a box whose writing does not match its label before it
goes into the font.

Input: a letter on the positional-form canvas (``formgen.to_canvas``, 64 x
256, AHAWP framing at half size). Output: logits over the 65 AHAWP forms.
Trained on the AHAWP training writers only (``scripts/train_formclass.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassConfig:
    classes: int = 65
    width: int = 32


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class FormClassifier(nn.Module):
    def __init__(self, config: ClassConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or ClassConfig()
        w = c.width
        self.features = nn.Sequential(_block(1, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * w, c.classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,1,64,256) ink in [0,1] -> logits (B, classes)."""
        f = self.features(x)
        return self.head(torch.amax(f, dim=(2, 3)))     # where on the canvas does not matter

    def checkpoint(self, names: list[str]) -> dict:
        return {"config": asdict(self.config), "model": self.state_dict(), "names": names}

    @classmethod
    def from_checkpoint(cls, data: dict) -> tuple["FormClassifier", list[str]]:
        model = cls(ClassConfig(**data["config"]))
        model.load_state_dict(data["model"])
        return model, data["names"]


def augment(x: torch.Tensor, shift: float = 0.06, scale: float = 0.12, rotate: float = 0.08) -> torch.Tensor:
    """Random small shift, scale and rotation, as boxes differ in where and how big a letter sits."""
    b = x.shape[0]
    s = 1 + (torch.rand(b, device=x.device) * 2 - 1) * scale
    a = (torch.rand(b, device=x.device) * 2 - 1) * rotate
    t = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1] = s * torch.cos(a), -s * torch.sin(a) * 0.25
    theta[:, 1, 0], theta[:, 1, 1] = s * torch.sin(a) * 4.0, s * torch.cos(a)
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
