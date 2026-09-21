"""Evaluation harness shared by both phases."""

from .metrics import compare_rasters, evaluate_checkpoint
from .shaping import ShapingReport, check_font, shape_text

__all__ = [
    "compare_rasters",
    "evaluate_checkpoint",
    "ShapingReport",
    "check_font",
    "shape_text",
]
