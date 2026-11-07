"""Evaluation package for Infrared-Visible Image Fusion (IVIF).

Provides verified implementations of:
- EN (Entropy)
- SD (Standard Deviation)
- SCD (Sum of Correlation Differences)
- MS-SSIM (Multi-Scale Structural Similarity Index)
along with dataset evaluators, image I/O, and contract-compliant reporting.
"""

from __future__ import annotations

from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator
from src.evaluation.image_io import extract_channel, load_image, rgb_to_y, validate_matching_shapes
from src.evaluation.metrics import (
    entropy,
    fusion_ms_ssim,
    ms_ssim,
    pearson_correlation,
    single_scale_ssim,
    ssim_components,
    standard_deviation,
    sum_of_correlation_differences,
)

__all__ = [
    "EvaluationConfig",
    "FusionEvaluator",
    "entropy",
    "extract_channel",
    "fusion_ms_ssim",
    "load_image",
    "ms_ssim",
    "pearson_correlation",
    "rgb_to_y",
    "single_scale_ssim",
    "ssim_components",
    "standard_deviation",
    "sum_of_correlation_differences",
    "validate_matching_shapes",
]
