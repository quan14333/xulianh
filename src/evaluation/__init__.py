"""Evaluation package for Infrared-Visible Image Fusion (IVIF).

Provides verified implementations of the 8 mandatory metrics from CVPR 2025 SAGE (Figure 5):
No-Reference IQA Metrics:
- PaQ-2-PiQ (Patch-to-Picture IQA)
- MUSIQ (Multi-Scale Image Quality Transformer)
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- NIQE (Natural Image Quality Evaluator, scaled by 10 in plot)

Fusion Fidelity Metrics:
- EN (Entropy, scaled by 5 in plot)
- SD (Standard Deviation, scaled by 0.5 in plot)
- SCD (Sum of Correlation Differences, scaled by 20 in plot)
- MS-SSIM (Multi-Scale Structural Similarity Index, scaled by 20 in plot)

Along with dataset evaluators, image I/O, and contract-compliant reporting.
"""

from __future__ import annotations

from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator
from src.evaluation.image_io import extract_channel, load_image, rgb_to_y, validate_matching_shapes
from src.evaluation.iqa_metrics import (
    METRIC_DIRECTIONS,
    PAPER_SCALING_FACTORS,
    apply_paper_scaling,
    brisque,
    musiq,
    niqe,
    paq2piq,
)
from src.evaluation.metrics import (
    average_gradient,
    edge_correlation,
    entropy,
    fusion_ms_ssim,
    ms_ssim,
    pearson_correlation,
    q_abf,
    single_scale_ssim,
    sobel_gradients,
    spatial_frequency,
    ssim_components,
    standard_deviation,
    sum_of_correlation_differences,
)

__all__ = [
    "METRIC_DIRECTIONS",
    "PAPER_SCALING_FACTORS",
    "EvaluationConfig",
    "FusionEvaluator",
    "apply_paper_scaling",
    "average_gradient",
    "brisque",
    "edge_correlation",
    "entropy",
    "extract_channel",
    "fusion_ms_ssim",
    "load_image",
    "ms_ssim",
    "musiq",
    "niqe",
    "paq2piq",
    "pearson_correlation",
    "q_abf",
    "rgb_to_y",
    "single_scale_ssim",
    "sobel_gradients",
    "spatial_frequency",
    "ssim_components",
    "standard_deviation",
    "sum_of_correlation_differences",
    "validate_matching_shapes",
]
