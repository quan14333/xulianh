"""Unit tests for core evaluation metrics (EN, SD, SCD, MS-SSIM).

All tests are CPU-only, deterministic, and verify mathematical invariants.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

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


def test_entropy_constant_image() -> None:
    """A constant image contains zero information: EN = 0.0."""
    const_img = np.full((100, 100), 128, dtype=np.uint8)
    assert math.isclose(entropy(const_img), 0.0, abs_tol=1e-9)


def test_entropy_uniform_distribution() -> None:
    """A discrete uniform distribution across 256 levels has entropy = log2(256) = 8.0."""
    # 256 * 100 pixels, each level occurs exactly 100 times
    uniform_img = np.tile(np.arange(256, dtype=np.uint8), 100).reshape(160, 160)
    en = entropy(uniform_img, bins=256)
    assert math.isclose(en, 8.0, abs_tol=1e-7)


def test_entropy_bimodal_distribution() -> None:
    """A 50/50 mixture of two values has entropy = 1.0 bit/pixel."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:, :50] = 255
    en = entropy(arr, bins=256)
    assert math.isclose(en, 1.0, abs_tol=1e-7)


def test_entropy_empty() -> None:
    """Empty image returns 0.0 safely."""
    empty = np.array([], dtype=np.uint8)
    assert entropy(empty) == 0.0


def test_standard_deviation_constant() -> None:
    """Constant image has zero variance."""
    const_img = np.full((50, 50), 42, dtype=np.uint8)
    assert math.isclose(standard_deviation(const_img), 0.0, abs_tol=1e-9)


def test_standard_deviation_known_values() -> None:
    """Verify ddof=0 population standard deviation against known values."""
    arr = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    # mean = 25, variance = ((15^2 + 5^2 + 5^2 + 15^2)/4) = (225 + 25 + 25 + 225)/4 = 500/4 = 125
    expected_std = math.sqrt(125.0)
    assert math.isclose(standard_deviation(arr, ddof=0), expected_std, rel_tol=1e-7)


def test_pearson_correlation_properties() -> None:
    """Verify Pearson correlation identity, anti-symmetry, and orthogonality."""
    rng = np.random.default_rng(1234)
    x = rng.normal(10.0, 2.0, size=(100, 100))

    # Identity: r(x, x) == 1.0
    assert math.isclose(pearson_correlation(x, x), 1.0, abs_tol=1e-7)

    # Inverted: r(x, -x) == -1.0
    assert math.isclose(pearson_correlation(x, -x), -1.0, abs_tol=1e-7)

    # Degenerate constant: r = 0.0
    const_img = np.ones((100, 100))
    assert pearson_correlation(x, const_img) == 0.0


def test_scd_constant_shift_invariance() -> None:
    """SCD must be mathematically invariant to adding a constant scalar to F.

    Since difference images shift by c, and Pearson correlation subtracts the mean,
    r(F + c - VIS, IR) == r(F - VIS, IR).
    """
    rng = np.random.default_rng(42)
    ir = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    vis = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    fused = (ir.astype(np.float64) + vis.astype(np.float64)) * 0.5

    scd_base = sum_of_correlation_differences(fused, ir, vis)
    scd_shifted = sum_of_correlation_differences(fused + 50.0, ir, vis)

    assert math.isclose(scd_base, scd_shifted, abs_tol=1e-9)


def test_scd_float64_casting_prevents_uint8_wrap() -> None:
    """SCD calculation on uint8 inputs must not suffer from modular wrap-around."""
    ir = np.array([[200, 250], [220, 240]], dtype=np.uint8)
    vis = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    fused = np.array([[100, 100], [100, 100]], dtype=np.uint8)

    # If computed in uint8, fused - ir would wrap around to large positive integers.
    # sum_of_correlation_differences casts to float64:
    scd_val = sum_of_correlation_differences(fused, ir, vis)
    assert isinstance(scd_val, float)
    assert -2.0 <= scd_val <= 2.0


def test_scd_shape_mismatch_raises() -> None:
    """Shape mismatch between inputs raises ValueError."""
    f = np.zeros((10, 10))
    ir = np.zeros((10, 12))
    vis = np.zeros((10, 10))
    with pytest.raises(ValueError, match="Shape mismatch"):
        sum_of_correlation_differences(f, ir, vis)


def test_ssim_components_identity() -> None:
    """SSIM components for identical images must all equal 1.0."""
    rng = np.random.default_rng(999)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float64)

    lum_comp, c, s = ssim_components(img, img, data_range=255.0)
    assert math.isclose(lum_comp, 1.0, abs_tol=1e-5)
    assert math.isclose(c, 1.0, abs_tol=1e-5)
    assert math.isclose(s, 1.0, abs_tol=1e-5)

    ssim_val = single_scale_ssim(img, img, data_range=255.0)
    assert math.isclose(ssim_val, 1.0, abs_tol=1e-5)


def test_ms_ssim_identity() -> None:
    """Multi-Scale SSIM between identical images must equal 1.0."""
    rng = np.random.default_rng(2026)
    img = rng.integers(10, 240, (180, 180), dtype=np.uint8)

    score = ms_ssim(img, img, data_range=255.0)
    assert math.isclose(score, 1.0, abs_tol=1e-4)


def test_ms_ssim_degraded_score_lower() -> None:
    """Adding significant noise reduces MS-SSIM score below 1.0."""
    rng = np.random.default_rng(2026)
    clean = rng.integers(50, 200, (180, 180), dtype=np.uint8)
    noisy = np.clip(clean.astype(float) + rng.normal(0, 30, (180, 180)), 0, 255).astype(np.uint8)

    clean_score = ms_ssim(clean, clean, data_range=255.0)
    noisy_score = ms_ssim(clean, noisy, data_range=255.0)

    assert clean_score > noisy_score
    assert 0.0 <= noisy_score <= 1.0


def test_ms_ssim_adaptive_scale_for_small_images() -> None:
    """Images smaller than 176x176 adaptively reduce scale count without error."""
    small_img = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
    score = ms_ssim(small_img, small_img, data_range=255.0)
    assert math.isclose(score, 1.0, abs_tol=1e-4)


def test_fusion_ms_ssim_aggregation_methods() -> None:
    """Verify mean, sum, and weighted aggregations for dual-source fusion."""
    rng = np.random.default_rng(777)
    ir = rng.integers(0, 256, (180, 180), dtype=np.uint8)
    vis = rng.integers(0, 256, (180, 180), dtype=np.uint8)
    fused = (ir.astype(float) * 0.7 + vis.astype(float) * 0.3).astype(np.uint8)

    res_mean = fusion_ms_ssim(fused, ir, vis, aggregation="mean")
    res_sum = fusion_ms_ssim(fused, ir, vis, aggregation="sum")
    res_wt = fusion_ms_ssim(fused, ir, vis, aggregation="weighted", ir_weight=0.7)

    # Check relationships
    assert math.isclose(res_mean["MS_SSIM"], res_mean["MS_SSIM_mean"], abs_tol=1e-9)
    assert math.isclose(res_sum["MS_SSIM"], res_sum["MS_SSIM_sum"], abs_tol=1e-9)
    assert math.isclose(res_sum["MS_SSIM_sum"], res_mean["MS_SSIM_IR"] + res_mean["MS_SSIM_VIS"], abs_tol=1e-9)
    assert math.isclose(res_mean["MS_SSIM_mean"], 0.5 * res_sum["MS_SSIM_sum"], abs_tol=1e-9)
    expected_wt = 0.7 * res_mean["MS_SSIM_IR"] + 0.3 * res_mean["MS_SSIM_VIS"]
    assert math.isclose(res_wt["MS_SSIM"], expected_wt, abs_tol=1e-9)


def test_spatial_frequency_properties() -> None:
    """Verify SF calculation on constant image (0.0) and high-frequency pattern."""
    const_img = np.full((50, 50), 100, dtype=np.uint8)
    assert math.isclose(spatial_frequency(const_img), 0.0, abs_tol=1e-9)

    # Checkerboard pattern has high spatial frequency
    checker = np.indices((50, 50)).sum(axis=0) % 2 * 255
    sf_val = spatial_frequency(checker)
    assert sf_val > 50.0

    # Invariant to additive constant shift
    assert math.isclose(spatial_frequency(checker), spatial_frequency(checker + 20), abs_tol=1e-9)


def test_average_gradient_properties() -> None:
    """Verify AG calculation on constant image and gradient ramp."""
    const_img = np.full((40, 40), 75, dtype=np.uint8)
    assert math.isclose(average_gradient(const_img), 0.0, abs_tol=1e-9)

    # Linear horizontal gradient ramp: I(x, y) = x
    ramp = np.tile(np.arange(40, dtype=np.float64), (40, 1))
    ag_val = average_gradient(ramp)
    # dx = 0, dy = 1 -> sqrt(1/2) = ~0.7071
    assert math.isclose(ag_val, 1.0 / math.sqrt(2.0), abs_tol=1e-5)


def test_sobel_gradients() -> None:
    """Verify Sobel gradient magnitude on vertical and horizontal steps."""
    img = np.zeros((20, 20), dtype=np.float64)
    img[:, 10:] = 100.0  # Vertical step
    mag, angle = sobel_gradients(img)
    assert mag.shape == (20, 20)
    assert angle.shape == (20, 20)
    assert float(np.max(mag[:, 9:11])) > 0.0


def test_edge_correlation_properties() -> None:
    """Verify Sobel edge correlation between identical and orthogonal images."""
    rng = np.random.default_rng(101)
    img = rng.integers(20, 220, (60, 60), dtype=np.uint8)

    # Self edge correlation is 1.0
    assert math.isclose(edge_correlation(img, img), 1.0, abs_tol=1e-7)

    # Inverted image has the exact same gradient magnitude -> edge_corr == 1.0
    inv = 255 - img
    assert math.isclose(edge_correlation(img, inv), 1.0, abs_tol=1e-5)


def test_q_abf_properties() -> None:
    """Verify Q_ABF bounds, constant image handling, and identity preservation."""
    rng = np.random.default_rng(202)
    ir = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    vis = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    fused = ((ir.astype(float) + vis.astype(float)) / 2.0).astype(np.uint8)

    score = q_abf(fused, ir, vis)
    assert 0.0 <= score <= 1.0

    # Constant images yield 0.0
    const_arr = np.full((80, 80), 128, dtype=np.uint8)
    assert math.isclose(q_abf(const_arr, const_arr, const_arr), 0.0, abs_tol=1e-9)

    # Fused preserving salient edges yields high Q_ABF
    # Target in IR:
    hot_ir = np.zeros((80, 80), dtype=np.uint8)
    hot_ir[30:50, 30:50] = 240
    hot_vis = np.zeros((80, 80), dtype=np.uint8)
    hot_fused = np.maximum(hot_ir, hot_vis)
    high_q = q_abf(hot_fused, hot_ir, hot_vis)
    assert high_q > 0.8

