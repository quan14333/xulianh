"""Core mathematical metric implementations for image fusion evaluation.

Supported metrics:
- EN (Entropy)
- SD (Standard Deviation)
- SCD (Sum of Correlation Differences)
- SSIM / MS-SSIM (Multi-Scale Structural Similarity Index)

References:
1. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
2. Hou, J., et al. (2014) / Aslan, M. S., et al. (2015). A novel method for image fusion
   performance evaluation based on sum of correlations of differences. Optik / Signal Processing.
3. Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural similarity for
   image quality assessment. In IEEE Asilomar Conference on Signals, Systems and Computers.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def entropy(image: np.ndarray, bins: int = 256, eps: float = 1e-12) -> float:
    """Calculate the Shannon Entropy (EN) of a single-channel image.

    Formula:
        EN = - \\sum_{i=0}^{L-1} p_i \\log_2(p_i)
    where p_i is the empirical probability of intensity level i, with p_i > 0.

    Properties:
        - Theoretical bounds: [0, log2(bins)]. For 256 bins, max EN = 8.0 bits/pixel.
        - Constant image yields EN = 0.0.
        - Measures the amount of information / richness of gray levels in the fused image.

    Args:
        image: 2D numpy array of shape (H, W).
        bins: Number of histogram bins (default: 256 for 8-bit images).
        eps: Small epsilon for numerical safety.

    Returns:
        Entropy value in bits/pixel as float.
    """
    if image.size == 0:
        return 0.0

    arr = np.asarray(image, dtype=np.float64)
    # Check if image is constant
    min_val, max_val = float(np.min(arr)), float(np.max(arr))
    if math.isclose(min_val, max_val, abs_tol=eps):
        return 0.0

    # If integer-like in [0, 255]
    if np.issubdtype(image.dtype, np.integer) and min_val >= 0 and max_val < bins:
        counts = np.bincount(image.ravel(), minlength=bins).astype(np.float64)
    else:
        # Continuous or normalized values
        counts, _ = np.histogram(arr, bins=bins, range=(0.0, 255.0 if max_val > 1.0 else 1.0))

    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def standard_deviation(image: np.ndarray, ddof: int = 0) -> float:
    """Calculate the Standard Deviation (SD) of an image.

    Formula:
        SD = \\sqrt{ \\frac{1}{N - ddof} \\sum_{k=1}^N (I_k - \\mu)^2 }
    where \\mu is the mean intensity.

    Convention:
        - In image fusion literature (e.g. VIFB, SAGE, SeAFusion), ddof=0
          (population standard deviation) is the standard convention.
        - Inputs are cast to float64 prior to subtraction to prevent uint8 overflow.

    Properties:
        - Measures overall contrast and intensity dispersion.
        - Constant image yields SD = 0.0.

    Args:
        image: 2D or 3D numpy array.
        ddof: Delta Degrees of Freedom (default: 0).

    Returns:
        Standard deviation as float.
    """
    if image.size == 0:
        return 0.0
    arr = np.asarray(image, dtype=np.float64)
    return float(np.std(arr, ddof=ddof))


def pearson_correlation(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Calculate Pearson Correlation Coefficient r between two images.

    Formula:
        r(A, B) = \\frac{\\sum (A_i - \\bar{A})(B_i - \\bar{B})}
                       {\\sqrt{\\sum (A_i - \\bar{A})^2} \\sqrt{\\sum (B_i - \\bar{B})^2}}

    Args:
        a: Array of shape (H, W).
        b: Array of shape (H, W).
        eps: Threshold below which image is treated as zero-variance.

    Returns:
        Correlation value in [-1.0, 1.0], or 0.0 if either image is degenerate.
    """
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()

    if a_f.size != b_f.size or a_f.size == 0:
        return 0.0

    a_c = a_f - np.mean(a_f)
    b_c = b_f - np.mean(b_f)

    norm_a = float(np.linalg.norm(a_c))
    norm_b = float(np.linalg.norm(b_c))

    if norm_a < eps or norm_b < eps:
        return 0.0

    r = float(np.dot(a_c, b_c) / (norm_a * norm_b))
    return float(np.clip(r, -1.0, 1.0))


def sum_of_correlation_differences(
    fused: np.ndarray,
    ir: np.ndarray,
    vis: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Calculate the Sum of Correlation Differences (SCD).

    Formula (Aslan et al. 2015 / Hou et al. 2014):
        D_1 = F - B  (Difference between fused and visible)
        D_2 = F - A  (Difference between fused and infrared)
        SCD = r(F - B, A) + r(F - A, B)
            = r(F - VIS, IR) + r(F - IR, VIS)

    Mathematical rationale:
        If F successfully captures complementary features from both modalities:
        - Removing the visible content (F - VIS) should leave residual information
          that strongly correlates with the infrared source IR: r(F - VIS, IR).
        - Removing the infrared content (F - IR) should leave residual information
          that strongly correlates with the visible source VIS: r(F - IR, VIS).

    Key properties verified:
        - Invariance to additive constant: SCD(F + c, IR, VIS) == SCD(F, IR, VIS).
        - Theoretical bounds: [-2.0, 2.0]. Higher values indicate superior information transfer.
        - Casts inputs to float64 before differences to avoid uint8 underflow.

    Args:
        fused: Fused image (H, W).
        ir: Infrared image (H, W).
        vis: Visible image (H, W).
        eps: Epsilon threshold for zero-variance difference detection.

    Returns:
        SCD value in [-2.0, 2.0].
    """
    f = np.asarray(fused, dtype=np.float64)
    a = np.asarray(ir, dtype=np.float64)
    b = np.asarray(vis, dtype=np.float64)

    if f.shape != a.shape or f.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: fused={f.shape}, ir={a.shape}, vis={b.shape}"
        )

    # Difference maps: D_vis = F - IR, D_ir = F - VIS
    d_vis = f - a
    d_ir = f - b

    r_vis = pearson_correlation(d_vis, b, eps=eps)
    r_ir = pearson_correlation(d_ir, a, eps=eps)

    return float(r_vis + r_ir)


# --- 2D Separable Gaussian Convolution & Downsampling for MS-SSIM ---


def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Generate a normalized 1D Gaussian kernel."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def _conv1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Perform 1D convolution with reflection padding along a specified axis."""
    k = len(kernel)
    pad = k // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
        return sum(w * padded[i : i + arr.shape[0], :] for i, w in enumerate(kernel))
    elif axis == 1:
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
        return sum(w * padded[:, i : i + arr.shape[1]] for i, w in enumerate(kernel))
    else:
        raise ValueError("Axis must be 0 or 1 for 2D array.")


def _conv2d_separable_gaussian(
    arr: np.ndarray,
    kernel_1d: np.ndarray,
) -> np.ndarray:
    """Convolve 2D array with separable 1D Gaussian kernel."""
    return _conv1d_reflect(_conv1d_reflect(arr, kernel_1d, axis=0), kernel_1d, axis=1)


def _downsample_2x2(arr: np.ndarray) -> np.ndarray:
    """Downsample image by 2x using 2x2 average decimation."""
    h, w = arr.shape
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    cropped = arr[:h_even, :w_even]
    return (
        cropped[0::2, 0::2]
        + cropped[1::2, 0::2]
        + cropped[0::2, 1::2]
        + cropped[1::2, 1::2]
    ) * 0.25


def ssim_components(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[float, float, float]:
    """Compute the three SSIM components (luminance, contrast, structure).

    Formulas (Wang et al. 2003, 2004):
        l(x, y) = (2 * mu_x * mu_y + C_1) / (mu_x^2 + mu_y^2 + C_1)
        c(x, y) = (2 * sigma_x * sigma_y + C_2) / (sigma_x^2 + sigma_y^2 + C_2)
        s(x, y) = (sigma_xy + C_3) / (sigma_x * sigma_y + C_3)
    where:
        C_1 = (K_1 * L)^2
        C_2 = (K_2 * L)^2
        C_3 = C_2 / 2

    Returns:
        tuple (mean_luminance, mean_contrast, mean_structure)
    """
    x_f = np.asarray(x, dtype=np.float64)
    y_f = np.asarray(y, dtype=np.float64)

    if x_f.shape != y_f.shape:
        raise ValueError(f"Shape mismatch in SSIM: {x_f.shape} vs {y_f.shape}")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c3 = c2 / 2.0

    kernel = _gaussian_kernel_1d(size=kernel_size, sigma=sigma)

    mu_x = _conv2d_separable_gaussian(x_f, kernel)
    mu_y = _conv2d_separable_gaussian(y_f, kernel)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = np.maximum(_conv2d_separable_gaussian(x_f * x_f, kernel) - mu_x_sq, 0.0)
    sigma_y_sq = np.maximum(_conv2d_separable_gaussian(y_f * y_f, kernel) - mu_y_sq, 0.0)
    sigma_xy = _conv2d_separable_gaussian(x_f * y_f, kernel) - mu_xy

    sigma_x = np.sqrt(sigma_x_sq)
    sigma_y = np.sqrt(sigma_y_sq)

    l_map = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
    c_map = (2.0 * sigma_x * sigma_y + c2) / (sigma_x_sq + sigma_y_sq + c2)
    s_map = (sigma_xy + c3) / (sigma_x * sigma_y + c3)

    return float(np.mean(l_map)), float(np.mean(c_map)), float(np.mean(s_map))


def single_scale_ssim(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Calculate standard single-scale Structural Similarity (SSIM).

    Formula:
        SSIM(x, y) = l(x, y) * c(x, y) * s(x, y)
    """
    lum_val, c_val, s_val = ssim_components(x, y, data_range=data_range)
    return float(lum_val * c_val * s_val)


DEFAULT_MS_SSIM_WEIGHTS: tuple[float, ...] = (
    0.0448,
    0.2856,
    0.3001,
    0.2363,
    0.1333,
)


def ms_ssim(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float = 255.0,
    weights: Sequence[float] = DEFAULT_MS_SSIM_WEIGHTS,
    clamp_negative: bool = True,
) -> float:
    """Calculate Multi-Scale Structural Similarity Index (MS-SSIM).

    Paper Reference:
        Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural
        similarity for image quality assessment. IEEE ACSSC.

    Formula:
        MS-SSIM(x, y) = [l_M(x, y)]^{\\alpha_M} \\prod_{j=1}^M [c_j(x, y)]^{\\beta_j} [s_j(x, y)]^{\\gamma_j}
    In standard literature:
        \\alpha_M = \\beta_M = \\gamma_M, and weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).

    Properties:
        - Multi-scale analysis preserves both low-frequency structural information
          and high-frequency details.
        - Perfect reconstruction MS-SSIM(A, A) == 1.0.
        - Values are bounded in [0.0, 1.0] when negative similarities are clamped.

    Args:
        x: First single-channel image (H, W).
        y: Second single-channel image (H, W).
        data_range: Dynamic range of pixel values (default: 255.0 for uint8 range).
        weights: Exponent weights for M scales. Default is standard 5-scale weights.
        clamp_negative: If True, clamp negative component terms to 0.0 to prevent
            complex numbers.

    Returns:
        MS-SSIM score as float in [0.0, 1.0].
    """
    x_curr = np.asarray(x, dtype=np.float64)
    y_curr = np.asarray(y, dtype=np.float64)

    if x_curr.shape != y_curr.shape:
        raise ValueError(f"Shape mismatch in MS-SSIM: {x_curr.shape} vs {y_curr.shape}")

    num_scales = len(weights)
    min_dim = min(x_curr.shape)
    # Check minimum dimension requirement for 5-level downsampling with 11x11 filter
    required_min = 11 * (2 ** (num_scales - 1))
    if min_dim < required_min:
        # Dynamic fallback: if image is small, adaptively reduce number of scales
        active_scales = max(1, int(math.floor(math.log2(min_dim / 11))) + 1)
        active_scales = min(active_scales, num_scales)
        active_weights = np.array(weights[:active_scales], dtype=np.float64)
        active_weights = active_weights / np.sum(active_weights)
    else:
        active_scales = num_scales
        active_weights = np.array(weights, dtype=np.float64)

    mcs_list: list[float] = []
    final_l = 1.0

    for i in range(active_scales):
        lum_comp, c_comp, s_comp = ssim_components(x_curr, y_curr, data_range=data_range)
        cs_prod = c_comp * s_comp
        if clamp_negative:
            cs_prod = max(cs_prod, 0.0)
            lum_comp = max(lum_comp, 0.0)

        mcs_list.append(cs_prod)

        if i == active_scales - 1:
            final_l = lum_comp
        else:
            x_curr = _downsample_2x2(x_curr)
            y_curr = _downsample_2x2(y_curr)

    # Combine multi-scale components
    # val = (l_M ^ beta_M) * prod_{j=1}^M (c_j * s_j) ^ beta_j
    val = (final_l ** active_weights[-1])
    for cs_val, w in zip(mcs_list, active_weights):
        val *= (cs_val ** w)

    return float(np.clip(val, 0.0, 1.0))


def fusion_ms_ssim(
    fused: np.ndarray,
    ir: np.ndarray,
    vis: np.ndarray,
    data_range: float = 255.0,
    aggregation: str = "mean",
    ir_weight: float = 0.5,
) -> dict[str, float]:
    """Calculate MS-SSIM for dual-modality Image Fusion (IVIF).

    Because image fusion combines two sources (IR and VIS) into one fused output F,
    structural fidelity must be measured against both sources.

    Aggregation Methods:
        1. 'mean':
            MS-SSIM_mean = 0.5 * MS-SSIM(F, IR) + 0.5 * MS-SSIM(F, VIS)
            Bounded in [0.0, 1.0]. Standard convention for paired similarity.
        2. 'sum':
            MS-SSIM_sum = MS-SSIM(F, IR) + MS-SSIM(F, VIS)
            Bounded in [0.0, 2.0]. Used by several IVIF benchmarks (e.g. PCDFusion / VIFB scripts).
        3. 'weighted':
            MS-SSIM_weighted = ir_weight * MS-SSIM(F, IR) + (1 - ir_weight) * MS-SSIM(F, VIS)
            Weighted by variance or energy.

    Args:
        fused: Fused image (H, W).
        ir: Infrared source image (H, W).
        vis: Visible source image (H, W).
        data_range: Dynamic range (default: 255.0).
        aggregation: Aggregation scheme ('mean', 'sum', or 'weighted').
        ir_weight: Weight for IR source when aggregation is 'weighted' (default: 0.5).

    Returns:
        dict containing:
            - 'MS_SSIM': aggregated score according to requested method
            - 'MS_SSIM_IR': MS-SSIM(F, IR)
            - 'MS_SSIM_VIS': MS-SSIM(F, VIS)
            - 'MS_SSIM_mean': equal-weight average
            - 'MS_SSIM_sum': sum of both components
    """
    ms_ir = ms_ssim(fused, ir, data_range=data_range)
    ms_vis = ms_ssim(fused, vis, data_range=data_range)

    mean_score = 0.5 * (ms_ir + ms_vis)
    sum_score = ms_ir + ms_vis

    if aggregation == "sum":
        primary = sum_score
    elif aggregation == "weighted":
        primary = ir_weight * ms_ir + (1.0 - ir_weight) * ms_vis
    else:
        primary = mean_score

    return {
        "MS_SSIM": float(primary),
        "MS_SSIM_IR": float(ms_ir),
        "MS_SSIM_VIS": float(ms_vis),
        "MS_SSIM_mean": float(mean_score),
        "MS_SSIM_sum": float(sum_score),
    }


def spatial_frequency(image: np.ndarray) -> float:
    """Calculate Spatial Frequency (SF) of an image.

    Formula (Eskicioglu & Fisher 1995):
        RF = \\sqrt{ \\frac{1}{M N} \\sum_{i=1}^M \\sum_{j=2}^N (I(i,j) - I(i,j-1))^2 }
        CF = \\sqrt{ \\frac{1}{M N} \\sum_{i=2}^M \\sum_{j=1}^N (I(i,j) - I(i-1,j))^2 }
        SF = \\sqrt{RF^2 + CF^2}

    Measures overall active gradient and spatial detail richness. Higher is better.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0

    m, n = arr.shape
    if m < 2 or n < 2:
        return 0.0

    # Row frequency (horizontal differences)
    rf_sq = np.sum((arr[:, 1:] - arr[:, :-1]) ** 2) / (m * n)
    # Column frequency (vertical differences)
    cf_sq = np.sum((arr[1:, :] - arr[:-1, :]) ** 2) / (m * n)

    return float(np.sqrt(rf_sq + cf_sq))


def average_gradient(image: np.ndarray) -> float:
    """Calculate Average Gradient (AG) of an image.

    Formula:
        AG = \\frac{1}{(M-1)(N-1)} \\sum_{i=1}^{M-1} \\sum_{j=1}^{N-1}
             \\sqrt{ \\frac{(I(i+1,j) - I(i,j))^2 + (I(i,j+1) - I(i,j))^2}{2} }

    Reflects the contrast of fine details and edge sharpness. Higher is better.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0

    m, n = arr.shape
    if m < 2 or n < 2:
        return 0.0

    dx = arr[1:, :-1] - arr[:-1, :-1]
    dy = arr[:-1, 1:] - arr[:-1, :-1]

    grad_mag = np.sqrt((dx**2 + dy**2) / 2.0)
    return float(np.mean(grad_mag))


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Sobel gradient magnitude and orientation angle in pure NumPy.

    Returns:
        tuple (magnitude, orientation_angle_in_radians)
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros_like(arr), np.zeros_like(arr)

    # 3x3 Sobel kernels
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64) / 8.0
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64) / 8.0

    pad = np.pad(arr, 1, mode="reflect")
    h, w = arr.shape
    gx = sum(kx[i, j] * pad[i : i + h, j : j + w] for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * pad[i : i + h, j : j + w] for i in range(3) for j in range(3))

    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    return mag, angle


def edge_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Pearson correlation between Sobel edge magnitudes of two images.

    Used by SAGE and PCDFusion benchmarks to assess edge preservation from IR and VIS.
    """
    mag_a, _ = sobel_gradients(a)
    mag_b, _ = sobel_gradients(b)
    return pearson_correlation(mag_a, mag_b)


def q_abf(
    fused: np.ndarray,
    ir: np.ndarray,
    vis: np.ndarray,
    gamma_g: float = 0.9994,
    k_g: float = -15.0,
    sigma_g: float = 0.5,
    gamma_a: float = 0.9879,
    k_a: float = -22.0,
    sigma_a: float = 0.8,
) -> float:
    """Calculate Q^{AB/F} (Xydeas & Petrovic 2004) edge information preservation metric.

    Formula:
        Q^{AB/F} = \\frac{\\sum_{i,j} (Q^{AF}(i,j) w_A(i,j) + Q^{BF}(i,j) w_B(i,j))}
                        {\\sum_{i,j} (w_A(i,j) + w_B(i,j))}

    Properties:
        - Objective non-reference image fusion quality metric.
        - Bounded in [0.0, 1.0]. Higher values indicate greater edge preservation.
        - Pure NumPy implementation, cross-platform and CPU-only.
    """
    f = np.asarray(fused, dtype=np.float64)
    a = np.asarray(ir, dtype=np.float64)
    b = np.asarray(vis, dtype=np.float64)

    if not (f.shape == a.shape == b.shape):
        raise ValueError(f"Shape mismatch in Q_ABF: {f.shape}, {a.shape}, {b.shape}")

    g_f, a_f = sobel_gradients(f)
    g_a, a_a = sobel_gradients(a)
    g_b, a_b = sobel_gradients(b)

    def _calc_preservation(g_src: np.ndarray, alpha_src: np.ndarray) -> np.ndarray:
        ratio = np.where(
            g_src > g_f,
            np.where(g_src > 1e-12, g_f / (g_src + 1e-12), 1.0),
            np.where(g_f > 1e-12, g_src / (g_f + 1e-12), 1.0),
        )
        q_g = gamma_g / (1.0 + np.exp(k_g * (ratio - sigma_g)))

        d_alpha = np.abs(alpha_src - a_f)
        d_alpha = np.where(d_alpha > np.pi / 2.0, np.pi - d_alpha, d_alpha)
        norm_d_alpha = 1.0 - (d_alpha / (np.pi / 2.0))
        q_a = gamma_a / (1.0 + np.exp(k_a * (norm_d_alpha - sigma_a)))

        return q_g * q_a

    q_af = _calc_preservation(g_a, a_a)
    q_bf = _calc_preservation(g_b, a_b)

    w_a = g_a
    w_b = g_b
    weight_total = float(np.sum(w_a + w_b))

    if weight_total < 1e-12:
        return 0.0

    score = float(np.sum(q_af * w_a + q_bf * w_b) / weight_total)
    return float(np.clip(score, 0.0, 1.0))
