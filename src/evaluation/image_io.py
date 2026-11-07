"""Image I/O and channel preprocessing utilities for fusion evaluation.

Compliant with:
- configs/evaluation/fusion_metrics.yaml
- docs/EXPERIMENT_PROTOCOL.md (strict input contract, float64 cast before subtraction)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def rgb_to_y(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to luminance (Y channel) using ITU-R BT.601.

    Formula:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
    """
    arr = np.asarray(rgb_arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3+), got {arr.shape}")
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def extract_channel(
    image: np.ndarray,
    channel: str = "Y",
) -> np.ndarray:
    """Extract requested channel from image array.

    Args:
        image: Numpy array of shape (H, W) or (H, W, C).
        channel: Channel selection:
            - 'Y' / 'YCbCr': Luminance channel (BT.601 standard for IVIF).
            - 'gray' / 'grayscale': Grayscale luminance.
            - 'RGB': Returns full RGB array (float64).

    Returns:
        Numpy array with selected channel in float64.
    """
    arr = np.asarray(image, dtype=np.float64)

    channel_normalized = channel.strip().upper()
    if channel_normalized in {"Y", "YCBCR", "GRAY", "GRAYSCALE"}:
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 3:
            return rgb_to_y(arr)
        else:
            raise ValueError(f"Unsupported image array dimensions: {arr.shape}")
    elif channel_normalized == "RGB":
        if arr.ndim == 2:
            # Replicate to 3 channels if grayscale
            return np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            return arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported image array dimensions: {arr.shape}")
    else:
        raise ValueError(
            f"Unsupported channel '{channel}'. Allowed: 'Y', 'gray', 'grayscale', 'RGB'."
        )


def load_image(
    path: Union[str, Path],
    target_channel: str = "Y",
    cast_dtype: type = np.float64,
) -> np.ndarray:
    """Load an image from disk and preprocess to requested channel and dtype.

    Args:
        path: Path to image file.
        target_channel: Target channel ('Y', 'gray', 'RGB').
        cast_dtype: Numpy float data type (default: np.float64).

    Returns:
        Preprocessed image array.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    with Image.open(file_path) as pil_img:
        # Load pixels into memory
        pil_img.load()
        if pil_img.mode in ("RGB", "RGBA"):
            arr = np.array(pil_img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
        elif pil_img.mode in ("L", "I", "F"):
            arr = np.array(pil_img)
        else:
            # Convert paletted / other modes to RGB first
            rgb_conv = pil_img.convert("RGB")
            arr = np.array(rgb_conv)

    channel_data = extract_channel(arr, channel=target_channel)
    return np.asarray(channel_data, dtype=cast_dtype)


def validate_matching_shapes(
    fused: np.ndarray,
    ir: np.ndarray,
    vis: np.ndarray,
    image_id: str = "",
) -> None:
    """Ensure fused, infrared, and visible images have identical spatial dimensions."""
    shape_f = fused.shape[:2]
    shape_ir = ir.shape[:2]
    shape_vis = vis.shape[:2]

    if not (shape_f == shape_ir == shape_vis):
        raise ValueError(
            f"Image shape mismatch for ID '{image_id}': "
            f"fused={shape_f}, ir={shape_ir}, vis={shape_vis}"
        )
