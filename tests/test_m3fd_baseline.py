"""M3FD admission checks and independent local-map MS-SSIM reference."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.run_m3fd_baseline import audit_pairs, index_images
from src.evaluation.metrics import ms_ssim, single_scale_ssim
from src.evaluation.metrics import entropy
from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator
from scripts.run_evaluation import build_triplets


def reference_maps(x, y):
    """Direct 2D Gaussian windows, independent of separable production convolution."""
    coordinates = np.arange(11) - 5
    kernel = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / 4.5)
    kernel /= kernel.sum()

    def blur(a):
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(a, 5, mode="reflect"), (11, 11))
        return np.einsum("ijkl,kl->ij", windows, kernel)

    ux, uy = blur(x), blur(y)
    vx, vy = np.maximum(blur(x*x)-ux*ux, 0), np.maximum(blur(y*y)-uy*uy, 0)
    covariance = blur(x*y)-ux*uy
    luminance = (2*ux*uy+6.5025)/(ux*ux+uy*uy+6.5025)
    cs = (2*covariance+58.5225)/(vx+vy+58.5225)
    return np.mean(luminance*cs), np.mean(cs)


def test_local_ssim_against_independent_2d_reference():
    rng = np.random.default_rng(77)
    x = rng.uniform(0, 255, (48, 52))
    y = x.copy()
    y[:24] *= .3
    y[24:] = rng.uniform(0, 255, (24, 52))
    expected, _ = reference_maps(x, y)
    assert single_scale_ssim(x, y) == pytest.approx(expected, abs=1e-12)
    assert ms_ssim(x, y, weights=[1]) == pytest.approx(expected, abs=1e-12)


def test_multiscale_local_maps_against_independent_reference():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 255, (48, 52))
    y = .6*x + .4*rng.uniform(0, 255, x.shape)
    _, first_cs = reference_maps(x, y)
    # Explicit block averaging independent of production's slicing implementation.
    xd = x.reshape(24, 2, 26, 2).mean(axis=(1, 3))
    yd = y.reshape(24, 2, 26, 2).mean(axis=(1, 3))
    final_ssim, _ = reference_maps(xd, yd)
    expected = first_cs**.4 * final_ssim**.6
    assert ms_ssim(x, y, weights=[.4, .6]) == pytest.approx(expected, abs=1e-12)


def test_full_five_scale_reference():
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 255, (192, 208))
    y = .5*x + .5*rng.uniform(0, 255, x.shape)
    expected = 1.0
    a, b = x.copy(), y.copy()
    weights = [.0448, .2856, .3001, .2363, .1333]
    for i, weight in enumerate(weights):
        ssim, cs = reference_maps(a, b)
        expected *= max(ssim if i == 4 else cs, 0)**weight
        if i < 4:
            h, w = a.shape
            a = a[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            b = b[:h//2*2, :w//2*2].reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
    assert ms_ssim(x, y) == pytest.approx(expected, abs=1e-12)


def create_pair(root: Path, image_id="001", ir_size=(32, 32), vis_size=(32, 32)):
    for folder, size in [("Ir", ir_size), ("Vis", vis_size)]:
        (root / folder).mkdir(exist_ok=True)
        Image.new("RGB", size).save(root / folder / f"{image_id}.png")


def test_pair_audit_rejects_missing_and_wrong_counts(tmp_path):
    create_pair(tmp_path)
    with pytest.raises(ValueError, match="Expected 300"):
        audit_pairs(tmp_path)
    assert len(audit_pairs(tmp_path, expected_count=1)) == 1
    (tmp_path / "Ir/001.png").rename(tmp_path / "Ir/002.png")
    with pytest.raises(ValueError, match="Unmatched IDs"):
        audit_pairs(tmp_path, expected_count=1)


def test_pair_audit_rejects_shape_and_duplicate_ids(tmp_path):
    create_pair(tmp_path, vis_size=(31, 32))
    with pytest.raises(ValueError, match="Shape mismatch"):
        audit_pairs(tmp_path, expected_count=1)
    Image.new("RGB", (32, 32)).save(tmp_path / "Ir/001.jpg")
    with pytest.raises(ValueError, match="Duplicate image ID"):
        index_images(tmp_path / "Ir")


def test_dark_float_entropy_uses_fixed_8bit_domain():
    image = np.tile([0., .25, .5, .75], (16, 16))
    assert entropy(image) == 0.0
    assert entropy(np.array([[0., 1.]])) == 1.0


@pytest.mark.parametrize("bad", [np.full((16, 16), np.nan), np.full((16, 16), np.inf),
                                  np.full((16, 16), -1.), np.full((16, 16), 256.),
                                  np.empty((0, 0))])
def test_invalid_pixels_cannot_produce_valid_metric_record(bad):
    evaluator = FusionEvaluator(EvaluationConfig(metrics=["EN", "SD", "SCD", "MS_SSIM"]))
    result = evaluator.evaluate_arrays(bad, bad, bad)
    assert not result["valid"]
    assert result["error_reason"]


def test_cli_refuses_silent_partial_evaluation(tmp_path):
    create_pair(tmp_path)
    fused = tmp_path / "fused"
    fused.mkdir()
    with pytest.raises(ValueError, match="ID sets differ"):
        build_triplets(None, fused, tmp_path / "Ir", tmp_path / "Vis")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id\n001\n")
    with pytest.raises(ValueError, match="Missing fused"):
        build_triplets(manifest, fused, tmp_path / "Ir", tmp_path / "Vis")
    with pytest.raises(FileNotFoundError):
        build_triplets(tmp_path / "absent.csv", fused, tmp_path / "Ir", tmp_path / "Vis")
