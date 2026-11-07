"""Unit tests for FusionEvaluator, image I/O, and OUTPUT_CONTRACT schemas."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator
from src.evaluation.image_io import extract_channel, load_image, rgb_to_y, validate_matching_shapes


def test_rgb_to_y_standard_bt601() -> None:
    """Verify standard ITU-R BT.601 conversion on primary colors."""
    # Pure Red: Y = 0.299 * 255 = 76.245
    pure_red = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_red[:, :, 0] = 255
    y_red = rgb_to_y(pure_red)
    assert np.allclose(y_red, 76.245, atol=1e-5)

    # Pure Green: Y = 0.587 * 255 = 149.685
    pure_green = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_green[:, :, 1] = 255
    y_green = rgb_to_y(pure_green)
    assert np.allclose(y_green, 149.685, atol=1e-5)

    # Pure White: Y = 255.0
    pure_white = np.full((2, 2, 3), 255, dtype=np.uint8)
    y_white = rgb_to_y(pure_white)
    assert np.allclose(y_white, 255.0, atol=1e-5)


def test_extract_channel_modes() -> None:
    """Test channel extraction for 2D, 3D, and uppercase/lowercase mode strings."""
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 100

    y_extracted = extract_channel(rgb, channel="Y")
    assert y_extracted.shape == (10, 10)
    assert np.allclose(y_extracted, 29.9, atol=1e-5)

    # 2D pass-through
    gray = np.full((10, 10), 50, dtype=np.uint8)
    assert np.allclose(extract_channel(gray, channel="Y"), 50.0)


def test_evaluation_config_and_load_image(tmp_path: Path) -> None:
    """Test EvaluationConfig from YAML and load_image helper."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        "schema_version: 1\n"
        "evaluation:\n"
        "  metrics: [EN, SD]\n"
        "  input_channel: Y\n"
        "  require_matching_ids: true\n"
        "  require_matching_shapes: true\n"
        "  cast_before_subtraction: float64\n"
        "reporting:\n"
        "  statistics: [mean, std]\n",
        encoding="utf-8",
    )
    cfg = EvaluationConfig.from_yaml(config_file)
    assert cfg.metrics == ["EN", "SD"]
    assert cfg.input_channel == "Y"

    img_file = tmp_path / "sample.png"
    Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8)).save(img_file)
    loaded = load_image(img_file, target_channel="Y")
    assert loaded.shape == (32, 32)
    assert np.allclose(loaded, 100.0)


def test_validate_matching_shapes() -> None:
    """Check shape validator passes matching shapes and rejects mismatches."""
    a = np.zeros((100, 80))
    b = np.zeros((100, 80))
    c = np.zeros((100, 80))
    # Should not raise
    validate_matching_shapes(a, b, c, image_id="test_01")

    # Mismatch
    d = np.zeros((100, 81))
    with pytest.raises(ValueError, match="Image shape mismatch"):
        validate_matching_shapes(a, b, d, image_id="test_02")


def test_evaluator_arrays_contract() -> None:
    """Check single evaluate_arrays output dictionary keys."""
    evaluator = FusionEvaluator()
    rng = np.random.default_rng(42)
    ir = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    vis = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    fused = ((ir.astype(float) + vis.astype(float)) / 2).astype(np.uint8)

    rec = evaluator.evaluate_arrays(fused, ir, vis, image_id="sample_001")
    assert rec["image_id"] == "sample_001"
    assert rec["valid"] is True
    assert rec["error_reason"] == ""
    for key in ("EN", "SD", "SCD", "MS_SSIM", "MS_SSIM_IR", "MS_SSIM_VIS", "MS_SSIM_sum"):
        assert key in rec
        assert not np.isnan(rec[key])


def test_evaluator_dataset_output_contract(tmp_path: Path) -> None:
    """End-to-end test generating per_image.csv, summary.json according to OUTPUT_CONTRACT."""
    # Create sample synthetic images
    ir_dir = tmp_path / "Infrared"
    vis_dir = tmp_path / "Visible"
    fused_dir = tmp_path / "fused"
    out_dir = tmp_path / "metrics"

    ir_dir.mkdir()
    vis_dir.mkdir()
    fused_dir.mkdir()

    rng = np.random.default_rng(100)
    triplets = []

    for i in range(3):
        stem = f"test_{i:03d}"
        ir_arr = rng.integers(0, 256, (40, 40), dtype=np.uint8)
        vis_arr = rng.integers(0, 256, (40, 40), dtype=np.uint8)
        fused_arr = ((ir_arr.astype(float) + vis_arr.astype(float)) / 2).astype(np.uint8)

        ir_p = ir_dir / f"{stem}.png"
        vis_p = vis_dir / f"{stem}.png"
        fused_p = fused_dir / f"{stem}.png"

        Image.fromarray(ir_arr).save(ir_p)
        Image.fromarray(vis_arr).save(vis_p)
        Image.fromarray(fused_arr).save(fused_p)

        triplets.append(
            {
                "image_id": stem,
                "fused_path": fused_p,
                "ir_path": ir_p,
                "vis_path": vis_p,
            }
        )

    evaluator = FusionEvaluator()
    records, summary = evaluator.evaluate_dataset(
        triplets,
        output_dir=out_dir,
        run_id="test_run_01",
    )

    assert len(records) == 3
    assert summary["valid_images"] == 3
    assert summary["total_images"] == 3

    # Check generated files
    per_image_csv = out_dir / "per_image.csv"
    summary_json = out_dir / "summary.json"
    assert per_image_csv.is_file()
    assert summary_json.is_file()

    # Verify per_image schema
    with per_image_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert "image_id" in rows[0]
        assert "EN" in rows[0]
        assert "SD" in rows[0]
        assert "SCD" in rows[0]
        assert "MS_SSIM" in rows[0]
        assert "valid" in rows[0]

    # Verify summary.json schema
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["run_id"] == "test_run_01"
        assert "metrics" in data
        assert "EN" in data["metrics"]
        assert "mean" in data["metrics"]["EN"]
        assert "std" in data["metrics"]["EN"]
        assert "median" in data["metrics"]["EN"]
        assert "valid_count" in data["metrics"]["EN"]


def test_paired_comparison_generation(tmp_path: Path) -> None:
    """Verify paired comparison generation and delta calculation."""
    base_csv = tmp_path / "base_per_image.csv"
    cand_csv = tmp_path / "cand_per_image.csv"
    out_csv = tmp_path / "paired_comparison.csv"

    # Base rows
    base_rows = [
        {"image_id": "img1", "EN": "6.0", "SD": "30.0", "SCD": "1.5", "MS_SSIM": "0.75", "valid": "True"},
        {"image_id": "img2", "EN": "5.5", "SD": "25.0", "SCD": "1.4", "MS_SSIM": "0.70", "valid": "True"},
    ]
    # Candidate rows: higher EN (+0.5), lower SD (-2.0)
    cand_rows = [
        {"image_id": "img1", "EN": "6.5", "SD": "28.0", "SCD": "1.6", "MS_SSIM": "0.80", "valid": "True"},
        {"image_id": "img2", "EN": "5.8", "SD": "26.0", "SCD": "1.45", "MS_SSIM": "0.72", "valid": "True"},
    ]

    fieldnames = ["image_id", "EN", "SD", "SCD", "MS_SSIM", "valid"]
    with base_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)

    with cand_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cand_rows)

    paired = FusionEvaluator.compute_paired_comparison(base_csv, cand_csv, out_csv)
    assert len(paired) == 2
    assert out_csv.is_file()

    row1 = paired[0]
    assert row1["image_id"] == "img1"
    assert math.isclose(row1["delta_EN"], 0.5, abs_tol=1e-5)
    assert math.isclose(row1["delta_SD"], -2.0, abs_tol=1e-5)
    assert math.isclose(row1["delta_SCD"], 0.1, abs_tol=1e-5)
    assert math.isclose(row1["delta_MS_SSIM"], 0.05, abs_tol=1e-5)


def test_evaluator_extended_metrics_contract() -> None:
    """Verify that evaluator computes extended metrics (Q_ABF, SF, AG, edge_corr)."""
    cfg = EvaluationConfig(
        metrics=["EN", "SD", "SCD", "MS_SSIM", "Q_ABF", "SF", "AG", "edge_corr"]
    )
    evaluator = FusionEvaluator(config=cfg)

    rng = np.random.default_rng(333)
    ir = rng.integers(0, 256, (60, 60), dtype=np.uint8)
    vis = rng.integers(0, 256, (60, 60), dtype=np.uint8)
    fused = ((ir.astype(float) + vis.astype(float)) / 2).astype(np.uint8)

    rec = evaluator.evaluate_arrays(fused, ir, vis, image_id="test_ext_01")
    assert rec["valid"] is True
    for key in ("Q_ABF", "SF", "AG", "edge_corr_IR", "edge_corr_VIS"):
        assert key in rec
        assert not math.isnan(rec[key])


def test_loc_be_summary_parity() -> None:
    """Verify that Loc be's reported summary table aligns with Loc lon's parity table."""
    root = Path(__file__).resolve().parents[1]
    loc_be_summary = root / "reports" / "tables" / "loc_be" / "pcdfusion_summary.csv"
    loc_lon_parity = root / "reports" / "tables" / "loc_lon" / "loc_be_parity_verification.csv"

    assert loc_be_summary.is_file(), f"Missing {loc_be_summary}"
    assert loc_lon_parity.is_file(), f"Missing {loc_lon_parity}"

    # Read Loc be summary into a lookup map
    loc_be_data = {}
    with loc_be_summary.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc_be_data[(row["method"], row["metric"])] = float(row["mean"])

    # Verify every entry in loc_lon_parity matches loc_be_summary
    with loc_lon_parity.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["method"], row["metric"])
            assert key in loc_be_data, f"Key {key} not found in Loc be summary"
            reported = float(row["reported_mean"])
            actual = loc_be_data[key]
            assert math.isclose(reported, actual, rel_tol=1e-4)

