"""Check the committed exploratory records without downloading images or models."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from src.analysis.controlled_roi_audit import AGGREGATES, IDS, RINGS, ROIS


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables/loc_be"
METRICS = (
    "EN",
    "SD",
    "SCD",
    "MS_SSIM_sum",
    "edge_corr_IR",
    "edge_corr_VIS",
    "local_IR_retention",
    "local_IR_abs_error",
)
METHODS = {"SAGE", "PCDFusion", "Y_AVG", "Y_MAX"}


def read_csv(name):
    with (TABLES / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    # CSVs store nine decimal places; derived differences can accumulate rounding.
    assert math.isclose(float(actual), expected, rel_tol=0, abs_tol=2e-9)


def test_per_image_summary_and_paired_records_agree():
    rows = read_csv("pcdfusion_per_image.csv")
    by_key = {(r["image_id"], r["method"]): r for r in rows}
    assert len(rows) == len(by_key) == 40
    assert set(by_key) == {(sid, method) for sid in IDS for method in METHODS}
    assert all(r["valid"] == "True" and not r["error_reason"] for r in rows)
    for row in rows:
        assert all(math.isfinite(float(row[metric])) for metric in METRICS)
        close(row["MS_SSIM_sum"], float(row["MS_SSIM_IR"]) + float(row["MS_SSIM_VIS"]))

    summary = read_csv("pcdfusion_summary.csv")
    assert len(summary) == 32
    assert {(r["method"], r["metric"]) for r in summary} == {
        (method, metric) for method in METHODS for metric in METRICS
    }
    for row in summary:
        values = [float(by_key[sid, row["method"]][row["metric"]]) for sid in IDS]
        close(row["mean"], statistics.mean(values))
        close(row["std"], statistics.stdev(values))
        close(row["median"], statistics.median(values))
        assert int(row["count"]) == 10

    paired = read_csv("pcdfusion_paired_comparison.csv")
    assert len(paired) == 10 and {r["image_id"] for r in paired} == set(IDS)
    for row in paired:
        assert row["comparison"] == "PCDFusion_minus_SAGE"
        for metric in METRICS:
            sage = float(by_key[row["image_id"], "SAGE"][metric])
            pcd = float(by_key[row["image_id"], "PCDFusion"][metric])
            close(row["SAGE_" + metric], sage)
            close(row["PCDFusion_" + metric], pcd)
            close(row["delta_" + metric], pcd - sage)
    verified = read_csv("paired_summary_verified.csv")
    assert {r["metric"] for r in verified} == set(METRICS)
    for row in verified:
        metric = row["metric"]
        values = [float(p["delta_" + metric]) for p in paired]
        close(row["SAGE_mean"], statistics.mean(float(by_key[s, "SAGE"][metric]) for s in IDS))
        close(
            row["PCDFusion_mean"],
            statistics.mean(float(by_key[s, "PCDFusion"][metric]) for s in IDS),
        )
        close(row["PCD_minus_SAGE"], statistics.mean(values))
        close(row["paired_median_delta"], statistics.median(values))
        assert int(row["PCD_higher_count"]) == sum(v > 0 for v in values)
        assert int(row["PCD_lower_count"]) == sum(v < 0 for v in values)
        assert int(row["n_pairs"]) == 10
    delta_rows = read_csv("paired_key_deltas.csv")
    assert len(delta_rows) == 10
    for row in delta_rows:
        matching = next(p for p in paired if p["image_id"] == row["image_id"])
        assert all(
            row[k] == matching[k] or float(row[k]) == float(matching[k])
            for k in row
            if k != "image_id"
        )


def test_all_manual_roi_coordinates_settings_and_primary_rows_are_preserved():
    rows = read_csv("roi_sensitivity_all.csv")
    by_key = {(r["roi_id"], int(r["ring_radius_px"]), r["aggregate"]): r for r in rows}
    expected_keys = {
        (r[1], radius, aggregate) for r in ROIS for radius in RINGS for aggregate in AGGREGATES
    }
    assert len(rows) == len(by_key) == 54 and set(by_key) == expected_keys
    for sid, rid, description, *box in ROIS:
        for radius in RINGS:
            for aggregate in AGGREGATES:
                row = by_key[rid, radius, aggregate]
                assert row["image_id"] == sid and row["description"] == description
                assert row["box_xyxy"] == ",".join(map(str, box))
                assert int(row["roi_pixels"]) > 0 and int(row["background_pixels"]) > 0
    primary = read_csv("roi_primary.csv")
    assert len(primary) == 9
    assert primary == [r for r in rows if r["aggregate"] == "mean" and r["ring_radius_px"] == "24"]
    for radius, expected in [(12, 6), (24, 7), (36, 8)]:
        median = [
            r for r in rows if r["aggregate"] == "median" and int(r["ring_radius_px"]) == radius
        ]
        mean = [r for r in rows if r["aggregate"] == "mean" and int(r["ring_radius_px"]) == radius]
        assert sum(float(r["SAGE_vs_PCD_contrast_delta"]) > 0 for r in median) == expected
        assert sum(float(r["SAGE_vs_PCD_contrast_delta"]) > 0 for r in mean) == 9
    flags = read_csv("roi_robustness_flags.csv")
    assert len(flags) == 9 and sum(r["IR_salient_all_6"] == "True" for r in flags) == 7
    assert all(r["SAGE_below_VIS_all_6"] == "False" for r in flags)


def test_manifest_and_imported_source_checksums():
    manifest = read_csv("manifest.csv")
    assert len(manifest) == 10 and {r["image_id"] for r in manifest} == set(IDS)
    assert all(
        (r["split"], r["height"], r["width"]) == ("FMB/test_repackaged", "512", "640")
        for r in manifest
    )
    provenance = read_csv("file_provenance.csv")
    assert len(provenance) == 40
    assert {(r["image_id"], r["method"]) for r in provenance} == {
        (sid, method) for sid in IDS for method in ("IR", "VIS", "SAGE", "PCDFusion")
    }
    records = json.loads((TABLES / "source_provenance.json").read_text(encoding="utf-8"))
    for relative, record in records["files"].items():
        assert (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            == record["repository_sha256"]
        ), relative


def test_notebooks_have_parseable_code_and_no_saved_output_or_attachments():
    paths = sorted((ROOT / "notebooks/loc_be").glob("*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            assert not cell.get("attachments")
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and not cell["outputs"]
                ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
