"""Replay the manual, post-hoc SAGE/PCDFusion ROI audit on external PNGs.

These nine rectangles are exploratory annotations, not semantic ground truth.
OpenCV is optional and imported only when reading the external images; importing
this module and testing the numerical functions requires only NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


SOURCE_RUN_ID = "sage_pcdfusion_08nov_20261106_170517"
IDS = ("00014", "00024", "00061", "00064", "00071", "00075", "00104", "00109", "00112", "00145")
IMAGE_SHAPE = (512, 640)
# image_id, roi_id, manual description, x1, y1, x2, y2; half-open rectangles.
ROIS = (
    ("00064", "R1", "Người đi bộ gần xe phía phải", 270, 291, 321, 376),
    ("00064", "R2", "Nhóm người đi bộ ở xa", 186, 307, 226, 355),
    ("00104", "R3", "Hai người đi bộ gần camera bên trái", 33, 303, 77, 362),
    ("00104", "R4", "Nhóm người đi bộ gần bên trái", 122, 308, 173, 365),
    ("00109", "R5", "Người đi bộ gần mép trái", 26, 266, 65, 334),
    ("00109", "R6", "Nhóm người đi bộ ở xa", 104, 285, 157, 330),
    ("00112", "R7", "Người đi bộ ở giữa đường", 175, 281, 197, 328),
    ("00112", "R8", "Hai người ở xa phía trái", 49, 285, 75, 320),
    ("00145", "R9", "Xe máy gần camera", 125, 328, 227, 467),
)
RINGS = (12, 24, 36)
AGGREGATES = ("mean", "median")


def masks_for(
    image_id: str,
    box: Sequence[int],
    pad: int,
    shape: tuple[int, int] = IMAGE_SHAPE,
    rois: Sequence[tuple] = ROIS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectangle and clipped ring, excluding every ROI in the same scene."""
    h, w = shape
    if h <= 0 or w <= 0 or pad <= 0 or int(pad) != pad:
        raise ValueError("Image dimensions and integer ring radius must be positive")
    x1, y1, x2, y2 = box
    if any(int(v) != v for v in box) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"ROI must be an integer, nonempty box inside {shape}: {box}")
    roi = np.zeros(shape, dtype=bool)
    roi[y1:y2, x1:x2] = True
    background = np.zeros(shape, dtype=bool)
    background[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)] = True
    background &= ~roi
    for sid, _rid, _description, xx1, yy1, xx2, yy2 in rois:
        if sid == image_id:
            background[yy1:yy2, xx1:xx2] = False
    if not background.any():
        raise ValueError("ROI has no background pixels after clipping/exclusions")
    return roi, background


def contrast(image: np.ndarray, roi: np.ndarray, background: np.ndarray, mode: str) -> float:
    """Signed ROI minus ring statistic, in floating point to avoid uint8 overflow."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.shape != roi.shape or image.shape != background.shape:
        raise ValueError("Image and masks must have the same two-dimensional shape")
    if roi.dtype != bool or background.dtype != bool:
        raise ValueError("ROI and background must be boolean masks")
    if not roi.any() or not background.any() or np.any(roi & background):
        raise ValueError("Masks must be nonempty and disjoint")
    values, reference = image[roi], image[background]
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("ROI/background contain nonfinite values")
    if mode == "mean":
        return float(values.mean() - reference.mean())
    if mode == "median":
        return float(np.median(values) - np.median(reference))
    raise ValueError(f"Unknown aggregation: {mode}")


def source_paths(sage_root: Path, pcdfusion_root: Path, sid: str) -> dict[str, Path]:
    return {
        "IR": pcdfusion_root / "dataset/FMB/test/ir" / f"{sid}.png",
        "VIS": pcdfusion_root / "dataset/FMB/test/vi" / f"{sid}.png",
        "SAGE": sage_root / "fused_png" / f"{sid}.png",
        "PCDFusion": pcdfusion_root / "fused/PCDFusion" / f"{sid}.png",
    }


def load_sources(sage_root: Path, pcdfusion_root: Path) -> tuple[dict, list[dict]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "PNG replay requires: python -m pip install opencv-python-headless"
        ) from exc

    sources, provenance = {}, []
    for sid in IDS:
        images = {}
        for label, path in source_paths(sage_root, pcdfusion_root, sid).items():
            # imdecode handles Unicode Windows paths as well as POSIX paths.
            payload = path.read_bytes()
            mode = cv2.IMREAD_GRAYSCALE if label == "IR" else cv2.IMREAD_COLOR
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), mode)
            if image is None or image.shape[:2] != IMAGE_SHAPE:
                raise ValueError(f"Unreadable PNG or expected shape {IMAGE_SHAPE} not met: {path}")
            if label != "IR":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            images[label] = image.astype(np.float64)
            root = sage_root if label == "SAGE" else pcdfusion_root
            provenance.append(
                {
                    "image_id": sid,
                    "method": label,
                    "root_argument": "sage-root" if label == "SAGE" else "pcdfusion-root",
                    "relative_file": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        sources[sid] = images
    return sources, provenance


def audit_rows(sources: dict) -> list[dict]:
    rows = []
    for sid, rid, description, x1, y1, x2, y2 in ROIS:
        for pad in RINGS:
            roi, background = masks_for(sid, (x1, y1, x2, y2), pad)
            for mode in AGGREGATES:
                values = {k: contrast(im, roi, background, mode) for k, im in sources[sid].items()}
                ir, vis, sage, pcd = (values[k] for k in ("IR", "VIS", "SAGE", "PCDFusion"))
                rows.append(
                    {
                        "image_id": sid,
                        "roi_id": rid,
                        "description": description,
                        "box_xyxy": f"{x1},{y1},{x2},{y2}",
                        "ring_radius_px": pad,
                        "aggregate": mode,
                        "roi_pixels": int(roi.sum()),
                        "background_pixels": int(background.sum()),
                        "IR_contrast": round(ir, 4),
                        "VIS_contrast": round(vis, 4),
                        "SAGE_contrast": round(sage, 4),
                        "PCDFusion_contrast": round(pcd, 4),
                        "IR_minus_VIS": round(ir - vis, 4),
                        "SAGE_minus_VIS": round(sage - vis, 4),
                        "PCDFusion_minus_VIS": round(pcd - vis, 4),
                        "SAGE_abs_IR_contrast_error": round(abs(sage - ir), 4),
                        "PCDFusion_abs_IR_contrast_error": round(abs(pcd - ir), 4),
                        "SAGE_vs_PCD_contrast_delta": round(sage - pcd, 4),
                        "IR_salient_vs_VIS": bool(ir >= 12 and ir - vis >= 12),
                        "SAGE_below_VIS": bool(sage < vis),
                        "PCDFusion_below_VIS": bool(pcd < vis),
                    }
                )
    return rows


def robustness_flags(rows: list[dict]) -> list[dict]:
    flags = []
    for sid, rid, description, *_box in ROIS:
        subset = [r for r in rows if r["roi_id"] == rid]
        main = next(r for r in subset if r["ring_radius_px"] == 24 and r["aggregate"] == "mean")
        salient = all(r["IR_salient_vs_VIS"] for r in subset)
        flags.append(
            {
                "image_id": sid,
                "roi_id": rid,
                "description": description,
                "IR_salient_all_6": salient,
                "SAGE_below_VIS_all_6": salient and all(r["SAGE_below_VIS"] for r in subset),
                "PCDFusion_below_VIS_all_6": salient
                and all(r["PCDFusion_below_VIS"] for r in subset),
                "SAGE_closer_to_IR_count_6": sum(
                    r["SAGE_abs_IR_contrast_error"] < r["PCDFusion_abs_IR_contrast_error"]
                    for r in subset
                ),
                "PCDFusion_closer_to_IR_count_6": sum(
                    r["SAGE_abs_IR_contrast_error"] > r["PCDFusion_abs_IR_contrast_error"]
                    for r in subset
                ),
                "SAGE_higher_contrast_count_6": sum(
                    r["SAGE_vs_PCD_contrast_delta"] > 0 for r in subset
                ),
                "PCDFusion_higher_contrast_count_6": sum(
                    r["SAGE_vs_PCD_contrast_delta"] < 0 for r in subset
                ),
                **{
                    f"primary_{k}_contrast": main[f"{k}_contrast"]
                    for k in ("IR", "VIS", "SAGE", "PCDFusion")
                },
            }
        )
    return flags


def summarize(rows: list[dict], flags: list[dict]) -> dict:
    primary = [r for r in rows if r["ring_radius_px"] == 24 and r["aggregate"] == "mean"]
    return {
        "source_run_id": SOURCE_RUN_ID,
        "source_images": len(IDS),
        "models": ["SAGE pretrained", "PCDFusion pretrained"],
        "roi_count": len(ROIS),
        "sensitivity_settings": {"ring_radius_px": list(RINGS), "aggregation": list(AGGREGATES)},
        "ir_salient_count_all6": sum(x["IR_salient_all_6"] for x in flags),
        "sage_below_visible_all6": sum(x["SAGE_below_VIS_all_6"] for x in flags),
        "pcdfusion_below_visible_all6": sum(x["PCDFusion_below_VIS_all_6"] for x in flags),
        "sage_closer_ir_primary": sum(
            r["SAGE_abs_IR_contrast_error"] < r["PCDFusion_abs_IR_contrast_error"] for r in primary
        ),
        "pcd_closer_ir_primary": sum(
            r["SAGE_abs_IR_contrast_error"] > r["PCDFusion_abs_IR_contrast_error"] for r in primary
        ),
        "sage_higher_contrast_primary": sum(r["SAGE_vs_PCD_contrast_delta"] > 0 for r in primary),
        "pcd_higher_contrast_primary": sum(r["SAGE_vs_PCD_contrast_delta"] < 0 for r in primary),
        "mean_contrast_by_source": {
            k: round(float(np.mean([r[k + "_contrast"] for r in primary])), 3)
            for k in ("IR", "VIS", "SAGE", "PCDFusion")
        },
        "caveats": [
            "Manual rectangles are exploratory, not pixel-level FMB semantic labels.",
            "ROIs are post-hoc on previously inspected test scenes; do not tune on these samples.",
            "Thermal grayscale is not calibrated physical temperature.",
            "Brightness contrast is not object recall or mIoU.",
            "10 repackaged FMB/test images are not the official 280-image benchmark.",
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(sage_root: Path, pcdfusion_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    sources, provenance = load_sources(sage_root, pcdfusion_root)
    rows = audit_rows(sources)
    flags = robustness_flags(rows)
    summary = summarize(rows, flags)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "roi_sensitivity_all.csv", rows)
    write_csv(
        output_dir / "roi_primary.csv",
        [r for r in rows if r["ring_radius_px"] == 24 and r["aggregate"] == "mean"],
    )
    write_csv(output_dir / "roi_robustness_flags.csv", flags)
    write_csv(output_dir / "file_provenance.csv", provenance)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage-root", required=True, type=Path, help="Extracted ZIP root containing fused_png/"
    )
    parser.add_argument(
        "--pcdfusion-root",
        required=True,
        type=Path,
        help="Extracted ZIP root containing dataset/ and fused/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory; must not already exist",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_audit(args.sage_root, args.pcdfusion_root, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ROI audit failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
