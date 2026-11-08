"""CLI entrypoint to evaluate fused images against infrared and visible modalities.

Usage:
    python scripts/run_evaluation.py \\
        --config configs/evaluation/fusion_metrics.yaml \\
        --manifest data/manifests/test.csv \\
        --fused-dir runs/sage_baseline/fused \\
        --ir-dir data/raw/FMB/test/Infrared \\
        --vis-dir data/raw/FMB/test/Visible \\
        --output-dir runs/sage_baseline/metrics \\
        --run-id sage_fmb_baseline_metrics
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator  # noqa: E402

# Dataset presets to auto-link real raw data directories without manual typing
DATASET_PRESETS: dict[str, dict[str, Path | None]] = {
    "fmb_test": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "FMB" / "test" / "Infrared",
        "vis_dir": REPO_ROOT / "data" / "raw" / "FMB" / "test" / "Visible",
        "manifest": REPO_ROOT / "data" / "manifests" / "test.csv",
    },
    "fmb_train": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "FMB" / "train" / "Infrared",
        "vis_dir": REPO_ROOT / "data" / "raw" / "FMB" / "train" / "Visible",
        "manifest": REPO_ROOT / "data" / "manifests" / "train.csv",
    },
    "fmb_dev": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "FMB" / "train" / "Infrared",
        "vis_dir": REPO_ROOT / "data" / "raw" / "FMB" / "train" / "Visible",
        "manifest": REPO_ROOT / "data" / "manifests" / "dev.csv",
    },
    "m3fd": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "M3FD_Fusion" / "Ir",
        "vis_dir": REPO_ROOT / "data" / "raw" / "M3FD_Fusion" / "Vis",
        "manifest": None,
    },
    "roadscene": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "roadscene" / "ir",
        "vis_dir": REPO_ROOT / "data" / "raw" / "roadscene" / "vi",
        "manifest": None,
    },
    "tno": {
        "ir_dir": REPO_ROOT / "data" / "raw" / "tno" / "ir",
        "vis_dir": REPO_ROOT / "data" / "raw" / "tno" / "vi",
        "manifest": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IVIF metrics evaluation on image triplets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "evaluation" / "fusion_metrics.yaml",
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fmb_test",
        choices=["fmb_test", "fmb_train", "fmb_dev", "m3fd", "roadscene", "tno", "custom"],
        help="Preset dataset to auto-link real raw IR/VIS directories (default: fmb_test).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to manifest CSV containing image_id column (defaults to dataset manifest).",
    )
    parser.add_argument(
        "--fused-dir",
        type=Path,
        required=True,
        help="Directory containing fused images.",
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        default=None,
        help="Directory containing infrared images (defaults to dataset raw IR folder).",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="Directory containing visible images (defaults to dataset raw VIS folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save per_image.csv, summary.json, etc.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="evaluation_run",
        help="Unique identifier for this evaluation run.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Optional baseline per_image.csv to generate paired_comparison.csv.",
    )
    return parser.parse_args()


def build_triplets(
    manifest_path: Path | None,
    fused_dir: Path,
    ir_dir: Path,
    vis_dir: Path,
) -> list[dict[str, str | Path]]:
    triplets = []

    if manifest_path is not None and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id") or row.get("id") or row.get("name")
                if not img_id:
                    continue
                # Remove extension if provided
                stem = Path(img_id).stem
                # Find matching files with common extensions
                fused_candidates = list(fused_dir.glob(f"{stem}.*"))

                # Check direct or glob matching for IR and VIS
                ir_file: Path | None = None
                if (ir_dir / f"{stem}.png").is_file():
                    ir_file = ir_dir / f"{stem}.png"
                else:
                    ir_candidates = list(ir_dir.glob(f"{stem}.*"))
                    if ir_candidates:
                        ir_file = ir_candidates[0]

                vis_file: Path | None = None
                if (vis_dir / f"{stem}.png").is_file():
                    vis_file = vis_dir / f"{stem}.png"
                else:
                    vis_candidates = list(vis_dir.glob(f"{stem}.*"))
                    if vis_candidates:
                        vis_file = vis_candidates[0]

                if fused_candidates and ir_file is not None and vis_file is not None:
                    triplets.append(
                        {
                            "image_id": stem,
                            "fused_path": fused_candidates[0],
                            "ir_path": ir_file,
                            "vis_path": vis_file,
                        }
                    )
    else:
        # Scan fused_dir directly
        for fused_file in sorted(fused_dir.iterdir()):
            if fused_file.is_file() and fused_file.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
            }:
                stem = fused_file.stem
                ir_candidates = list(ir_dir.glob(f"{stem}.*"))
                vis_candidates = list(vis_dir.glob(f"{stem}.*"))

                if ir_candidates and vis_candidates:
                    triplets.append(
                        {
                            "image_id": stem,
                            "fused_path": fused_file,
                            "ir_path": ir_candidates[0],
                            "vis_path": vis_candidates[0],
                        }
                    )

    return triplets


def main() -> int:
    args = parse_args()
    config = EvaluationConfig.from_yaml(args.config)
    evaluator = FusionEvaluator(config=config)

    # Auto-resolve dataset IR, VIS, and manifest paths from preset if not provided
    dataset_key = str(args.dataset).lower()
    preset = DATASET_PRESETS.get(dataset_key, {})
    ir_dir = args.ir_dir or preset.get("ir_dir")
    vis_dir = args.vis_dir or preset.get("vis_dir")
    manifest_path = args.manifest or preset.get("manifest")

    if ir_dir is None or not Path(ir_dir).is_dir():
        print(f"Error: Infrared directory not found: {ir_dir}")
        return 1
    if vis_dir is None or not Path(vis_dir).is_dir():
        print(f"Error: Visible directory not found: {vis_dir}")
        return 1

    print(f"Linking to dataset [{dataset_key}]:")
    print(f"  - IR directory : {ir_dir}")
    print(f"  - VIS directory: {vis_dir}")
    if manifest_path is not None:
        print(f"  - Manifest CSV : {manifest_path}")

    triplets = build_triplets(
        manifest_path,
        args.fused_dir,
        Path(ir_dir),
        Path(vis_dir),
    )

    if not triplets:
        print(f"Error: No matching image triplets found between {args.fused_dir}, {args.ir_dir}, {args.vis_dir}")
        return 1

    print(f"Found {len(triplets)} image triplets. Evaluating with channel={config.input_channel}...")
    records, summary = evaluator.evaluate_dataset(
        triplets,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    print(f"Completed evaluation for {summary['valid_images']}/{summary['total_images']} images.")
    print("\n--- Summary Metrics (Raw) ---")
    for m, stat in summary["metrics"].items():
        if m.endswith("_scaled"):
            continue
        m_val = f"{stat['mean']:.4f}" if not math.isnan(stat['mean']) else "NaN"
        s_val = f"{stat['std']:.4f}" if not math.isnan(stat['std']) else "NaN"
        med_val = f"{stat['median']:.4f}" if not math.isnan(stat['median']) else "NaN"
        print(f"  - {m:15s}: mean={m_val:>8s}, std={s_val:>8s}, median={med_val:>8s}")

    if "paper_benchmark" in summary and "metrics" in summary["paper_benchmark"]:
        print("\n--- SAGE CVPR 2025 Figure 5 Benchmark (Scaled) ---")
        for label, val in summary["paper_benchmark"]["metrics"].items():
            val_str = f"{val:.4f}" if not math.isnan(val) else "N/A"
            print(f"  * {label:15s}: {val_str:>8s}")

    if args.baseline_csv is not None and args.baseline_csv.is_file():
        paired_out = args.output_dir / "paired_comparison.csv"
        evaluator.compute_paired_comparison(
            baseline_csv_path=args.baseline_csv,
            candidate_csv_path=args.output_dir / "per_image.csv",
            output_csv_path=paired_out,
        )
        print(f"Paired comparison written to: {paired_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
