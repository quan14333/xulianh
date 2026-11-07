"""Evaluation engine and contract-compliant reporting for IVIF.

Produces artifacts conforming to docs/OUTPUT_CONTRACT.md:
- metrics/per_image.csv
- metrics/summary.json
- metrics/paired_comparison.csv
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.evaluation.image_io import load_image, validate_matching_shapes
from src.evaluation.metrics import (
    average_gradient,
    edge_correlation,
    entropy,
    fusion_ms_ssim,
    q_abf,
    spatial_frequency,
    standard_deviation,
    sum_of_correlation_differences,
)


@dataclass
class EvaluationConfig:
    """Evaluation configuration parsed from configs/evaluation/fusion_metrics.yaml."""

    metrics: list[str] = field(
        default_factory=lambda: ["EN", "SD", "SCD", "MS_SSIM"]
    )
    input_channel: str = "Y"
    require_matching_ids: bool = True
    require_matching_shapes: bool = True
    cast_before_subtraction: str = "float64"
    statistics: list[str] = field(
        default_factory=lambda: ["mean", "std", "median", "valid_count"]
    )
    paired_comparison: bool = True
    ms_ssim_aggregation: str = "mean"
    data_range: float = 255.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config YAML not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        eval_dict = raw.get("evaluation", {})
        rep_dict = raw.get("reporting", {})

        metrics = eval_dict.get("metrics", ["EN", "SD", "SCD", "MS_SSIM"])
        input_channel = eval_dict.get("input_channel", "Y")
        if "REPLACE" in str(input_channel):
            input_channel = "Y"

        return cls(
            metrics=metrics,
            input_channel=str(input_channel),
            require_matching_ids=bool(eval_dict.get("require_matching_ids", True)),
            require_matching_shapes=bool(eval_dict.get("require_matching_shapes", True)),
            cast_before_subtraction=str(eval_dict.get("cast_before_subtraction", "float64")),
            statistics=rep_dict.get("statistics", ["mean", "std", "median", "valid_count"]),
            paired_comparison=bool(rep_dict.get("paired_comparison", True)),
            ms_ssim_aggregation=str(eval_dict.get("ms_ssim_aggregation", "mean")),
            data_range=float(eval_dict.get("data_range", 255.0)),
        )


class FusionEvaluator:
    """Evaluates fused images against infrared and visible sources."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.cast_dtype = getattr(np, self.config.cast_before_subtraction, np.float64)

    def evaluate_arrays(
        self,
        fused: np.ndarray,
        ir: np.ndarray,
        vis: np.ndarray,
        image_id: str = "",
    ) -> dict[str, Any]:
        """Evaluate a single preprocessed image triplet in-memory.

        Returns a dictionary with metric values, 'valid', and 'error_reason'.
        """
        record: dict[str, Any] = {
            "image_id": image_id,
            "valid": True,
            "error_reason": "",
        }

        try:
            if self.config.require_matching_shapes:
                validate_matching_shapes(fused, ir, vis, image_id=image_id)

            if "EN" in self.config.metrics:
                record["EN"] = entropy(fused)

            if "SD" in self.config.metrics:
                record["SD"] = standard_deviation(fused, ddof=0)

            if "SCD" in self.config.metrics:
                record["SCD"] = sum_of_correlation_differences(fused, ir, vis)

            if "MS_SSIM" in self.config.metrics or "MS-SSIM" in self.config.metrics:
                ms_results = fusion_ms_ssim(
                    fused,
                    ir,
                    vis,
                    data_range=self.config.data_range,
                    aggregation=self.config.ms_ssim_aggregation,
                )
                record["MS_SSIM"] = ms_results["MS_SSIM"]
                record["MS_SSIM_IR"] = ms_results["MS_SSIM_IR"]
                record["MS_SSIM_VIS"] = ms_results["MS_SSIM_VIS"]
                record["MS_SSIM_mean"] = ms_results["MS_SSIM_mean"]
                record["MS_SSIM_sum"] = ms_results["MS_SSIM_sum"]

            if "Q_ABF" in self.config.metrics or "QABF" in self.config.metrics:
                record["Q_ABF"] = q_abf(fused, ir, vis)

            if "SF" in self.config.metrics:
                record["SF"] = spatial_frequency(fused)

            if "AG" in self.config.metrics:
                record["AG"] = average_gradient(fused)

            if any(m in self.config.metrics for m in ("edge_corr", "edge_corr_IR", "edge_corr_VIS")):
                record["edge_corr_IR"] = edge_correlation(fused, ir)
                record["edge_corr_VIS"] = edge_correlation(fused, vis)

        except Exception as exc:
            record["valid"] = False
            record["error_reason"] = f"{type(exc).__name__}: {str(exc)}"
            for m in (
                "EN",
                "SD",
                "SCD",
                "MS_SSIM",
                "MS_SSIM_IR",
                "MS_SSIM_VIS",
                "MS_SSIM_mean",
                "MS_SSIM_sum",
                "Q_ABF",
                "SF",
                "AG",
                "edge_corr_IR",
                "edge_corr_VIS",
            ):
                if m not in record:
                    record[m] = float("nan")

        return record

    def evaluate_files(
        self,
        fused_path: Path | str,
        ir_path: Path | str,
        vis_path: Path | str,
        image_id: str,
    ) -> dict[str, Any]:
        """Load image triplet from disk and evaluate metrics."""
        try:
            fused_arr = load_image(
                fused_path,
                target_channel=self.config.input_channel,
                cast_dtype=self.cast_dtype,
            )
            ir_arr = load_image(
                ir_path,
                target_channel=self.config.input_channel,
                cast_dtype=self.cast_dtype,
            )
            vis_arr = load_image(
                vis_path,
                target_channel=self.config.input_channel,
                cast_dtype=self.cast_dtype,
            )
            return self.evaluate_arrays(fused_arr, ir_arr, vis_arr, image_id=image_id)
        except Exception as exc:
            return {
                "image_id": image_id,
                "EN": float("nan"),
                "SD": float("nan"),
                "SCD": float("nan"),
                "MS_SSIM": float("nan"),
                "MS_SSIM_IR": float("nan"),
                "MS_SSIM_VIS": float("nan"),
                "MS_SSIM_mean": float("nan"),
                "MS_SSIM_sum": float("nan"),
                "valid": False,
                "error_reason": f"{type(exc).__name__}: {str(exc)}",
            }

    def evaluate_dataset(
        self,
        triplets: Sequence[Mapping[str, Any]],
        output_dir: Path | str | None = None,
        run_id: str = "evaluation_run",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate a sequence of image triplets.

        Each element of `triplets` is a dict with keys:
            - 'image_id': str
            - 'fused_path': str | Path
            - 'ir_path': str | Path
            - 'vis_path': str | Path

        Returns:
            (per_image_records, summary_statistics)
        """
        records: list[dict[str, Any]] = []

        for item in triplets:
            rec = self.evaluate_files(
                fused_path=item["fused_path"],
                ir_path=item["ir_path"],
                vis_path=item["vis_path"],
                image_id=str(item["image_id"]),
            )
            records.append(rec)

        summary = self.compute_summary(records, run_id=run_id)

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            self.save_per_image_csv(records, out_path / "per_image.csv")
            self.save_summary_json(summary, out_path / "summary.json")

        return records, summary

    def compute_summary(
        self,
        records: Sequence[dict[str, Any]],
        run_id: str = "evaluation_run",
    ) -> dict[str, Any]:
        """Compute mean, std, median, and valid_count for metrics across valid records."""
        total_count = len(records)
        valid_records = [r for r in records if r.get("valid", False)]
        valid_count = len(valid_records)

        metric_names = [
            "EN",
            "SD",
            "SCD",
            "MS_SSIM",
            "MS_SSIM_IR",
            "MS_SSIM_VIS",
            "MS_SSIM_mean",
            "MS_SSIM_sum",
            "Q_ABF",
            "SF",
            "AG",
            "edge_corr_IR",
            "edge_corr_VIS",
        ]

        summary_metrics: dict[str, dict[str, float | int]] = {}

        for m in metric_names:
            vals = [
                float(r[m])
                for r in valid_records
                if m in r and not math.isnan(float(r[m]))
            ]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                summary_metrics[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=0)),
                    "median": float(np.median(arr)),
                    "valid_count": len(vals),
                }
            else:
                summary_metrics[m] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "valid_count": 0,
                }

        return {
            "run_id": run_id,
            "total_images": total_count,
            "valid_images": valid_count,
            "invalid_images": total_count - valid_count,
            "input_channel": self.config.input_channel,
            "ms_ssim_aggregation": self.config.ms_ssim_aggregation,
            "metrics": summary_metrics,
        }

    @staticmethod
    def save_per_image_csv(
        records: Sequence[dict[str, Any]],
        file_path: Path | str,
    ) -> None:
        """Write per-image metrics to CSV according to docs/OUTPUT_CONTRACT.md."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "image_id",
            "EN",
            "SD",
            "SCD",
            "MS_SSIM",
            "MS_SSIM_IR",
            "MS_SSIM_VIS",
            "MS_SSIM_mean",
            "MS_SSIM_sum",
            "Q_ABF",
            "SF",
            "AG",
            "edge_corr_IR",
            "edge_corr_VIS",
            "valid",
            "error_reason",
        ]

        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    @staticmethod
    def save_summary_json(
        summary: dict[str, Any],
        file_path: Path | str,
    ) -> None:
        """Write summary statistics to JSON according to docs/OUTPUT_CONTRACT.md."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_paired_comparison(
        baseline_csv_path: Path | str,
        candidate_csv_path: Path | str,
        output_csv_path: Path | str,
    ) -> list[dict[str, Any]]:
        """Compute per-image delta = candidate - baseline for each metric.

        Saves results to metrics/paired_comparison.csv conforming to OUTPUT_CONTRACT.
        """
        base_records = {}
        with Path(baseline_csv_path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                base_records[row["image_id"]] = row

        cand_records = {}
        with Path(candidate_csv_path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cand_records[row["image_id"]] = row

        common_ids = sorted(set(base_records.keys()) & set(cand_records.keys()))
        metrics_to_compare = [
            "EN",
            "SD",
            "SCD",
            "MS_SSIM",
            "MS_SSIM_IR",
            "MS_SSIM_VIS",
            "MS_SSIM_sum",
        ]

        paired_rows: list[dict[str, Any]] = []
        for img_id in common_ids:
            b_row = base_records[img_id]
            c_row = cand_records[img_id]

            row_data: dict[str, Any] = {"image_id": img_id}
            is_valid = (
                b_row.get("valid", "True").lower() == "true"
                and c_row.get("valid", "True").lower() == "true"
            )
            row_data["valid"] = is_valid

            for m in metrics_to_compare:
                if m in b_row and m in c_row:
                    try:
                        b_val = float(b_row[m])
                        c_val = float(c_row[m])
                        delta = c_val - b_val
                        row_data[f"baseline_{m}"] = b_val
                        row_data[f"candidate_{m}"] = c_val
                        row_data[f"delta_{m}"] = delta
                    except (ValueError, TypeError):
                        row_data[f"baseline_{m}"] = float("nan")
                        row_data[f"candidate_{m}"] = float("nan")
                        row_data[f"delta_{m}"] = float("nan")

            paired_rows.append(row_data)

        # Write CSV
        target = Path(output_csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if paired_rows:
            fieldnames = list(paired_rows[0].keys())
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for pr in paired_rows:
                    writer.writerow(pr)

        return paired_rows
