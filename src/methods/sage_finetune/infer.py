"""B3-compatible full-resolution PNG export, with no Development Test access."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .checkpoint import atomic_json, environment, sha256, strict_load
from .config import authorize, load_config, new_output_directory
from .dataset import load_protocol, read_pair
from .model import fused_output, load_pretrained
from .train import configure_runtime, synchronize


def reconstruct_bgr(output, visible_ycrcb):
    """Exactly B3: clamp Y, torch.round uint8, retain VIS Cr/Cb, OpenCV to BGR."""
    if (output.dtype != torch.float32 or output.shape != (1, 1, *visible_ycrcb.shape[:2])
            or visible_ycrcb.dtype != np.uint8 or visible_ycrcb.ndim != 3
            or visible_ycrcb.shape[2] != 3):
        raise ValueError("Y/chroma shape/dtype mismatch")
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("Cannot save NaN/Inf output")
    if output.min().item() < -1.000001 or output.max().item() > 1.000001:
        raise ValueError("Raw Y exceeds Tanh range")
    y = output.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()[0, 0]
    merged = np.dstack([y, visible_ycrcb[:, :, 1:]])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def save_png(path, bgr):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite PNG: {path}")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR uint8 PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise IOError("OpenCV PNG encoding failed")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(decoded, bgr):
        raise IOError("Saved PNG did not round-trip exactly")
    return sha256(path)


def export(cfg, output, *, checkpoint=None, split="dev", limit=5, device=None):
    if split not in {"train", "dev"}:
        raise PermissionError("FMB Test export is disabled in this Development implementation")
    if limit is None or limit > 5:
        authorize(cfg, "train")  # Full export also requires a config-bound review record.
    if cfg["status"] != "APPROVED" and (limit is None or not 1 <= limit <= 5):
        raise PermissionError("PROVISIONAL export is capped at FIVE images")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    run_dir = new_output_directory(output, cfg)
    try:
        selected = configure_runtime(cfg, device)
        train_records, dev_records = load_protocol(cfg)
        records = train_records if split == "train" else dev_records
        records = records[:limit] if limit is not None else records
        network = load_pretrained(cfg)
        chosen = Path(checkpoint or cfg["model"]["pretrained"]).resolve()
        chosen_sha = sha256(chosen)
        metadata = strict_load(network, chosen)
        if metadata and metadata["provenance"]["pretrained_sha256"] != cfg["model"]["pretrained_sha256"]:
            raise ValueError("Fine-tuned checkpoint has a different pretrained origin")
        network.to(selected).eval()
        atomic_json(run_dir / "environment.json", environment(selected))
        atomic_json(run_dir / "config.json", cfg.snapshot())
        rows = []
        start = time.perf_counter()
        with torch.no_grad():
            for record in records:
                ycc, ir, hashes = read_pair(record)
                vis_t = torch.from_numpy(ycc[:, :, 0].copy())[None, None].float().div(255).to(selected)
                ir_t = torch.from_numpy(ir.copy())[None, None].float().div(255).to(selected)
                synchronize(selected)
                t0 = time.perf_counter()
                raw = fused_output(network, vis_t, ir_t)
                synchronize(selected)
                seconds = time.perf_counter() - t0
                bgr = reconstruct_bgr(raw, ycc)
                relative = Path(cfg["method"]) / "train" / f"{record.image_id}.png"
                digest = save_png(run_dir / relative, bgr)
                rows.append({
                    "image_id": record.image_id, "manifest_split": split, "source_split": "train",
                    "path": relative.as_posix(), "width": record.width, "height": record.height,
                    "vis_path": str(record.vis), "ir_path": str(record.ir),
                    "vis_sha256": hashes["vis"], "ir_sha256": hashes["ir"],
                    "checkpoint_sha256": chosen_sha, "pretrained_sha256": cfg["model"]["pretrained_sha256"],
                    "student_source_sha256": cfg["model"]["source_sha256"],
                    "config_sha256": cfg.fingerprint(), "png_sha256": digest,
                    "export_contract": cfg["inference"]["contract"], "dtype": "uint8",
                    "file_color": "RGB (OpenCV arrays BGR)", "raw_min": raw.min().item(),
                    "raw_max": raw.max().item(), "inference_seconds": seconds, "qa": "PASS",
                })
        with (run_dir / "output_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if sha256(chosen) != chosen_sha or sha256(cfg["model"]["pretrained"]) != cfg["model"]["pretrained_sha256"]:
            raise RuntimeError("Input checkpoint changed while exporting")
        result = {
            "status": "PASS", "count": len(rows), "split": split,
            "image_ids": [r["image_id"] for r in rows], "checkpoint_sha256": chosen_sha,
            "checkpoint_dry_run_only": metadata.get("dry_run_only", False),
            "raw_checkpoint_origin": "recorded_pipeline" if metadata else "raw_state_dict",
            "pretrained_origin_verified": bool(metadata) or chosen_sha == cfg["model"]["pretrained_sha256"],
            "config_sha256": cfg.fingerprint(), "export_contract": cfg["inference"]["contract"],
            "inference_seconds": sum(r["inference_seconds"] for r in rows),
            "wall_seconds": time.perf_counter() - start, "fmb_test_used": False,
            "output_manifest_sha256": sha256(run_dir / "output_manifest.csv"),
        }
        atomic_json(run_dir / "export_result.json", result)
        return result
    except Exception:
        import traceback
        atomic_json(run_dir / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Raw official state_dict or pipeline checkpoint")
    parser.add_argument("--output", required=True, help="New, nonexisting output directory")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--limit", type=int, default=5, help="PROVISIONAL cap=5; 0 means all if APPROVED")
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"])
    args = parser.parse_args(argv)
    result = export(load_config(args.config), args.output, checkpoint=args.checkpoint,
                    split=args.split, limit=args.limit or None, device=args.device)
    print(f"PASS: {result['count']} PNGs, inference {result['inference_seconds']:.6f}s")


if __name__ == "__main__":
    main()
