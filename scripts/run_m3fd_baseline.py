"""Run the pinned SAGE Student on all 300 M3FD Fusion pairs, then evaluate.

CPU and CUDA supported. No training or segmentation dependency is required.
See notebooks/loc_lon/M3FD_PROTOCOL.md for provenance and numerical conventions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import EvaluationConfig, FusionEvaluator  # noqa: E402

UPSTREAM_COMMIT = "377fdbef6b8b6ccbd1139358ad66bb41ea084dc1"
CHECKPOINT_SHA256 = "ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836"
PROTOCOL = "loclon_m3fd_y_reflect_v2"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory missing: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image ID: {path.stem} in {directory}")
            result[path.stem] = path
    return result


def audit_pairs(dataset: Path, expected_count: int = 300) -> list[dict]:
    ir, vis = index_images(dataset / "Ir"), index_images(dataset / "Vis")
    if ir.keys() != vis.keys():
        raise ValueError(f"Unmatched IDs: IR-only={sorted(ir.keys()-vis.keys())}, "
                         f"VIS-only={sorted(vis.keys()-ir.keys())}")
    if len(ir) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs; found {len(ir)}")
    pairs = []
    for image_id in sorted(ir):
        with Image.open(ir[image_id]) as a, Image.open(vis[image_id]) as b:
            a.load()
            b.load()
            if a.size != b.size:
                raise ValueError(f"Shape mismatch: {image_id}: {a.size} vs {b.size}")
            width, height = a.size
        pairs.append(dict(image_id=image_id, ir_path=ir[image_id], vis_path=vis[image_id],
                          width=width, height=height, ir_sha256=sha256(ir[image_id]),
                          vis_sha256=sha256(vis[image_id])))
    return pairs


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                   text=True).strip()


def make_preview(pairs: list[dict], run: Path) -> None:
    selected = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 6, dtype=int)]
    canvas = Image.new("RGB", (900, len(selected) * 245), "white")
    draw = ImageDraw.Draw(canvas)
    for row, item in enumerate(selected):
        for col, (label, path) in enumerate([
            ("IR", item["ir_path"]), ("VIS", item["vis_path"]),
            ("SAGE", run / "fused" / f"{item['image_id']}.png"),
        ]):
            with Image.open(path) as source:
                tile = source.convert("RGB")
                tile.thumbnail((300, 220))
                canvas.paste(tile, (col * 300, row * 245 + 25))
            draw.text((col * 300 + 5, row * 245 + 5), f"{item['image_id']} {label}", fill="black")
    (run / "figures").mkdir()
    canvas.save(run / "figures" / "preview_ir_vis_sage.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/M3FD_Fusion")
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/SAGE_IVIF")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    pairs = audit_pairs(args.dataset_root.resolve())
    if args.audit_only:
        print(f"Audit passed: {len(pairs)} readable pairs, exact IDs and matching shapes")
        return 0

    import cv2
    import torch

    if git_commit(args.upstream) != UPSTREAM_COMMIT:
        raise ValueError(f"Use upstream commit {UPSTREAM_COMMIT}")
    model_path = args.upstream / "model_sub/model.py"
    if subprocess.check_output(["git", "-C", str(args.upstream), "diff", "HEAD", "--",
                                "model_sub/model.py"], text=True):
        raise ValueError("Upstream model has local modifications")
    checkpoint = args.checkpoint or args.upstream / "checkpoints/checkpoints.pt"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the FMB baseline")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2)
    np.random.seed(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    spec = importlib.util.spec_from_file_location("sage_pinned_student", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Network().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_sage_m3fd_student"
    run = args.output_dir or ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    (run / "fused").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run / "logs/run.log"), logging.StreamHandler()])
    config_path = ROOT / "configs/evaluation/m3fd_fusion.yaml"
    config = dict(run_id=run_id, source="team-reproduced", method="SAGE pretrained Student",
                  dataset="M3FD_Fusion", expected_count=300, seed=2, batch_size=1,
                  upstream_url="https://github.com/RollingPlain/SAGE_IVIF",
                  upstream_commit=UPSTREAM_COMMIT, upstream_license="MIT",
                  model_sha256=sha256(model_path), checkpoint_sha256=CHECKPOINT_SHA256,
                  student_parameters=sum(p.numel() for p in model.parameters()),
                  repository_commit=git_commit(ROOT),
                  protocol_id=PROTOCOL, evaluation_config_sha256=sha256(config_path),
                  evaluator_sha256=sha256(ROOT / "src/evaluation/metrics.py"),
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [
                      ROOT / "src/evaluation/evaluator.py", ROOT / "src/evaluation/image_io.py",
                      ROOT / "src/evaluation/metrics.py", config_path]},
                  runner_sha256=sha256(Path(__file__)), device=str(device), threads=args.threads,
                  preprocessing="OpenCV BGR->YCrCb (uint8)/255; IR BGR->GRAY/255; no resize",
                  postprocessing="upstream YCrCb2RGB matrix; clamp; multiply 255 add .5; uint8 RGB PNG",
                  adaptations=["import model_sub/model.py (upstream test.py references missing model_sub.py)",
                               "device-neutral upstream color matrix; no architecture or weight changes",
                               "sorted exact ID pairing; deterministic seed; explicit CPU support"],
                  evaluation="Saved PNG BT.601 float64 Y; raw metrics; reflect padding v2; mean MS-SSIM",
                  status="running")
    write_json(run / "run_config.json", config)
    write_json(run / "environment.json", dict(python=platform.python_version(),
               platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
               opencv=cv2.__version__, pillow=Image.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None))
    # Paths are relative to dataset root; hashes make the snapshot portable.
    manifest = [{**p, "ir_path": str(p["ir_path"].relative_to(args.dataset_root.resolve())),
                 "vis_path": str(p["vis_path"].relative_to(args.dataset_root.resolve()))} for p in pairs]
    write_csv(run / "manifest.csv", manifest)
    evaluator = FusionEvaluator(EvaluationConfig.from_yaml(config_path))
    records, inference = [], []
    matrix = torch.tensor([[1., 1., 1.], [1.403, -.714, 0.], [0., -.344, 1.773]], device=device)
    bias = torch.tensor([0., -.5, -.5], device=device)
    try:
        with torch.inference_mode():
            for index, pair in enumerate(pairs, 1):
                started = time.perf_counter()
                ir = cv2.cvtColor(cv2.imread(str(pair["ir_path"])), cv2.COLOR_BGR2GRAY)
                ycrcb = cv2.cvtColor(cv2.imread(str(pair["vis_path"])), cv2.COLOR_BGR2YCrCb)
                tensor = torch.from_numpy(ycrcb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255
                infrared = torch.from_numpy(ir.copy()).float()[None, None].to(device) / 255
                output, features = model(tensor[:, :1], infrared)
                if not torch.isfinite(output).all():
                    raise ValueError(f"Nonfinite inference: {pair['image_id']}")
                flat = torch.cat([output, tensor[:, 1:]], dim=1).permute(0, 2, 3, 1)
                colored = (flat + bias) @ matrix
                pixels = colored.mul(255).add(.5).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                fused_path = run / "fused" / f"{pair['image_id']}.png"
                Image.fromarray(pixels).save(fused_path)
                del features, output, colored, flat, tensor, infrared
                record = evaluator.evaluate_files(fused_path, pair["ir_path"], pair["vis_path"], pair["image_id"])
                if not record["valid"] or not all(np.isfinite(record[m]) for m in ["EN", "SD", "SCD", "MS_SSIM"]):
                    raise ValueError(f"Invalid evaluation: {record}")
                records.append(record)
                inference.append(dict(image_id=pair["image_id"], valid=True,
                                      elapsed_seconds=time.perf_counter()-started, fused_sha256=sha256(fused_path)))
                logging.info("%d/300 %s %.2fs EN=%.4f SD=%.4f SCD=%.4f MS_SSIM=%.4f",
                             index, pair["image_id"], inference[-1]["elapsed_seconds"],
                             *[record[m] for m in ["EN", "SD", "SCD", "MS_SSIM"]])
                # Preserve progress on interruption; partial runs never marked complete.
                evaluator.save_per_image_csv(records, run / "metrics/per_image.csv")
                write_csv(run / "inference.csv", inference)
        if set(index_images(run / "fused")) != {p["image_id"] for p in pairs}:
            raise ValueError("Output ID mismatch")
        summary = evaluator.compute_summary(records, run_id)
        # This task evaluates only four metrics; exclude uncomputed NR-IQA placeholders.
        summary["metrics"] = {k: v for k, v in summary["metrics"].items() if v["valid_count"]}
        summary.pop("paper_benchmark", None)
        summary.update(protocol_id=PROTOCOL, source="team-reproduced",
                       paper_parity="not established", checkpoint_sha256=CHECKPOINT_SHA256)
        write_json(run / "metrics/summary.json", summary)
        make_preview(pairs, run)
        config["status"] = "complete"
        config["completed_images"] = len(records)
        write_json(run / "run_config.json", config)
        logging.info("COMPLETE: %s", run)
    except BaseException:
        config["status"] = "failed_or_interrupted"
        config["completed_images"] = len(records)
        write_json(run / "run_config.json", config)
        logging.exception("Run incomplete")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
