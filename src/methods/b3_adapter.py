"""Unified SAGE/Quân interface and B3 folder export, independent of Mân's code."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from .fusion_interface import FusionPair, FusionPrediction, WaitingForMethodSpec
from .quan_method import QuanMethodPending
from .sage_finetune.checkpoint import atomic_json, environment, read_checkpoint, sha256, strict_load
from .sage_finetune.config import REPO_ROOT, authorize, load_config, new_output_directory
from .sage_finetune.dataset import load_protocol, read_pair
from .sage_finetune.infer import reconstruct_bgr, save_png
from .sage_finetune.model import fused_output, load_pretrained
from .sage_finetune.train import configure_runtime, synchronize


def method_statuses():
    return {
        "sage": "FROZEN_PRETRAINED",
        "sage_finetuned": "CHECKPOINT_AND_PROVENANCE_REQUIRED",
        "quan_improved": "WAITING_FOR_METHOD_SPEC",
    }


class SageMethod:
    def __init__(self, cfg, variant, *, checkpoint=None, expected_sha256=None,
                 allow_pilot_checkpoint=False, device="cpu"):
        if variant not in {"sage", "sage_finetuned"}:
            raise ValueError(f"Unsupported SAGE variant: {variant}")
        chosen = Path(checkpoint or cfg["model"]["pretrained"]).resolve()
        digest = sha256(chosen)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Adapter checkpoint SHA256 mismatch")
        if variant == "sage" and digest != cfg["model"]["pretrained_sha256"]:
            raise ValueError("The pretrained sage baseline cannot be replaced by fine-tuned weights")
        _, packet = read_checkpoint(chosen)
        role, epochs, samples = "FROZEN_PRETRAINED", None, None
        if variant == "sage_finetuned":
            if digest == cfg["model"]["pretrained_sha256"] or not packet:
                raise ValueError("sage_finetuned requires a provenance-bearing pipeline checkpoint")
            if packet["provenance"]["pretrained_sha256"] != cfg["model"]["pretrained_sha256"]:
                raise ValueError("Different original Student checkpoint")
            if packet["config_hash"] != cfg.fingerprint():
                raise ValueError("Fine-tuning recipe hash differs from the supplied config")
            if packet.get("mode") not in {"smoke", "train"}:
                raise ValueError("Unknown fine-tuning checkpoint mode")
            epochs = packet["progress"]["epoch"]
            samples = sum(row["train_samples"] for row in packet["history"])
            role = "PILOT_ONLY" if packet["dry_run_only"] else "TRAINED_STUDENT"
            if epochs < cfg["training"]["epochs"] and role != "PILOT_ONLY":
                role = "PARTIAL_TRAINING"
            if role != "TRAINED_STUDENT" and not allow_pilot_checkpoint:
                raise PermissionError("Incomplete/pilot checkpoint: explicit --allow-pilot-checkpoint required")
        self.network = load_pretrained(cfg)
        strict_load(self.network, chosen)
        self.network.to(device).eval()
        self.device = torch.device(device)
        self.identity = {
            "variant": variant, "checkpoint": str(chosen), "checkpoint_sha256": digest,
            "checkpoint_role": role, "completed_finetuning_epochs": epochs,
            "train_samples_processed": samples, "full_evaluation_eligible": role != "PILOT_ONLY"
            and role != "PARTIAL_TRAINING" and cfg["status"] == "APPROVED",
            "source_sha256": cfg["model"]["source_sha256"],
            "pretrained_sha256": cfg["model"]["pretrained_sha256"],
            "pretraining_ids": "UNKNOWN",
            "finetuning_protocol": cfg["protocol"] if packet else None,
            "finetuning_train_manifest_sha256": packet["provenance"]["train_manifest_sha256"] if packet else None,
            "checkpoint_selection": packet.get("canonical_selection") if packet else None,
            "target_epochs": cfg["training"]["epochs"] if packet else None,
            "config_sha256": cfg.fingerprint(),
            "export_contract": cfg["inference"]["contract"],
        }

    def fuse(self, pair):
        pair.validate()
        ycc = cv2.cvtColor(pair.vis_bgr, cv2.COLOR_BGR2YCrCb)
        ir = cv2.cvtColor(pair.ir_bgr, cv2.COLOR_BGR2GRAY)
        tensor = lambda x: torch.from_numpy(x.copy()).float().div(255)[None, None].to(self.device)
        with torch.no_grad():
            synchronize(self.device)
            start = time.perf_counter()
            raw = fused_output(self.network, tensor(ycc[:, :, 0]), tensor(ir))
            synchronize(self.device)
            seconds = time.perf_counter() - start
            bgr = reconstruct_bgr(raw, ycc)
        prediction = FusionPrediction(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {
            **self.identity, "inference_seconds": seconds,
            "raw_min": raw.min().item(), "raw_max": raw.max().item(),
        })
        prediction.validate(pair)
        return prediction


def build_method(variant, cfg, **kwargs):
    if variant == "quan_improved":
        return QuanMethodPending()
    if variant not in method_statuses():
        raise ValueError(f"Unknown variant: {variant}")
    return SageMethod(cfg, variant, **kwargs)


def export_variant(cfg, variant, output, *, checkpoint=None, expected_sha256=None,
                   allow_pilot_checkpoint=False, split="dev", limit=5, device=None):
    if split not in {"train", "dev"}:
        raise PermissionError("Development adapter does not read FMB Test")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if limit is None or limit > 5:
        authorize(cfg, "train")
    # Method availability must fail before creating an output or reading source images.
    if variant == "quan_improved":
        raise WaitingForMethodSpec("Quân method has no supplied specification/prototype")
    selected = configure_runtime(cfg, device)
    method = build_method(variant, cfg, checkpoint=checkpoint, expected_sha256=expected_sha256,
                          allow_pilot_checkpoint=allow_pilot_checkpoint, device=selected)
    if method.identity["checkpoint_role"] in {"PILOT_ONLY", "PARTIAL_TRAINING"} and (limit is None or limit > 5):
        raise PermissionError("Pilot/partial checkpoints cannot be used for full dataset comparisons")
    root = new_output_directory(output, cfg)
    train, dev = load_protocol(cfg)
    records = train if split == "train" else dev
    records = records[:limit] if limit else records
    rows = []
    for record in records:
        # Verify original bytes/shape/MD5 first. Decode the same bytes in BGR for the common API.
        _, _, hashes = read_pair(record)
        v_bytes, i_bytes = record.vis.read_bytes(), record.ir.read_bytes()
        if (hashlib.sha256(v_bytes).hexdigest() != hashes["vis"]
                or hashlib.sha256(i_bytes).hexdigest() != hashes["ir"]):
            raise RuntimeError("Source image changed between QA and fusion")
        vis = cv2.imdecode(np.frombuffer(v_bytes, np.uint8), cv2.IMREAD_COLOR)
        ir = cv2.imdecode(np.frombuffer(i_bytes, np.uint8), cv2.IMREAD_COLOR)
        pair = FusionPair(record.image_id, vis, ir, "train")
        pred = method.fuse(pair)
        relative = Path(variant) / pair.source_split / (pair.image_id + ".png")
        png_hash = save_png(root / relative, cv2.cvtColor(pred.rgb, cv2.COLOR_RGB2BGR))
        rows.append({
            "image_id": pair.image_id, "manifest_split": split, "source_split": pair.source_split,
            "variant": variant, "path": relative.as_posix(), "width": record.width, "height": record.height,
            "vis_sha256": hashes["vis"], "ir_sha256": hashes["ir"], "png_sha256": png_hash,
            "checkpoint_sha256": method.identity["checkpoint_sha256"],
            "checkpoint_role": method.identity["checkpoint_role"],
            "completed_finetuning_epochs": method.identity["completed_finetuning_epochs"],
            "config_sha256": cfg.fingerprint(), "inference_seconds": pred.metadata["inference_seconds"],
            "raw_min": pred.metadata["raw_min"], "raw_max": pred.metadata["raw_max"],
            "export_contract": cfg["inference"]["contract"], "qa": "PASS",
        })
    with (root / "output_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if sha256(method.identity["checkpoint"]) != method.identity["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during export")
    result = {
        "status": "PASS", "variant": variant, "count": len(rows),
        "image_ids": [r["image_id"] for r in rows], "identity": method.identity,
        "inference_seconds": sum(r["inference_seconds"] for r in rows),
        "fmb_test_used": False, "semantic_metrics_measured": False,
        "manifest_sha256": sha256(root / "output_manifest.csv"),
    }
    atomic_json(root / "environment.json", environment(selected))
    atomic_json(root / "variant_registration.json", {
        "status": "TECHNICAL_SMOKE_ONLY" if cfg["status"] == "PROVISIONAL" else "REVIEW_RECORD_REQUIRED_FOR_B3",
        "identity": method.identity, "student_config": cfg.snapshot(),
        "b3_training_authorized": False,
    })
    atomic_json(root / "export_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--variant", choices=list(method_statuses()), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--allow-pilot-checkpoint", action="store_true")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda")
    args = parser.parse_args(argv)
    with Path(args.registry).open(encoding="utf-8-sig") as stream:
        registry = yaml.safe_load(stream)
    if registry["schema_version"] != 1:
        raise ValueError("Unknown registry schema")
    entry = registry["variants"][args.variant]
    config = load_config(REPO_ROOT / registry["fusion_config"])
    try:
        result = export_variant(
            config, args.variant, args.output,
            checkpoint=args.checkpoint or entry.get("checkpoint"),
            expected_sha256=args.checkpoint_sha256 or entry.get("sha256"),
            allow_pilot_checkpoint=args.allow_pilot_checkpoint,
            split=args.split, limit=args.limit or None, device=args.device)
    except WaitingForMethodSpec as err:
        import json
        print(json.dumps({"status": err.status, "reason": str(err)}))
        raise SystemExit(2)
    print(f"PASS {result['variant']}: {result['count']} PNG; "
          f"{result['identity']['checkpoint_role']}; {result['inference_seconds']:.6f}s")


if __name__ == "__main__":
    main()
