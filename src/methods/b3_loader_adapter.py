"""Read-only bridge to Mân's actual FMBSeg, with a strict Dev-only path resolver."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

import cv2
import numpy as np
import torch
import yaml

from .sage_finetune.checkpoint import sha256
from .sage_finetune.config import REPO_ROOT, load_config
from .sage_finetune.dataset import load_protocol


def resolve_dev_label(root, relative, allowed_ids):
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or len(rel.parts) != 3 or rel.parts[:2] != ("train", "Label")
            or rel.stem not in allowed_ids or rel.suffix.lower() != ".png"
            or ".." in rel.parts or "\\" in relative or ":" in relative):
        raise PermissionError(f"Only selected Dev labels are allowed: {relative}")
    root = Path(root).resolve()
    candidates = [root / Path(*rel.parts), root / "train" / Path(*rel.parts)]
    found = []
    for p in candidates:
        resolved = p.resolve()
        if not resolved.is_relative_to(root) or any(x.lower() == "test" for x in resolved.relative_to(root).parts):
            raise PermissionError("Resolved label path escapes Development")
        if resolved.is_file():
            found.append(resolved)
    found = list(dict.fromkeys(found))
    if len(found) != 1:
        raise FileNotFoundError(f"Missing/ambiguous Dev label: {relative}")
    return found[0]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # No __pycache__ or other writes into the B3 reference.
    exec(compile(Path(path).read_bytes(), str(path), "exec"), mod.__dict__)
    return mod


@contextmanager
def actual_b3_dataset(reference_root, manifest_dir, fused_root, resolver, expected_hashes=None):
    root = Path(reference_root).resolve() / "src/segmentation"
    hashes = {name: sha256(root/name) for name in ("chung.py", "train_seg.py")}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError("B3 loader source SHA256 differs from registry")
    previous = sys.modules.get("chung")
    common = _module("chung", root/"chung.py")
    # Only IO binding changes in memory. Normalization/mapping/FMBSeg code stays original.
    common.MAN, common.ANH = Path(manifest_dir), Path(fused_root)
    common.duong = resolver
    sys.modules["chung"] = common
    try:
        training = _module("_locbe_readonly_b3_train_seg", root/"train_seg.py")
        yield training.FMBSeg, hashes
    finally:
        if previous is None:
            sys.modules.pop("chung", None)
        else:
            sys.modules["chung"] = previous
        if hashes != {name: sha256(root/name) for name in hashes}:
            raise RuntimeError("B3 reference files changed")


def inspect_inputs(cfg, export_root, variant, reference_root, output, expected_hashes=None,
                   expected_checkpoint_sha256=None):
    if variant not in {"sage", "sage_finetuned"}:
        raise ValueError("No PNGs exist for an unspecified Quân method")
    root, output = Path(export_root).resolve(), Path(output).resolve()
    protected = [Path(cfg["data"]["root"]).resolve(), Path(cfg["model"]["source_root"]).resolve(),
                 Path(cfg["data"]["dev_manifest"]).resolve().parent,
                 Path(reference_root).resolve()/"src/segmentation"]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("Protected loader-check output path")
    if output.exists():
        raise FileExistsError("Refuse to overwrite a loader check")
    with (root/"output_manifest.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    if not 1 <= len(exported) <= 5:
        raise ValueError("Development loader smoke must contain 1..5 images")
    ids = [r["image_id"] for r in exported]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate exported IDs")
    _, dev = load_protocol(cfg)
    dev_by_id = {r.image_id: r for r in dev}
    if not set(ids) <= dev_by_id.keys():
        raise PermissionError("Export contains non-Dev IDs")
    for row in exported:
        if row["config_sha256"] != cfg.fingerprint():
            raise ValueError("Export recipe hash differs from supplied config")
        if expected_checkpoint_sha256 and row["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("Export checkpoint differs from registered variant")
        record = dev_by_id[row["image_id"]]
        if row["vis_sha256"] != sha256(record.vis) or row["ir_sha256"] != sha256(record.ir):
            raise ValueError("Original sources changed since export")
        expected = f"{variant}/train/{row['image_id']}.png"
        if (row["variant"] != variant or row["manifest_split"] != "dev"
                or row["source_split"] != "train" or row["path"] != expected
                or sha256(root/expected) != row["png_sha256"]):
            raise ValueError("Export layout/split/hash mismatch")
    with Path(cfg["data"]["dev_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    selected = [r for r in all_rows if r["image_id"] in set(ids)]
    for row in selected:
        if PurePosixPath(row["label_path"]).stem != row["image_id"]:
            raise ValueError("Dev label ID differs from image ID")
        label = resolve_dev_label(cfg["data"]["root"], row["label_path"], set(ids))
        if row.get("label_md5") and hashlib.md5(label.read_bytes()).hexdigest() != row["label_md5"]:
            raise ValueError("Dev label MD5 differs from manifest")
    subset = output.parent/(output.stem+"_manifests")
    subset.mkdir(parents=True, exist_ok=False)
    with (subset/"dev.csv").open("x", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(selected)
    label_paths = []

    def resolver(relative):
        label = resolve_dev_label(cfg["data"]["root"], relative, set(ids))
        label_paths.append(str(label))
        return label

    checks = []
    with actual_b3_dataset(reference_root, subset, root, resolver, expected_hashes) as (FMBSeg, hashes):
        dataset = FMBSeg(["dev"], variant, False, lop0=True)
        if len(dataset) != len(ids):
            raise ValueError("B3 selected manifest count mismatch")
        for index in range(len(dataset)):
            sid = dataset.df.image_id.iloc[index]
            x, y = dataset[index]  # Actual, unmodified FMBSeg implementation.
            record = dev_by_id[sid]
            if x.shape != (3, record.height, record.width) or y.shape != (record.height, record.width):
                raise ValueError("B3 image/label shape mismatch")
            if x.dtype != torch.float32 or y.dtype != torch.int64 or not torch.isfinite(x).all():
                raise ValueError("B3 tensor dtype/finite check failed")
            unique = torch.unique(y).tolist()
            if min(unique) < 0 or max(unique) > 14:
                raise ValueError("Invalid raw FMB 15-class IDs")
            image = cv2.cvtColor(cv2.imread(str(root/variant/"train"/(sid+".png"))), cv2.COLOR_BGR2RGB)
            expected = (image.astype(np.float32)/255 - np.array([.485,.456,.406],np.float32))
            expected /= np.array([.229,.224,.225],np.float32)
            if not torch.equal(x, torch.from_numpy(expected.transpose(2,0,1).copy())):
                raise ValueError("B3 normalization/color contract mismatch")
            raw_label = cv2.imread(str(resolve_dev_label(cfg["data"]["root"],
                selected[index]["label_path"], set(ids))), cv2.IMREAD_UNCHANGED)
            if not np.array_equal(y.numpy(), raw_label.astype(np.int64)):
                raise ValueError("15-class label mapping changed")
            checks.append({"image_id": sid, "status": "PASS", "image_shape": list(x.shape),
                           "mask_shape": list(y.shape), "label_ids": unique,
                           "image_dtype": str(x.dtype), "mask_dtype": str(y.dtype)})
    result = {
        "status": "PASS", "variant": variant, "count": len(checks),
        "loader": "actual Mân FMBSeg(train=False, lop0=True), source not copied",
        "b3_reference_root": str(Path(reference_root).resolve()), "source_sha256": hashes,
        "io_adaptation": "isolated module globals and selected-Dev label resolver only",
        "label_access": "selected Dev GT read ONLY for segmentation loader QA, never fusion loss",
        "label_paths": sorted(set(label_paths)), "fmb_test_used": False,
        "segmentation_inference": "NOT_AVAILABLE: no authorized local SegFormer-B3 FMB checkpoint",
        "segmentation_training": "NOT_RUN", "images": checks,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--variant", choices=["sage", "sage_finetuned"], required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.registry).open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    cfg = load_config(REPO_ROOT/registry["fusion_config"])
    result = inspect_inputs(cfg, args.export_root, args.variant, registry["b3"]["reference_root"],
                            args.output, registry["b3"]["source_sha256"],
                            registry["variants"][args.variant].get("sha256"))
    print(f"PASS actual B3 loader: {result['variant']} {result['count']} Dev pairs")


if __name__ == "__main__":
    main()
