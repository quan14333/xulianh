"""Manifest-only, label-free FMB pairs. Development never enumerates FMB Test."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .checkpoint import sha256
from .config import canonical_hash


@dataclass(frozen=True)
class Pair:
    image_id: str
    split: str
    vis: Path
    ir: Path
    width: int
    height: int
    vis_md5: str
    ir_md5: str


def resolve_fmb_path(root, relative, *, allow_test=False):
    """Resolve flat/nested layouts; Test support is opt-in for future adapters only."""
    root = Path(root).resolve()
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or len(rel.parts) != 3 or ".." in rel.parts
            or "\\" in relative or ":" in relative):
        raise ValueError(f"Unsafe manifest path: {relative}")
    split, modality, filename = rel.parts
    if split not in {"train", "test"} or modality not in {"Visible", "Infrared"}:
        raise ValueError(f"Unsupported source path: {relative}")
    if split == "test" and not allow_test:
        raise PermissionError("FMB Test access is disabled")
    if not re.fullmatch(r"\d{5}\.(png|bmp|jpg|jpeg)", filename, re.IGNORECASE):
        raise ValueError(f"Invalid image filename: {filename}")
    candidates = [root / split / modality / filename, root / split / split / modality / filename]
    found = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Source path escapes FMB root")
        # Check the resolved path too, so a junction cannot redirect Train into Test.
        if not allow_test and any(p.lower() == "test" for p in resolved.relative_to(root).parts):
            raise PermissionError("Resolved path points into Test")
        if candidate.is_file():
            found.append(resolved)
    unique = list(dict.fromkeys(found))
    if len(unique) != 1:
        raise FileNotFoundError(f"Missing or ambiguous FMB pair member: {relative}: {unique}")
    return unique[0]


def read_manifest(path, root, split, *, expected_count, expected_sha256):
    if split not in {"train", "dev"}:
        raise PermissionError("Only train/dev manifests are allowed")
    if sha256(path) != expected_sha256:
        raise ValueError(f"Manifest SHA256 mismatch: {path}")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise ValueError(f"{split} count {len(rows)} != {expected_count}")
    records, seen = [], set()
    for row in rows:
        sid = row["image_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"\d{5}", sid) or sid in seen:
            raise ValueError(f"Invalid/duplicate image_id: {sid!r}")
        seen.add(sid)
        if row["split"] != split:
            raise ValueError(f"Unexpected split for {sid}")
        for key, modality in (("vis_path", "Visible"), ("ir_path", "Infrared")):
            rel = PurePosixPath(row[key])
            if rel.parts[:2] != ("train", modality) or rel.stem != sid:
                raise PermissionError(f"Development source/ID mismatch: {row[key]}")
        width, height = int(row["width"]), int(row["height"])
        if min(width, height) < 4:
            raise ValueError("Image is too small for Student pooling")
        for key in ("vis_md5", "ir_md5"):
            if not re.fullmatch(r"[0-9a-f]{32}", row[key]):
                raise ValueError(f"Missing/invalid manifest MD5 for {sid}")
        records.append(Pair(
            sid, split, resolve_fmb_path(root, row["vis_path"]),
            resolve_fmb_path(root, row["ir_path"]), width, height,
            row["vis_md5"], row["ir_md5"],
        ))
    return sorted(records, key=lambda r: r.image_id)


def load_protocol(cfg):
    d = cfg["data"]
    train = read_manifest(d["train_manifest"], d["root"], "train",
                          expected_count=d["train_count"], expected_sha256=d["train_sha256"])
    dev = read_manifest(d["dev_manifest"], d["root"], "dev",
                        expected_count=d["dev_count"], expected_sha256=d["dev_sha256"])
    if {r.image_id for r in train} & {r.image_id for r in dev}:
        raise ValueError("Train/Dev ID overlap")
    return train, dev


def read_pair(record):
    """Official OpenCV color conversions; no labels, resize, or RGB normalization."""
    if record.split not in {"train", "dev"}:
        raise PermissionError("Test pairs are disabled")
    checksums = {}
    arrays = []
    for key, path, expected in (("vis", record.vis, record.vis_md5),
                                ("ir", record.ir, record.ir_md5)):
        modality = "Visible" if key == "vis" else "Infrared"
        if (path.parent.name != modality or path.stem != record.image_id
                or any(part.lower() == "test" for part in path.parts)):
            raise PermissionError("Source path is not an allowed Train image")
        data = path.read_bytes()
        if hashlib.md5(data).hexdigest() != expected:
            raise ValueError(f"Input MD5 changed: {record.image_id}/{key}")
        checksums[key] = hashlib.sha256(data).hexdigest()
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint8:
            raise ValueError(f"Expected 8-bit input: {record.image_id}/{key}")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if (image is None or image.dtype != np.uint8 or image.ndim != 3
                or image.shape != (record.height, record.width, 3)):
            raise ValueError(f"Invalid shape/dtype/decode: {record.image_id}/{key}")
        arrays.append(image)
    ycc = cv2.cvtColor(arrays[0], cv2.COLOR_BGR2YCrCb)
    ir = cv2.cvtColor(arrays[1], cv2.COLOR_BGR2GRAY)
    return ycc, ir, checksums


def audit_inputs(records, destination):
    """Decode and hash 100% of the requested input records before any update."""
    rows = []
    for r in records:
        ycc, ir, hashes = read_pair(r)
        rows.append({
            "image_id": r.image_id, "split": r.split, "width": r.width, "height": r.height,
            "vis_path": str(r.vis), "ir_path": str(r.ir),
            "vis_sha256": hashes["vis"], "ir_sha256": hashes["ir"], "qa": "PASS",
        })
        if len(records) >= 200 and len(rows) % 200 == 0:
            print(f"Input QA {len(rows)}/{len(records)} pairs", flush=True)
    if not rows:
        raise ValueError("Empty input inventory")
    with Path(destination).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return canonical_hash(rows)


class PairDataset(Dataset):
    def __init__(self, records, *, training, crop_hw=None, hflip_p=0.0, seed=2):
        self.records = list(records)
        if not self.records:
            raise ValueError("Empty dataset")
        if any(r.split not in {"train", "dev"} for r in self.records):
            raise PermissionError("Test is not supported by the Development dataset")
        if training and any(r.split != "train" for r in self.records):
            raise PermissionError("Only Train may update weights")
        self.training = training
        self.crop_hw = tuple(crop_hw) if crop_hw else None
        self.hflip_p, self.seed, self.epoch = hflip_p, seed, 0
        if training and self.crop_hw is None:
            raise ValueError("Training requires an explicit crop")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        ycc, ir, hashes = read_pair(r)
        y = ycc[:, :, 0]
        top = left = 0
        flipped = False
        if self.training:
            h, w = self.crop_hw
            if h > y.shape[0] or w > y.shape[1]:
                raise ValueError(f"Crop exceeds input: {r.image_id}")
            # Reproducible by ID/epoch, independent of global RNG and worker ordering.
            salt = f"{self.seed}:{self.epoch}:{r.image_id}".encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(salt).digest()[:8], "big"))
            top, left = rng.randrange(y.shape[0] - h + 1), rng.randrange(y.shape[1] - w + 1)
            y, ir = y[top:top+h, left:left+w], ir[top:top+h, left:left+w]
            flipped = rng.random() < self.hflip_p
            if flipped:
                y, ir = y[:, ::-1], ir[:, ::-1]
        return {
            "vis": torch.from_numpy(y.copy()).unsqueeze(0).float().div(255),
            "ir": torch.from_numpy(ir.copy()).unsqueeze(0).float().div(255),
            "image_id": r.image_id, "split": r.split,
            "vis_sha256": hashes["vis"], "ir_sha256": hashes["ir"],
            "top": top, "left": left, "flipped": flipped,
        }
