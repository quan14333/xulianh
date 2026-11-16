"""Validated, hashable configuration and an explicit full-training approval guard."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPORT_CONTRACT = "b3_y_clamp_round_uint8_then_opencv_ycrcb_to_bgr"
SECTIONS = {
    "status", "approval_record", "protocol", "method", "data", "model",
    "loss", "optimizer", "scheduler", "training", "inference",
}


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"Invalid {name}: {value}")


@dataclass
class Config:
    values: dict

    def __getitem__(self, key):
        return self.values[key]

    def snapshot(self):
        return copy.deepcopy(self.values)

    def fingerprint(self):
        # Approval status is metadata. The approved record must bind all recipe fields.
        recipe = {k: v for k, v in self.values.items() if k not in {"status", "approval_record"}}
        return canonical_hash(recipe)


def parse_config(values, base=REPO_ROOT):
    v = copy.deepcopy(values)
    if set(v) != SECTIONS:
        raise ValueError(f"Config sections differ: {set(v) ^ SECTIONS}")
    if v["status"] not in {"PROVISIONAL", "APPROVED"}:
        raise ValueError("status must be PROVISIONAL or APPROVED")
    if v["protocol"] != "development_1045_175":
        raise ValueError("Only the Development protocol is implemented; Test is disabled")
    if not isinstance(v["method"], str) or not v["method"].replace("_", "").isalnum():
        raise ValueError("Unsafe method name")
    for section in ("data", "model", "loss", "optimizer", "scheduler", "training", "inference"):
        if not isinstance(v[section], dict):
            raise ValueError(f"{section} must be a mapping")
    allowed = {
        "data": {"root", "train_manifest", "dev_manifest", "train_count", "dev_count",
                 "train_sha256", "dev_sha256", "verify_md5"},
        "model": {"source_root", "source_sha256", "pretrained", "pretrained_sha256",
                  "dim", "train_patterns", "freeze_patterns", "bn_running_stats"},
        "loss": {"intensity", "gradient", "anchor"},
        "optimizer": {"name", "lr", "weight_decay", "betas", "eps"},
        "scheduler": {"name", "eta_min"},
        "training": {"seed", "epochs", "microbatch", "accumulation", "crop_hw",
                     "hflip_p", "device", "num_workers", "cpu_threads", "precision",
                     "tf32", "deterministic", "gradient_clip_norm", "peak_limit_gib",
                     "min_free_gib", "checkpoint_selection"},
        "inference": {"contract"},
    }
    for section, keys in allowed.items():
        if set(v[section]) != keys:
            raise ValueError(f"Unknown/missing {section} fields: {set(v[section]) ^ keys}")
    for section, keys in (("data", ("root", "train_manifest", "dev_manifest")),
                          ("model", ("source_root", "pretrained"))):
        for key in keys:
            p = Path(v[section][key]).expanduser()
            v[section][key] = str((Path(base) / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("train_count", "dev_count"):
        _positive(v["data"][key], key)
        if not isinstance(v["data"][key], int):
            raise ValueError("Counts must be integers")
    if (v["data"]["train_count"], v["data"]["dev_count"]) != (1045, 175):
        raise ValueError("Production Development config must register exactly 1045/175 IDs")
    for section, keys in (("data", ("train_sha256", "dev_sha256")),
                          ("model", ("source_sha256", "pretrained_sha256"))):
        for key in keys:
            h = v[section][key]
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError(f"Invalid SHA256: {section}.{key}")
    if v["data"]["verify_md5"] is not True:
        raise ValueError("Manifest MD5 verification may not be disabled")
    if v["model"]["dim"] != 32:
        raise ValueError("Student architecture dim must remain 32")
    if v["model"]["bn_running_stats"] not in {"frozen_pretrained", "train"}:
        raise ValueError("Unknown BN policy")
    for key in ("train_patterns", "freeze_patterns"):
        if not isinstance(v["model"][key], list) or not all(isinstance(x, str) for x in v["model"][key]):
            raise ValueError(f"{key} must be a list of parameter-name patterns")
    if not v["model"]["train_patterns"]:
        raise ValueError("At least one train pattern is required")
    for key, value in v["loss"].items():
        _positive(value, f"loss.{key}", zero=True)
    if sum(v["loss"].values()) == 0:
        raise ValueError("All loss weights are zero")
    opt = v["optimizer"]
    if opt["name"] not in {"Adam", "AdamW"}:
        raise ValueError("Use Adam or AdamW")
    for key in ("lr", "eps"):
        _positive(opt[key], key)
    _positive(opt["weight_decay"], "weight_decay", zero=True)
    if len(opt["betas"]) != 2 or not all(0 <= x < 1 for x in opt["betas"]):
        raise ValueError("Invalid Adam betas")
    if v["scheduler"]["name"] not in {"cosine", "constant"}:
        raise ValueError("Unknown scheduler")
    _positive(v["scheduler"]["eta_min"], "eta_min", zero=True)
    if v["scheduler"]["eta_min"] > opt["lr"]:
        raise ValueError("eta_min exceeds lr")
    t = v["training"]
    for key in ("epochs", "microbatch", "accumulation", "cpu_threads"):
        _positive(t[key], key)
        if not isinstance(t[key], int):
            raise ValueError(f"{key} must be an integer")
    if not isinstance(t["seed"], int) or t["seed"] < 0:
        raise ValueError("Invalid seed")
    if t["num_workers"] != 0:
        raise ValueError("This replay implementation requires num_workers=0")
    if len(t["crop_hw"]) != 2 or any(not isinstance(x, int) or x < 4 for x in t["crop_hw"]):
        raise ValueError("Invalid crop")
    if not 0 <= t["hflip_p"] <= 1:
        raise ValueError("Invalid flip probability")
    if t["precision"] != "fp32" or t["tf32"] is not False:
        raise ValueError("This protocol supports FP32 with TF32 disabled")
    if not isinstance(t["deterministic"], bool):
        raise ValueError("deterministic must be boolean")
    if t["device"] not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Unsupported device")
    for key in ("gradient_clip_norm", "peak_limit_gib"):
        _positive(t[key], key)
    _positive(t["min_free_gib"], "min_free_gib", zero=True)
    if t["checkpoint_selection"] != "last":
        raise ValueError("Canonical checkpoint selection must remain last")
    if v["inference"]["contract"] != EXPORT_CONTRACT:
        raise ValueError("Unsupported export contract")
    return Config(v)


def load_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return parse_config(yaml.safe_load(stream))


def authorize(cfg, mode, smoke_batches=2):
    if mode not in {"smoke", "train"}:
        raise ValueError("mode must be smoke or train")
    if mode == "smoke":
        if cfg["status"] == "PROVISIONAL" and cfg["training"]["microbatch"] > 2:
            raise PermissionError("PROVISIONAL smoke is capped at four Train images")
        if not 1 <= smoke_batches <= 2:
            raise PermissionError("Smoke is capped at TWO microbatches")
        return
    if cfg["status"] != "APPROVED" or not cfg["approval_record"]:
        raise PermissionError("PROVISIONAL objective: full training is blocked; use --mode smoke")
    p = Path(cfg["approval_record"])
    with p.open(encoding="utf-8") as stream:
        approval = json.load(stream)
    if (approval.get("allow_full_training") is not True
            or not approval.get("approved_by") or not approval.get("evidence")
            or approval.get("config_sha256") != cfg.fingerprint()):
        raise PermissionError("Approval record does not bind this exact recipe")


def new_output_directory(path, cfg):
    target = Path(path).resolve()
    protected = [
        Path(cfg["data"]["root"]).resolve(),
        Path(cfg["model"]["source_root"]).resolve(),
        Path(cfg["data"]["train_manifest"]).resolve().parent,
        Path(cfg["model"]["source_root"]).resolve().parent / "xulianh-b3-review",
    ]
    if any(target == p or target.is_relative_to(p) for p in protected):
        raise ValueError(f"Protected output path: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target
