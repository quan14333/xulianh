"""Safe state-only loading, complete training resumes, and artifact provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import uuid

import numpy as np
import torch

SCHEMA = "sage_student_finetune_v1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_finite_state(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty state_dict")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State dictionary contains non-tensor entries")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")


def read_checkpoint(path):
    # No arbitrary pickle execution, key stripping, prefix rewriting, or strict=False.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and value.get("schema") == SCHEMA:
        state = value["model"]
        metadata = value
    else:
        state, metadata = value, {}
    assert_finite_state(state)
    return state, metadata


def strict_load(model, path):
    state, metadata = read_checkpoint(path)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Strict checkpoint load reported mismatches")
    actual = model.state_dict()
    if any(actual[k].dtype != v.dtype or not torch.equal(actual[k].cpu(), v) for k, v in state.items()):
        raise RuntimeError("A loaded tensor differs from the checkpoint")
    return metadata


def capture_rng():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(value):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot resume on CPU")
        torch.cuda.set_rng_state_all(value["cuda"])


def atomic_json(path, value):
    target = Path(path)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()  # Only the exact newly created temporary file.


def save_training_checkpoint(path, model, optimizer, scheduler, *, config_hash,
                             provenance, progress, best_loss, mode, history):
    payload = {
        "schema": SCHEMA, "model": cpu_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng(), "config_hash": config_hash,
        "provenance": provenance, "progress": progress, "best_loss": float(best_loss),
        "mode": mode, "history": history,
        "canonical_selection": "last", "best_selection": "auxiliary_validation_loss",
        "dry_run_only": mode == "smoke",
    }
    assert_finite_state(payload["model"])
    target = Path(path)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            torch.save(payload, stream)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return sha256(target)


def resume_training(path, model, optimizer, scheduler, *, config_hash, provenance, mode):
    state, payload = read_checkpoint(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Resume requires a complete pipeline checkpoint, not raw weights")
    if payload["config_hash"] != config_hash or payload["provenance"] != provenance:
        raise ValueError("Resume config/source/input hashes differ")
    if payload["mode"] != mode:
        raise PermissionError("Smoke checkpoints cannot be promoted to full training")
    strict_load(model, path)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload


def source_snapshot(cfg):
    source = Path(cfg["model"]["source_root"])
    paths = list((Path(__file__).parent).glob("*.py"))
    paths += [source / "model_sub/model.py"]
    return {str(p): sha256(p) for p in sorted(paths)}


def environment(device):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"
    versions = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "PyYAML", "Pillow"):
        try:
            versions[name] = str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import platform
    record = {
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": versions, "device": str(device), "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        record["gpu"] = {"name": p.name, "total_memory_bytes": p.total_memory,
                         "capability": list(torch.cuda.get_device_capability(device))}
    return record
