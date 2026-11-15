"""Import the pinned official Student without importing SAGE training/SAM code."""
from __future__ import annotations

import fnmatch
import importlib.util
from pathlib import Path

import torch
from torch import nn

from .checkpoint import sha256, state_digest, strict_load


def make_student(cfg):
    path = Path(cfg["model"]["source_root"]) / "model_sub/model.py"
    if sha256(path) != cfg["model"]["source_sha256"]:
        raise ValueError("Official Student source SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("audited_sage_student", path)
    module = importlib.util.module_from_spec(spec)
    # Execute the hash-pinned official file without creating upstream __pycache__.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module.Network(dim=cfg["model"]["dim"])


def load_pretrained(cfg):
    path = Path(cfg["model"]["pretrained"])
    if sha256(path) != cfg["model"]["pretrained_sha256"]:
        raise ValueError("Pretrained checkpoint SHA256 mismatch")
    model = make_student(cfg)
    strict_load(model, path)
    return model.float()


def apply_freezing(model, cfg):
    params = dict(model.named_parameters())
    for pattern in cfg["model"]["train_patterns"] + cfg["model"]["freeze_patterns"]:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in params):
            raise ValueError(f"Parameter pattern matches nothing: {pattern}")
    for name, param in params.items():
        train = any(fnmatch.fnmatchcase(name, p) for p in cfg["model"]["train_patterns"])
        frozen = any(fnmatch.fnmatchcase(name, p) for p in cfg["model"]["freeze_patterns"])
        param.requires_grad_(train and not frozen)
    if not any(p.requires_grad for p in params.values()):
        raise ValueError("No trainable parameters remain")


def training_mode(model, cfg):
    model.train()
    if cfg["model"]["bn_running_stats"] == "frozen_pretrained":
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # affine gradients stay enabled if allowed by config.


def initial_pair(cfg):
    student, reference = load_pretrained(cfg), load_pretrained(cfg)
    if state_digest(student.state_dict()) != state_digest(reference.state_dict()):
        raise RuntimeError("Initialization differs between Student and reference")
    reference.eval().requires_grad_(False)
    apply_freezing(student, cfg)
    return student, reference


def validate_inputs(vis, ir):
    if (vis.shape != ir.shape or vis.ndim != 4 or vis.shape[1] != 1
            or min(vis.shape[-2:]) < 4 or vis.dtype != torch.float32
            or ir.dtype != torch.float32 or vis.device != ir.device):
        raise ValueError("Expected same-device FP32 VIS_Y/IR tensors N1HW, H/W>=4")
    for value in (vis, ir):
        if not torch.isfinite(value).all().item() or value.min().item() < 0 or value.max().item() > 1:
            raise ValueError("Input must be finite and normalized to [0,1]")


def fused_output(model, vis, ir):
    validate_inputs(vis, ir)
    result = model(vis, ir)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Student must return (output, features)")
    output = result[0]
    if output.shape != vis.shape or output.dtype != torch.float32:
        raise ValueError("Student output shape/dtype differs from input")
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("Student output has NaN/Inf")
    if output.min().item() < -1.000001 or output.max().item() > 1.000001:
        raise ValueError("Raw Student output exceeds the Tanh range [-1,1]")
    return output
