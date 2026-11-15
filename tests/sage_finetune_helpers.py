"""Tiny synthetic fixtures only; never represent SAGE/FMB experimental results."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import textwrap

import cv2
import numpy as np
import torch
import yaml

from src.methods.sage_finetune.checkpoint import sha256
from src.methods.sage_finetune.config import Config, REPO_ROOT

TOY_SOURCE = """
import torch
from torch import nn
class Network(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Conv2d(2,4,3,padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.out = nn.Conv2d(4,1,1)
    def forward(self, vi, ir):
        x = torch.nn.functional.leaky_relu(self.bn(self.conv(torch.cat([vi,ir],1))))
        return self.out(x).tanh(), (x,x,x,x,x)
"""


def fixture(root, n_train=3, n_dev=2, nested=True):
    root = Path(root)
    source = root / "synthetic_source"
    (source / "model_sub").mkdir(parents=True)
    (source / "model_sub/model.py").write_text(textwrap.dedent(TOY_SOURCE), encoding="utf-8")
    namespace = {}
    exec(TOY_SOURCE, namespace)
    torch.manual_seed(17)
    model = namespace["Network"]()
    torch.save(model.state_dict(), source / "initial.pt")
    data = root / "synthetic_fmb"
    layout = data / "train" / "train" if nested else data / "train"
    for modality in ("Visible", "Infrared"):
        (layout / modality).mkdir(parents=True)
    rows = {"train": [], "dev": []}
    for idx in range(n_train + n_dev):
        sid = f"{idx+1:05d}"
        yy, xx = np.indices((17, 23))
        vis = np.stack([(xx*7+idx)%256, (yy*11+idx)%256, (xx+yy*3+idx)%256], axis=-1).astype(np.uint8)
        ir = ((xx*3 + yy*5 + 30 + idx*2)%256).astype(np.uint8)
        v, i = layout / "Visible" / (sid+".png"), layout / "Infrared" / (sid+".png")
        assert cv2.imwrite(str(v), vis) and cv2.imwrite(str(i), ir)
        split = "train" if idx < n_train else "dev"
        rows[split].append({
            "image_id": sid, "split": split, "vis_path": f"train/Visible/{sid}.png",
            "ir_path": f"train/Infrared/{sid}.png", "label_path": "train/Label/NEVER_READ.png",
            "width": 23, "height": 17,
            "vis_md5": hashlib.md5(v.read_bytes()).hexdigest(),
            "ir_md5": hashlib.md5(i.read_bytes()).hexdigest(),
        })
    manifests = root / "manifests"
    manifests.mkdir()
    for split in ("train", "dev"):
        with (manifests / (split+".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[split][0]))
            writer.writeheader()
            writer.writerows(rows[split])
    with (REPO_ROOT / "configs/methods/sage_student_finetune.yaml").open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw["data"].update(
        root=str(data), train_manifest=str(manifests/"train.csv"), dev_manifest=str(manifests/"dev.csv"),
        train_count=n_train, dev_count=n_dev, train_sha256=sha256(manifests/"train.csv"),
        dev_sha256=sha256(manifests/"dev.csv"))
    raw["model"].update(source_root=str(source), source_sha256=sha256(source/"model_sub/model.py"),
                        pretrained=str(source/"initial.pt"), pretrained_sha256=sha256(source/"initial.pt"))
    raw["training"].update(device="cpu", crop_hw=[8,12], cpu_threads=2)
    # Direct fixture Config only: production YAML parser enforces real 1045/175 counts.
    return Config(raw)
