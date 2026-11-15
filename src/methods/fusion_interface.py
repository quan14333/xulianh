"""Shared Development fusion contract: BGR sources -> RGB uint8 prediction."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

import numpy as np


class WaitingForMethodSpec(RuntimeError):
    status = "WAITING_FOR_METHOD_SPEC"


@dataclass(frozen=True)
class FusionPair:
    image_id: str
    vis_bgr: np.ndarray
    ir_bgr: np.ndarray
    source_split: str = "train"

    def validate(self):
        if not isinstance(self.image_id, str) or not re.fullmatch(r"\d{5}", self.image_id):
            raise ValueError("image_id must retain five decimal digits")
        if self.source_split != "train":
            raise PermissionError("Only official-Train sources are enabled in Development")
        if not isinstance(self.vis_bgr, np.ndarray) or not isinstance(self.ir_bgr, np.ndarray):
            raise ValueError("Source images must be NumPy arrays")
        if (self.vis_bgr.shape != self.ir_bgr.shape or self.vis_bgr.ndim != 3
                or self.vis_bgr.shape[2] != 3 or min(self.vis_bgr.shape[:2]) < 4
                or self.vis_bgr.dtype != np.uint8 or self.ir_bgr.dtype != np.uint8):
            raise ValueError("Aligned HWC3 BGR uint8 sources, H/W>=4, are required")


@dataclass(frozen=True)
class FusionPrediction:
    rgb: np.ndarray
    metadata: dict

    def validate(self, pair):
        pair.validate()
        if self.rgb.shape != pair.vis_bgr.shape or self.rgb.dtype != np.uint8:
            raise ValueError("Prediction must be full-resolution HWC3 RGB uint8")


class FusionMethod(Protocol):
    identity: dict

    def fuse(self, pair: FusionPair) -> FusionPrediction:
        """Return RGB; export adapters perform RGB->BGR only at the PNG writer boundary."""
        ...
