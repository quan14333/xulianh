"""Frozen DCTR-C3-T fusion. Direct transcription of the evaluated Kaggle notebook.

All images are uint8 OpenCV BGR of equal dimensions. No segmentation GT is used.
C3-T is a *fusion RGB image generator*, not a segmentation model.
"""
from __future__ import annotations
import cv2
import numpy as np
from dctr_c2_core import C2Config, c2_fuse

C3T_THERMAL_THRESHOLD = 0.30
# 'lambda_thermal' and 'lambda_conflict' were present in the frozen notebook config
# but are NOT read by its C3-T guard; leave them solely for provenance.
C3T_PARAMS = {'sigma_smooth': 1.0, 'edge_percentile': 90.0,
              'lambda_thermal': 0.75, 'lambda_conflict': 0.50, 'eps': 1e-8}

def grad_norm(im, p):
    im = cv2.GaussianBlur(im, (0, 0), float(p['sigma_smooth']))
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    scale = max(float(np.percentile(mag, p['edge_percentile'])), float(p['eps']))
    return gx, gy, mag, np.clip(mag/scale, 0, 1)

def signal_maps(vis, ir, p=C3T_PARAMS):
    gv = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    gi = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    vx, vy, vm, vs = grad_norm(gv, p)
    ix, iy, im, ins = grad_norm(gi, p)
    thermal = np.maximum(ins - vs, 0)
    cos = np.clip((vx*ix + vy*iy)/(vm*im + p['eps']), -1, 1)
    conflict = (1 - np.abs(cos))*np.minimum(ins, vs)
    return thermal, conflict

def c3t_fuse(vis_bgr, ir_bgr, sage_bgr, c2_bgr=None, return_mask=False):
    for img in (vis_bgr, ir_bgr, sage_bgr):
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Expected uint8 BGR images (H,W,3)')
    if not (vis_bgr.shape == ir_bgr.shape == sage_bgr.shape):
        raise ValueError('VIS/IR/SAGE image shapes differ')
    if c2_bgr is None:
        c2_bgr = c2_fuse(vis_bgr, sage_bgr, C2Config())
    if c2_bgr.shape != sage_bgr.shape or c2_bgr.dtype != np.uint8:
        raise ValueError('C2 BGR shape/dtype invalid')
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
    thermal, conflict = signal_maps(vis_rgb, ir_rgb)
    flag = thermal >= C3T_THERMAL_THRESHOLD
    out = c2_bgr.copy()
    out[flag] = sage_bgr[flag]
    assert np.array_equal(out[~flag], c2_bgr[~flag])
    assert np.array_equal(out[flag], sage_bgr[flag])
    return (out, flag) if return_mask else out
