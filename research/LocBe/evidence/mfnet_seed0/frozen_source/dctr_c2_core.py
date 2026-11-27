"""CS406 DCTR-C2 v0.1 — deterministic VIS edge-deficit / reliability residual.

Based on Quân's review: C2 ONLY (no thermal residual, no conflict gate).
Input: VIS BGR, SAGE BGR (the frozen SAGE output is generated from VIS+IR).
No GT access during fusion. No Dev feedback in parameter selection.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class C2Config:
    alpha: float = 0.20  # same alpha as H1, NOT tuned on Dev
    sigma_hpf: float = 1.20  # same Gaussian scale as H1
    sigma_tensor: float = 1.00  # structure-tensor local coherence, predeclared
    eps_gradient: float = 0.02  # normalized Sobel magnitude stabilization
    eps_tensor: float = 1e-4
    reliability_scale: float = 0.06  # smooth edge signal-to-noise surrogate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sobel(y: np.ndarray):
    gx = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return gx, gy, e


def c2_fuse(vis_bgr: np.ndarray, sage_bgr: np.ndarray,
            cfg: C2Config = C2Config(), return_masks: bool = False):
    assert vis_bgr.dtype == np.uint8 and sage_bgr.dtype == np.uint8
    assert vis_bgr.ndim == 3 and vis_bgr.shape == sage_bgr.shape and vis_bgr.shape[2] == 3
    v = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)
    s = cv2.cvtColor(sage_bgr, cv2.COLOR_BGR2YCrCb)
    yv = v[:, :, 0].astype(np.float32) / 255.0
    ys = s[:, :, 0].astype(np.float32) / 255.0
    gx, gy, ev = sobel(yv)
    _, _, es = sobel(ys)

    # Deficit: only when VIS local gradient exceeds SAGE local gradient.
    # No segmentation GT, class IDs or Dev-selected thresholds.
    m_def = np.clip((ev - es) / (ev + cfg.eps_gradient), 0.0, 1.0)

    # VIS reliability: local oriented-structure coherence, discounted when
    # gradient magnitude is weak. This is a testable proxy, not learned confidence.
    sig = cfg.sigma_tensor
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    coherence = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4 * jxy ** 2)) / (jxx + jyy + cfg.eps_tensor)
    m_rel = np.clip(coherence, 0, 1) * (ev / (ev + cfg.reliability_scale))
    mask = np.clip(m_def * m_rel, 0, 1)

    blur = cv2.GaussianBlur(yv, (0, 0), cfg.sigma_hpf)
    high = yv - blur
    delta = cfg.alpha * mask * high
    proposed = ys + delta
    clipped = np.clip(proposed, 0.0, 1.0)
    # Explicit round-to-nearest uint8 and preserve SAGE's own chroma.
    y_u8 = np.rint(clipped * 255.0).astype(np.uint8)
    fused_ycc = s.copy()
    fused_ycc[:, :, 0] = y_u8
    reconstructed = cv2.cvtColor(fused_ycc, cv2.COLOR_YCrCb2BGR)
    # Preserve byte-identical original SAGE BGR wherever quantization does not change Y.
    y_changed = (y_u8 != s[:, :, 0])
    out = np.where(y_changed[:, :, None], reconstructed, sage_bgr)
    assert out.dtype == np.uint8 and out.shape == sage_bgr.shape

    if not return_masks:
        return out
    _, _, eout = sobel(y_u8.astype(np.float32)/255.0)
    changed = np.any(out != sage_bgr, axis=2)
    # Proxy clipping rate, not percent of pixels ruined.
    clip_rate = float(np.mean((proposed < 0) | (proposed > 1)))
    high_def = m_def > 0.20  # diagnostic-only threshold, never used by fusion
    low_def = m_def < 0.05   # diagnostic-only threshold, never used by fusion
    absy = np.abs(y_u8.astype(np.int16) - s[:,:,0].astype(np.int16)).astype(np.float32)
    info = {
        'mean_deficit':float(m_def.mean()), 'mean_reliability':float(m_rel.mean()),
        'mean_effective_gate':float(mask.mean()),
        'changed_rgb_pixel_fraction':float(changed.mean()),
        'changed_y_pixel_fraction':float(np.mean(absy > 0)),
        'mae_rgb_0_255':float(np.mean(np.abs(out.astype(np.float32)-sage_bgr.astype(np.float32)))),
        'mean_abs_y_lsb':float(absy.mean()),
        'change_y_lsb_deficit_region':float(absy[high_def].mean()) if np.any(high_def) else 0.0,
        'change_y_lsb_nondeficit_region':float(absy[low_def].mean()) if np.any(low_def) else 0.0,
        'edge_ratio_sage_to_vis':float(es.mean()/(ev.mean()+1e-8)),
        'edge_ratio_c2_to_vis':float(eout.mean()/(ev.mean()+1e-8)),
        'clip_rate':clip_rate,
        'masked_fraction_deficit_gt_0_20':float(high_def.mean()),
    }
    return out, info, (m_def, m_rel, mask, high)


def synthetic_self_test():
    """CPU smoke for shape, determinism, provenance and zero-GT fusion."""
    yy, xx = np.mgrid[:128, :192]
    base = np.zeros((128,192,3),dtype=np.uint8)
    base[:, :] = [25,50,90]
    base[20:106, 30:150] = [220,190,180]
    base[45:65, 65:110] = [10,240,90]
    sage = cv2.GaussianBlur(base,(0,0),2.4)
    out, info, (d,r,g,h) = c2_fuse(base,sage,return_masks=True)
    out2=c2_fuse(base,sage)
    assert np.array_equal(out,out2)
    assert np.isfinite(out).all() and np.isfinite(d).all() and np.isfinite(r).all()
    assert np.all((d>=0)&(d<=1)) and np.all((r>=0)&(r<=1)) and np.all((g>=0)&(g<=1))
    assert info['changed_y_pixel_fraction'] > 0, info
    assert info['changed_rgb_pixel_fraction'] > 0, info
    # On an exact SAGE=VIS match, no edge deficit means no changes to the Y channel.
    out_eq, info_eq, masks_eq=c2_fuse(base,base,return_masks=True)
    assert np.allclose(masks_eq[0], 0, atol=1e-7)
    assert np.array_equal(out_eq, base), 'SAGE=VIS must be byte-identical when deficit is zero'
    return {'status':'PASS','demo_changed_rgb_fraction':info['changed_rgb_pixel_fraction'],
            'demo_changed_y_fraction':info['changed_y_pixel_fraction']}

if __name__ == '__main__':
    print(json.dumps(synthetic_self_test(),indent=2))