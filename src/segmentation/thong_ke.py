"""Bootstrap theo khối ảnh liền nhau (ảnh FMB cắt từ video: ảnh liền số gần như giống nhau).

thong_tin_anh_test.csv (280 dòng, đúng thứ tự data/manifests/test.csv):
  khoi   : số khối – các id liền nhau (id sau = id trước + 1) cùng một khối
  y_mean : độ sáng trung bình kênh Y của ảnh VIS gốc (dùng chia tối / vừa / sáng theo tam phân vị trên 273 ảnh)
  ir_mean: độ sáng trung bình ảnh IR
  leak   : 7 ảnh test rò (khung liền kề ảnh train, xem data/audit/5_anh_test_ro_da_chot.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 2000, 0
DAY = Path(__file__).resolve().parent


def doc_thong_tin_anh():
    return pd.read_csv(DAY / "thong_tin_anh_test.csv", dtype={"image_id": str})


def boot_idx(khoi, rng):
    """Sinh B bộ chỉ số ảnh, bốc lại theo khối (có hoàn lại)."""
    ks = np.unique(khoi)
    nhom = [np.flatnonzero(khoi == k) for k in ks]
    for _ in range(B):
        yield np.concatenate([nhom[i] for i in rng.integers(len(ks), size=len(ks))])


def ci(v):
    """Khoảng tin cậy 95% kiểu percentile."""
    return np.percentile(v, [2.5, 97.5])
