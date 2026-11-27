"""So cặp phương án THEO CẶP SEED (cùng seed segmentation), có CI. Dùng chung cho mọi bộ run (B0/B3, test/dev/val).

Vì sao: chênh giữa các phương án (~1 điểm mIoU) cỡ bằng dao động seed. So "TB 3 seed A vs TB 3 seed B" thiếu CI của
HIỆU SỐ. Ở đây mỗi seed s cho một hiệu d_s = mIoU(A, s) − mIoU(B, s) trên CÙNG bộ ảnh, rồi:
  - d_tb      : trung bình d_s qua seed (số chính).
  - CI_anh    : bootstrap theo KHỐI ẢNH (10 ảnh id liền nhau, như cong2_ci.py), cùng bộ ảnh cho mọi seed và mọi
                phương án; chỉ tính biến thiên do chọn ảnh (giống phần GỘP SEED của so_sanh_b3.py).
  - CI_anh_seed: bootstrap 2 tầng – bốc lại seed (có hoàn lại) + bốc lại khối ảnh. Tính cả biến thiên do seed.
                Với 2–3 seed, tầng seed rất thô (chỉ có 3^3 cách bốc) → xem kèm d theo từng seed.
  - so_seed_duong: số seed có d_s > 0.
Thang 3 mức (Mân chốt 18/11, xem tai_lap/QUY_TAC_CHAM.md):
  chắc chắn      : cận dưới CI_anh_seed > 0
  nhất quán      : d_tb > 0, MỌI seed d_s > 0, cận dưới CI_anh > 0
  chưa phân biệt : còn lại (đối xứng cho chiều kém)

Số phụ (không dùng để chọn phương án): mIoU 7 lớp của bài, IoU từng lớp (trung bình seed).

Tên run: {bien_the}_{model}_s{seed}{hau}, ví dụ sage_b3_s0_1220_15lop (hau = _1220_15lop) hoặc tb_b0_s1 (hau rỗng).
File cần trong mỗi run: {tap}_nham_{ckpt}.npz (eval_seg.py --tap {tap}).

Ví dụ (B3 test 280, bỏ 7 ảnh rò):
  SAGE_RUNS=../_kaggle_out/runs_gop python so_sanh_cap.py --bien_the vis tb sage nosam --model b3 --hau _1220_15lop
Ví dụ (B0 val 151 của kham_pha):
  SAGE_RUNS=runs python so_sanh_cap.py --bien_the tb sage nosam_kp841 m11_kp --model b0 --tap dev --cap m11_kp-tb m11_kp-sage
Ra: in bảng + ghi {RUNS}/so_sanh_cap_{tap}_{model}.csv
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from chung import RUNS, TEN_LOP15, cac_miou

B, SEED_BOOT, CO_KHOI = 2000, 0, 10


def khoi_anh(ids):
    """Khối liền (id liên tiếp) rồi cắt mỗi khối thành khối con 10 ảnh. Trả list chỉ số theo khối."""
    so = np.array([int(i) for i in ids])
    thu_tu = np.argsort(so)
    lien = np.r_[0, np.cumsum(np.diff(so[thu_tu]) != 1)]
    nhom = []
    for g in np.unique(lien):
        vt = thu_tu[lien == g]
        nhom += [vt[j:j + CO_KHOI] for j in range(0, len(vt), CO_KHOI)]
    return nhom


def muc(ds, lo1, hi1, lo2, hi2):
    """Thang 3 mức (QUY_TAC_CHAM.md)."""
    ds = np.asarray(ds)
    if lo2 > 0:
        return "hơn chắc chắn"
    if hi2 < 0:
        return "kém chắc chắn"
    if ds.mean() > 0 and (ds > 0).all() and lo1 > 0:
        return "hơn nhất quán"
    if ds.mean() < 0 and (ds < 0).all() and hi1 < 0:
        return "kém nhất quán"
    return "chưa phân biệt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--model", default="b3")
    ap.add_argument("--hau", default="", help="hậu tố tên run sau _s{seed}, vd _1220_15lop")
    ap.add_argument("--tap", default="test")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--seeds", nargs="*", type=int, help="mặc định: mọi seed có đủ tất cả biến thể")
    ap.add_argument("--cap", nargs="*", help="các cặp A-B; mặc định mọi cặp")
    ap.add_argument("--giu_ro", action="store_true", help="giữ 7 ảnh test rò (mặc định bỏ)")
    a = ap.parse_args()

    def duong(b, s):
        return RUNS / f"{b}_{a.model}_s{s}{a.hau}" / f"{a.tap}_nham_{a.ckpt}.npz"

    seeds = a.seeds or sorted({int(p.name.split("_s")[-1].split("_")[0]) for b in a.bien_the
                               for p in RUNS.glob(f"{b}_{a.model}_s*{a.hau}") if p.is_dir()})
    seeds = [s for s in seeds if all(duong(b, s).exists() for b in a.bien_the)]
    assert seeds, "không có seed nào đủ mọi biến thể"
    cap = [tuple(c.split("-", 1)) for c in a.cap] if a.cap else list(itertools.combinations(a.bien_the, 2))
    for x, y in cap:
        assert x in a.bien_the and y in a.bien_the, (x, y)

    C, ids = {}, None
    for s in seeds:
        for b in a.bien_the:
            z = np.load(duong(b, s))
            if ids is None:
                ids, leak = [str(i) for i in z["image_id"]], z["leak"].astype(bool)
            assert [str(i) for i in z["image_id"]] == ids, f"{b} s{s}: bộ ảnh khác"
            C[(s, b)] = z["conf"]
    chon = np.ones(len(ids), bool) if a.giu_ro else ~leak
    for k in C:
        C[k] = C[k][chon]
    ids = [i for i, c in zip(ids, chon) if c]
    nhom = khoi_anh(ids)
    print(f"{a.model} | tập {a.tap} | {len(ids)} ảnh ({'giữ' if a.giu_ro else 'bỏ'} ảnh rò: {int(leak.sum())}) | "
          f"{len(nhom)} khối | seed {seeds} | ckpt {a.ckpt}")

    mi = lambda c: cac_miou(c)[0]["mIoU_co_mat"]
    mi7 = lambda c: cac_miou(c)[0]["mIoU_7"]

    # 1) mIoU từng seed
    goc = {(s, b): mi(C[(s, b)].sum(0)) for s in seeds for b in a.bien_the}
    bang = pd.DataFrame({f"s{s}": [goc[(s, b)] for b in a.bien_the] for s in seeds}, index=a.bien_the)
    bang["TB"] = bang.mean(1)
    bang["SD"] = bang.iloc[:, :len(seeds)].std(1, ddof=1) if len(seeds) > 1 else np.nan
    bang["mIoU_7_TB"] = [np.mean([mi7(C[(s, b)].sum(0)) for s in seeds]) for b in a.bien_the]
    print("\n== mIoU_co_mat theo seed")
    print(bang.round(2).to_string())

    # 2) bootstrap: cùng bộ khối ảnh (và cùng bộ seed) cho mọi phương án
    rng = np.random.default_rng(SEED_BOOT)
    k = len(nhom)
    d_anh = {c: [] for c in cap}
    d_2t = {c: [] for c in cap}
    for _ in range(B):
        i = np.concatenate([nhom[j] for j in rng.integers(k, size=k)])
        v = {(s, b): mi(C[(s, b)][i].sum(0)) for s in seeds for b in a.bien_the}
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        for x, y in cap:
            d_anh[(x, y)].append(np.mean([v[(s, x)] - v[(s, y)] for s in seeds]))
            d_2t[(x, y)].append(np.mean([v[(s, x)] - v[(s, y)] for s in ss]))

    dong = []
    for x, y in cap:
        ds = [goc[(s, x)] - goc[(s, y)] for s in seeds]
        lo1, hi1 = np.percentile(d_anh[(x, y)], [2.5, 97.5])
        lo2, hi2 = np.percentile(d_2t[(x, y)], [2.5, 97.5])
        d7 = np.mean([mi7(C[(s, x)].sum(0)) - mi7(C[(s, y)].sum(0)) for s in seeds])
        dong.append({"cap": f"{x}−{y}", "d_tb": np.mean(ds), **{f"d_s{s}": d for s, d in zip(seeds, ds)},
                     "CI_anh_lo": lo1, "CI_anh_hi": hi1, "CI_anh_seed_lo": lo2, "CI_anh_seed_hi": hi2,
                     "so_seed_duong": int(np.sum(np.array(ds) > 0)), "d_mIoU7_tb": d7,
                     "ket_luan": muc(ds, lo1, hi1, lo2, hi2)})
    kq = pd.DataFrame(dong)
    print(f"\n== Hiệu số theo cặp seed (B = {B}); kết luận theo thang 3 mức")
    print(kq.round(2).to_string(index=False))

    # 3) IoU từng lớp (TB seed) – số phụ
    warnings.filterwarnings("ignore", "Mean of empty slice")   # bicycle: không có trong test
    iou = {b: np.nanmean([cac_miou(C[(s, b)].sum(0))[1] for s in seeds], axis=0) for b in a.bien_the}
    lop = pd.DataFrame(iou, index=TEN_LOP15[1:])
    print("\n== IoU từng lớp (TB seed, bỏ pixel unlabeled) – số phụ")
    print(lop.round(1).to_string())

    ra = RUNS / f"so_sanh_cap_{a.tap}_{a.model}.csv"
    kq.to_csv(ra, index=False)
    print(f"\nGhi {ra}")


if __name__ == "__main__":
    main()
