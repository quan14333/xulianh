"""So cặp mIoU chỉ trên các lớp có ĐỦ ẢNH trong tập chấm (lớp 1–2 ảnh làm IoU thành nhiễu).

Lớp được giữ: số ảnh có lớp đó trong GT >= --min_anh (đếm từ ma trận nhầm lẫn từng ảnh, cố định
trên tập gốc, không đếm lại trong mỗi vòng bootstrap). Bootstrap khối ảnh + seed giống so_sanh_cap.py.

Ví dụ:
  SAGE_RUNS=../scene_v1/runs_b0_gop python so_sanh_lop_du.py --bien_the nosam_kp841 sage tb --model b0 --tap dev
  SAGE_RUNS=../_kaggle_out/b3kp_gop python so_sanh_lop_du.py --bien_the nosam_kp841 sage tb --model b3 --tap dev --hau _15lop --seeds 0
"""
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from chung import RUNS, TEN_LOP, cac_miou
from so_sanh_cap import B, SEED_BOOT, khoi_anh, muc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--model", default="b3")
    ap.add_argument("--hau", default="")
    ap.add_argument("--tap", default="dev")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--cap", nargs="*")
    ap.add_argument("--min_anh", nargs="+", type=int, default=[20, 10, 1])
    ap.add_argument("--giu_ro", action="store_true")
    a = ap.parse_args()

    def duong(b, s):
        return RUNS / f"{b}_{a.model}_s{s}{a.hau}" / f"{a.tap}_nham_{a.ckpt}.npz"

    seeds = a.seeds or sorted({int(p.name.split("_s")[-1].split("_")[0]) for b in a.bien_the
                               for p in RUNS.glob(f"{b}_{a.model}_s*{a.hau}") if p.is_dir()})
    seeds = [s for s in seeds if all(duong(b, s).exists() for b in a.bien_the)]
    assert seeds, "không có seed nào đủ mọi biến thể"
    cap = [tuple(c.split("-", 1)) for c in a.cap] if a.cap else list(itertools.combinations(a.bien_the, 2))

    C, ids = {}, None
    for s in seeds:
        for b in a.bien_the:
            z = np.load(duong(b, s))
            if ids is None:
                ids, leak = [str(i) for i in z["image_id"]], z["leak"].astype(bool)
            assert [str(i) for i in z["image_id"]] == ids
            C[(s, b)] = z["conf"]
    chon = np.ones(len(ids), bool) if a.giu_ro else ~leak
    for k in C:
        C[k] = C[k][chon]
    ids = [i for i, c in zip(ids, chon) if c]
    nhom = khoi_anh(ids)

    # số ảnh có mỗi lớp (14 lớp 1..14) trong GT
    c0 = C[(seeds[0], a.bien_the[0])]
    g = c0.sum(2)                       # (ảnh, lớp GT)
    if c0.shape[1] == 15:
        g = g[:, 1:]
    so_anh = (g > 0).sum(0)
    print(f"{a.model} | tập {a.tap} | {len(ids)} ảnh | {len(nhom)} khối | seed {seeds}")
    print("Số ảnh mỗi lớp:", ", ".join(f"{t} {n}" for t, n in zip(TEN_LOP, so_anh)))

    iou = lambda c: cac_miou(c)[1]       # 14 IoU, bỏ pixel unlabeled
    rng = np.random.default_rng(SEED_BOOT)
    k = len(nhom)
    mau = []
    for _ in range(B):
        i = np.concatenate([nhom[j] for j in rng.integers(k, size=k)])
        mau.append((i, rng.choice(seeds, size=len(seeds), replace=True)))

    tat_ca = []
    for m in a.min_anh:
        giu = so_anh >= m
        print(f"\n===== Lớp có >= {m} ảnh: {int(giu.sum())} lớp ({', '.join(t for t, x in zip(TEN_LOP, giu) if x)})")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = lambda c: float(np.nanmean(iou(c)[giu]))
            goc = {(s, b): mi(C[(s, b)].sum(0)) for s in seeds for b in a.bien_the}
            bang = pd.DataFrame({f"s{s}": [goc[(s, b)] for b in a.bien_the] for s in seeds}, index=a.bien_the)
            bang["TB"] = bang.mean(1)
            print(bang.round(2).to_string())
            d1 = {c: [] for c in cap}
            d2 = {c: [] for c in cap}
            for i, ss in mau:
                v = {(s, b): mi(C[(s, b)][i].sum(0)) for s in seeds for b in a.bien_the}
                for x, y in cap:
                    d1[(x, y)].append(np.mean([v[(s, x)] - v[(s, y)] for s in seeds]))
                    d2[(x, y)].append(np.mean([v[(s, x)] - v[(s, y)] for s in ss]))
        dong = []
        for x, y in cap:
            ds = [goc[(s, x)] - goc[(s, y)] for s in seeds]
            lo1, hi1 = np.percentile(d1[(x, y)], [2.5, 97.5])
            lo2, hi2 = np.percentile(d2[(x, y)], [2.5, 97.5])
            dong.append({"min_anh": m, "so_lop": int(giu.sum()), "cap": f"{x}−{y}", "d_tb": np.mean(ds),
                         **{f"d_s{s}": d for s, d in zip(seeds, ds)},
                         "CI_anh": f"[{lo1:+.2f}; {hi1:+.2f}]", "CI_anh_seed": f"[{lo2:+.2f}; {hi2:+.2f}]",
                         "ket_luan": muc(ds, lo1, hi1, lo2, hi2)})
        kq = pd.DataFrame(dong)
        print(kq.drop(columns=["min_anh", "so_lop"]).round(2).to_string(index=False))
        tat_ca.append(kq)
    ra = RUNS / f"so_sanh_lop_du_{a.tap}_{a.model}.csv"
    pd.concat(tat_ca).to_csv(ra, index=False)
    print("\nGhi", ra)


if __name__ == "__main__":
    main()
