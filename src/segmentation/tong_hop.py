"""Gộp kết quả mọi run thành một bảng, đặt cạnh số của bài SAGE (Bảng 1, SegFormer-B3).

Ra: {RUNS}/tong_hop.csv và in bảng.  Ví dụ: python tong_hop.py
"""
import json

import numpy as np
import pandas as pd

from chung import LOP_BAI, RUNS, TEN_LOP, cac_miou

# THEO BÀI (arXiv 2503.01210v2, Bảng 1, khối SegFormer-B3). Lấy từ bản HTML, cần dò lại với PDF.
# Thứ tự: road, building, traffic_sign, vegetation, bus, person, car, mIoU
BAI = {
    "vis": [84.8, 81.0, 64.2, 84.7, 46.8, 61.1, 77.1, 49.1],
    "ir": [83.7, 77.6, 55.9, 80.9, 38.1, 57.1, 74.2, 42.7],
    "sage": [89.0, 83.9, 71.4, 86.3, 66.7, 64.1, 81.7, 61.2],
}


def main():
    dong = []
    for f in sorted(RUNS.glob("*/test_ket_qua_*.json")):
        kq = json.loads(f.read_text(encoding="utf-8"))
        # Tính lại từ ma trận nhầm lẫn từng ảnh, để thêm quy ước mới không phải chấm lại
        z = np.load(f.with_name(f"test_nham_{kq['ckpt']}.npz"))
        for n, chon in (("280", np.ones(len(z["leak"]), bool)), ("273", ~z["leak"])):
            m, iou, _ = cac_miou(z["conf"][chon].sum(0))
            d = {"run": f.parent.name, "bien_the": kq["bien_the"], "seed": kq["seed"], "ckpt": kq["ckpt"],
                 "so_anh": n, "mIoU_co_mat": m["mIoU_co_mat"], "mIoU_14": m["mIoU_14"],
                 "mIoU_7": m["mIoU_7"], "mIoU_15_nan0": m.get("mIoU_15_nan0"),
                 "mIoU_15_mrfs": m.get("mIoU_15_mrfs"),
                 "mIoU_7_ca_nhan0": m.get("mIoU_7_ca_nhan0"), "IoU_unlabeled": m.get("IoU_unlabeled")}
            d |= dict(zip(TEN_LOP, iou))
            dong.append(d)
    if not dong:
        print("Chưa có run nào chấm xong.")
        return
    df = pd.DataFrame(dong).round(2)
    df.to_csv(RUNS / "tong_hop.csv", index=False)
    pd.set_option("display.width", 250)

    for n in ("280", "273"):
        d = df[df.so_anh == n]
        print(f"\n===== {n} ảnh test – mIoU (%) theo các quy ước =====")
        print("(co_mat, 14, 7: bỏ pixel unlabeled | 15_nan0 kiểu SegMiF, 15_mrfs kiểu MRFS, 7_ca_nhan0: tính unlabeled là một lớp)")
        print(d[["run", "ckpt", "mIoU_co_mat", "mIoU_14", "mIoU_7", "mIoU_15_nan0", "mIoU_15_mrfs", "mIoU_7_ca_nhan0",
                 "IoU_unlabeled"]].to_string(index=False))
    d = df[df.so_anh == "280"]
    print("\n===== 280 ảnh – IoU đủ 14 lớp =====")
    print(d.set_index(d.run + "/" + d.ckpt)[TEN_LOP].T.round(1).to_string())

    print("\n===== So với bài (7 lớp bài in ra + mIoU của bài), 280 ảnh =====")
    bang = {}
    for bt, so in BAI.items():
        bang[f"BÀI {bt}"] = so[:7] + [sum(so[:7]) / 7, so[7]]
    for _, r in d.iterrows():
        co15 = pd.notna(r.mIoU_15_nan0)
        bang[f"{r.run}/{r.ckpt}"] = ([r[t] for t in LOP_BAI]
                                     + [r.mIoU_7, r.mIoU_15_nan0 if co15 else r.mIoU_co_mat])
    print(pd.DataFrame(bang, index=LOP_BAI + ["TB 7 lớp", "mIoU (*)"]).round(1).to_string())
    print("(*) Bài không nói mIoU tính trên lớp nào. Cột của nhóm: mIoU_15_nan0 (kiểu SegMiF) nếu model 15 lớp,\n"
          "    không thì mIoU_co_mat. IoU 7 lớp ở bảng này là loại bỏ pixel unlabeled.")

    tb = d[d.ckpt == "last"].groupby("bien_the")[["mIoU_co_mat", "mIoU_14", "mIoU_7", "mIoU_15_nan0"]].mean()
    if {"sage", "vis"} <= set(tb.index):
        c = tb.loc["sage"] - tb.loc["vis"]
        print(f"\nSAGE − VIS (last, TB các seed): có mặt {c.mIoU_co_mat:+.2f} | 14 lớp {c.mIoU_14:+.2f} | "
              f"7 lớp {c.mIoU_7:+.2f} | 15 lớp kiểu SegMiF {c.mIoU_15_nan0:+.2f}    | BÀI: mIoU +12,1 ; 7 lớp +6,2")
        print("Tiêu chí đặt trước (mIoU): ≥ +8 tái lập được khoảng cách của bài; ≤ +4 không tái lập được.")


if __name__ == "__main__":
    main()
