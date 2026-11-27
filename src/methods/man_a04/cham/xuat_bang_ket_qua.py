"""27/11: results_overall.csv + results_per_class.csv cho reports/experiments/man_a04 (từ test_nham_last.npz, 15x15/ảnh).
mIoU_co_mat theo chung.cac_miou (bỏ nhãn 0, lớp có mặt). python ../scene_v1/xuat_bang_ket_qua.py <thư mục runs> <thư mục ra>  (từ b3)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from chung import TEN_LOP15, cac_miou

RUNS, RA = Path(sys.argv[1]), Path(sys.argv[2])
BT, SEEDS = ["tbsua_a04", "tb", "sage", "nosam"], range(5)
tong, lop = [], {}
for b in BT:
    for s in SEEDS:
        p = RUNS / f"{b}_b3_s{s}_1220_15lop" / "test_nham_last.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for ten, chon in (("273", ~z["leak"].astype(bool)), ("280", np.ones(len(z["leak"]), bool))):
            kq, iou, _ = cac_miou(z["conf"][chon].sum(0))
            tong.append({"bien_the": b, "seed": s, "tap": ten, "so_anh": int(chon.sum()),
                         "mIoU_13": round(kq["mIoU_co_mat"], 3), "mIoU_7": round(kq["mIoU_7"], 3), "pixel_acc": round(kq["pixel_acc"], 3)})
            if ten == "273":
                lop.setdefault(b, []).append(iou)
df = pd.DataFrame(tong)
df.to_csv(RA / "results_overall.csv", index=False)
ten_lop = TEN_LOP15[-len(lop["tb"][0]):]
pc = pd.DataFrame({b: np.nanmean(np.stack(v), 0) for b, v in lop.items()}, index=ten_lop).round(2)
pc.insert(0, "so_seed_a04_tb_sage", 5)
pc["d_a04_vs_tb"] = (pc.tbsua_a04 - pc.tb).round(2)
pc["d_a04_vs_sage"] = (pc.tbsua_a04 - pc.sage).round(2)
pc.index.name = "lop"
pc.to_csv(RA / "results_per_class.csv")
print(df.groupby(["tap", "bien_the"]).mIoU_13.agg(["mean", "std", "count"]).round(2)); print(pc)
