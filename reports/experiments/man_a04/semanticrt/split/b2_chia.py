"""Bước 2: chia lại SemanticRT theo NHÓM GÓC MÁY + lấy mẫu bớt (bộ kiểm tổng quát zero-shot cho mô hình ghép).

LUẬT (đặt trước, không nhìn kết quả mô hình nào):
 1. Chỉ giữ ảnh 1280x1024 (bỏ 271 ảnh nhỏ của camera khác).
 2. Gộp cụm góc máy thành ĐỊA ĐIỂM (xem khối NHOM bên dưới; cụm cũ trộn nhiều địa điểm, cùng phố quay ngày/đêm/zoom khác).
 3. TEST = cụm {5, 12, 19, 20} (không nối địa điểm với cụm nào). Đổi TRƯỚC khi có kết quả mô hình nào.
 4. Bỏ gần trùng trong cùng góc máy: duyệt ngẫu nhiên (seed 0), giữ ảnh nếu cos với mọi ảnh đã giữ cùng máy < 0,95
    (đặc trưng cạnh Sobel 48x40 của semanticrt_dactrung_bien.npy).
 5. Kiểm rò từng ảnh: bỏ ảnh test có láng giềng cos > 0,80 trong TOÀN BỘ ảnh thuộc nhóm train (trước khi lấy mẫu).
 6. Lấy mẫu ngẫu nhiên (seed 0) từ ảnh còn lại: train 1.000, test 300, chia theo tỉ lệ số ảnh mỗi nhóm.
Ra: train.csv, test.csv, README.md (số liệu + ô kiểm).
Chạy: python b2_chia.py   (cần thong_tin.csv của bước 1; dùng GPU nếu có)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DAY = Path(__file__).resolve().parent
NGOAI = DAY.parent
GOC = NGOAI / "semanticrt" / "SemanticRT_dataset"
# Bản 3 (22/11): gom ĐỊA ĐIỂM = hợp mọi bằng chứng: liên kết cạnh (cos > 0,8), ORB theo cụm (b3, điểm >= 8, nền ~5),
# xem mắt (7~15, 7~6, 7~2, 2~15, 10~4 ngày/đêm cùng phố). Thành phần lớn G = mọi cụm trừ {5, 12, 19, 20}.
# 4 cụm này không có cặp ORB >= 8 với cụm nào -> TEST. (Bản 1 {2,6,10,12}, bản 2 {2,7,10,12} đều trùng địa điểm train.)
G = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 21}
NHOM = {m: "G" for m in G}
TEST = {"5", "12", "19", "20"}
NGUONG_TRUNG, NGUONG_RO = 0.95, 0.80
SO = {"train": 1000, "test": 300}
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DAY / "thong_tin.csv")
F = torch.tensor(np.load(NGOAI / "semanticrt_dactrung_bien.npy"), device=dev)
F = F / F.norm(dim=1, keepdim=True)
m = json.load(open(NGOAI / "semanticrt_may.json", encoding="utf-8"))
assert m["ten"] == df.ten.tolist()
df["hang"] = np.arange(len(df))
df["nhom"] = df.may.map(lambda x: NHOM.get(x, str(x)))
df["split"] = np.where(df.nhom.isin(TEST), "test", "train")
log = {"tong": len(df)}

df = df[(df.rong == 1280) & (df.cao == 1024)].copy()
log["sau_bo_anh_nho"] = len(df)

# 4. bỏ gần trùng trong từng góc máy
giu = []
for may, g in df.groupby("may"):
    hang = rng.permutation(g.hang.values)
    da = []
    for h in hang:
        if not da or (F[da] @ F[h]).max().item() < NGUONG_TRUNG:
            da.append(h)
    giu += da
df = df[df.hang.isin(giu)].copy()
log["sau_bo_gan_trung"] = df.split.value_counts().to_dict()

# 5. kiểm rò: ảnh test vs TOÀN BỘ ảnh nhóm train (kể cả ảnh nhỏ, ảnh gần trùng đã bỏ)
tat_ca = pd.read_csv(DAY / "thong_tin.csv")
tat_ca["nhom"] = tat_ca.may.map(lambda x: NHOM.get(x, str(x)))
h_train_all = torch.tensor(np.where(~tat_ca.nhom.isin(TEST))[0], device=dev)
te = df[df.split == "test"]
s_max = (F[torch.tensor(te.hang.values, device=dev)] @ F[h_train_all].T).max(1).values.cpu().numpy()
df.loc[te.index, "cos_train_max"] = s_max
ro = te.index[s_max > NGUONG_RO]
log["test_bo_vi_ro"] = int(len(ro))
df = df.drop(ro)
df[df.split == "test"][["ten", "may", "nhom", "hang"]].to_csv(DAY / "ung_vien_test.csv", index=False)

# 5b. kiểm rò bằng ORB (b5_kiem_orb.py, 22/11): đặc trưng cạnh không nhận ra cùng địa điểm khi camera lệch góc/zoom.
#     Bỏ ảnh test có >= NGUONG_ORB inlier với một trong 30 ảnh train giống nhất (khác cảnh ~5, cùng địa điểm 47–238).
NGUONG_ORB = 15   # 22/11: hạ từ 25 -> 15 sau khi xem mắt ca 18 inlier (img_03649, có thể cùng góc phố nhìn hướng khác); khoảng 10–24 chỉ có ca này
f_orb = DAY / "ro_orb.csv"
if f_orb.exists():
    orb = pd.read_csv(f_orb)
    xau = set(orb.loc[orb.inlier_max >= NGUONG_ORB, "ten"])
    log["test_bo_vi_orb"] = int(df.ten.isin(xau).sum())
    log["test_bo_vi_orb_theo_may"] = df[df.ten.isin(xau)].may.value_counts().to_dict()
    df = df[~df.ten.isin(xau)]
else:
    print("CHƯA có ro_orb.csv -> chạy b5_kiem_orb.py rồi chạy lại b2")

# 6. lấy mẫu theo tỉ lệ nhóm
ra = []
for sp, n in SO.items():
    d = df[df.split == sp]
    chon = d.sample(n=n, random_state=0) if len(d) >= n else d
    ra.append(chon)
out = pd.concat(ra).sort_values(["split", "ten"])
out["image_id"] = out.ten
out["vis_path"] = "rgb/" + out.ten + ".jpg"
out["ir_path"] = "thermal/" + out.ten + ".jpg"
out["label_path"] = "labels/" + out.ten + ".png"
out["dem"] = out.sang_vis < 60
cot = ["image_id", "split", "vis_path", "ir_path", "label_path", "may", "nhom", "split_goc", "dem", "sang_vis",
       "cos_train_max"] + [f"px_{c}" for c in range(13)]
for sp in SO:
    out[out.split == sp][cot].to_csv(DAY / f"{sp}.csv", index=False)

# ===== Ô KIỂM =====
tr, ts = out[out.split == "train"], out[out.split == "test"]
kiem = {
    "giao train∩test = rỗng (id)": len(set(tr.image_id) & set(ts.image_id)) == 0,
    "giao NHÓM MÁY train∩test = rỗng": len(set(tr.nhom) & set(ts.nhom)) == 0,
    "không id trùng trong mỗi file": tr.image_id.is_unique and ts.image_id.is_unique,
    "mọi file tồn tại": all((GOC / p).exists() for c in ("vis_path", "ir_path", "label_path") for p in out[c]),
    f"mọi ảnh test có cos với train < {NGUONG_RO}": bool((ts.cos_train_max <= NGUONG_RO).all()),
    "đủ số ảnh": len(tr) == SO["train"] and len(ts) == SO["test"],
}
if f_orb.exists():
    kiem[f"mọi ảnh test có ORB inlier với train < {NGUONG_ORB}"] = bool(ts.image_id.isin(orb.loc[orb.inlier_max < NGUONG_ORB, "ten"]).all())
px = lambda d: (d[[f"px_{c}" for c in range(13)]] > 0).sum().tolist()
log.update(train=len(tr), test=len(ts), test_dem=float(ts.dem.mean()), train_dem=float(tr.dem.mean()),
           nhom_test=sorted(ts.nhom.unique()), so_anh_co_lop_train=px(tr), so_anh_co_lop_test=px(ts),
           cos_test_train_max=float(ts.cos_train_max.max()), split_goc_test=ts.split_goc.value_counts().to_dict())
for k, v in kiem.items():
    print(("OK  " if v else "LỖI ") + k)
print(json.dumps(log, ensure_ascii=False, indent=1))
json.dump({"kiem": kiem, **log}, open(DAY / "chia_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
assert all(kiem.values())
