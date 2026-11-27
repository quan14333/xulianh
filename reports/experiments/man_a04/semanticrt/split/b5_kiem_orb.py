"""Bước 5: kiểm rò ảnh test ứng viên bằng ORB + RANSAC (chịu được camera lệch góc/zoom).
Mỗi ảnh trong ung_vien_test.csv (do b2 ghi): lấy TOP_K ảnh giống nhất (cos đặc trưng cạnh) trong TOÀN BỘ ảnh
thuộc nhóm train (mọi kích thước), đếm inlier ORB với từng ảnh -> inlier_max.
Ra: ro_orb.csv (ten, may, inlier_max, ten_train_gan_nhat, may_train). b2 bỏ ảnh có inlier_max >= 25.
Chạy: python b5_kiem_orb.py [TOP_K] [SO_TIEN_TRINH]
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch
from b3_dia_diem import dac_trung, inlier

DAY = Path(__file__).resolve().parent
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
from b2_chia_hang import TEST, NHOM


def mot(args):
    q, cands = args
    fq = dac_trung(q)
    return [inlier(fq, dac_trung(c)) for c in cands]


if __name__ == "__main__":
    ti = pd.read_csv(DAY / "thong_tin.csv")
    ti["nhom"] = ti.may.map(lambda x: NHOM.get(x, str(x)))
    uv = pd.read_csv(DAY / "ung_vien_test.csv")
    F = torch.tensor(np.load(DAY.parent / "semanticrt_dactrung_bien.npy")).cuda()
    F = F / F.norm(dim=1, keepdim=True)
    tr = np.where(~ti.nhom.isin(TEST))[0]
    S = F[torch.tensor(uv.hang.values).cuda()] @ F[torch.tensor(tr).cuda()].T
    j = S.topk(TOP_K, dim=1).indices.cpu().numpy()
    cands = [[ti.ten[tr[k]] for k in row] for row in j]
    # đặc trưng ORB tính MỘT lần cho mỗi ảnh (ứng viên train dùng chung cho nhiều ảnh test), rồi khớp trong tiến trình chính
    can = sorted(set(uv.ten) | {c for cs in cands for c in cs})
    print(f"{len(uv)} ảnh test x {TOP_K} ứng viên; {len(can)} ảnh cần tính đặc trưng", flush=True)
    with ProcessPoolExecutor(NP) as ex:
        dt = dict(zip(can, ex.map(dac_trung, can, chunksize=16)))
    kq = [[inlier(dt[q], dt[c]) for c in cs] for q, cs in zip(uv.ten, cands)]
    uv["inlier_max"] = [max(n) for n in kq]
    uv["ten_train_gan_nhat"] = [c[int(np.argmax(n))] for c, n in zip(cands, kq)]
    uv["may_train"] = [int(ti.may[ti.ten == t].iloc[0]) for t in uv.ten_train_gan_nhat]
    uv.to_csv(DAY / "ro_orb.csv", index=False)
    print(uv.inlier_max.describe())
    print("Theo máy test – tỉ lệ ảnh inlier_max >= 25:")
    print(uv.groupby("may").inlier_max.apply(lambda s: f"{(s >= 25).mean():.2f} ({(s >= 25).sum()}/{len(s)})"))
    print("Máy train hay trùng:", uv[uv.inlier_max >= 25].may_train.value_counts().head(8).to_dict())
