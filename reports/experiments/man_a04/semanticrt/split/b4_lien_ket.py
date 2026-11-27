"""Bước 4: liên kết ảnh–ảnh giữa các cụm, xác nhận bằng ORB + RANSAC.
Mỗi ảnh 1280x1024: lấy TOP_K ảnh giống nhất (cos đặc trưng cạnh) thuộc CỤM KHÁC -> đếm inlier ORB.
Mức tham chiếu (22/11): hai cảnh khác nhau ~5 inlier; cùng địa điểm lệch góc/zoom 47–238; cùng cụm 89–2000.
Ra: lien_ket.csv (i, j, may_i, may_j, cos, inlier).
Chạy: python b4_lien_ket.py [TOP_K]
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch

sys.argv = sys.argv[:1] + sys.argv[1:]
from b3_dia_diem import dac_trung, inlier

DAY = Path(__file__).resolve().parent
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def mot(args):
    q, cands = args
    fq = dac_trung(q)
    return [inlier(fq, dac_trung(c)) for c in cands]


if __name__ == "__main__":
    ti = pd.read_csv(DAY / "thong_tin.csv")
    F = torch.tensor(np.load(DAY.parent / "semanticrt_dactrung_bien.npy")).cuda()
    F = F / F.norm(dim=1, keepdim=True)
    giu = np.where(ti.rong.values == 1280)[0]
    may = torch.tensor(ti.may.values).cuda()
    gi = torch.tensor(giu).cuda()
    viec, meta = [], []
    for s in range(0, len(giu), 512):
        h = gi[s:s + 512]
        S = F[h] @ F[gi].T
        S[may[h][:, None] == may[gi][None, :]] = -1          # chỉ xét cụm khác
        v, j = S.topk(TOP_K, dim=1)
        for r in range(len(h)):
            q = int(h[r]); cs = [int(giu[k]) for k in j[r].tolist()]
            viec.append((ti.ten[q], [ti.ten[c] for c in cs]))
            meta.append((q, cs, v[r].tolist()))
    with ProcessPoolExecutor(6) as ex:
        kq = list(ex.map(mot, viec, chunksize=16))
    rows = [dict(i=q, j=c, may_i=int(ti.may[q]), may_j=int(ti.may[c]), cos=round(cv, 4), inlier=n)
            for (q, cs, vs), ns in zip(meta, kq) for c, cv, n in zip(cs, vs, ns)]
    d = pd.DataFrame(rows)
    d.to_csv(DAY / "lien_ket.csv", index=False)
    print(d.inlier.describe())
    print("số liên kết inlier >= 25:", (d.inlier >= 25).sum())
