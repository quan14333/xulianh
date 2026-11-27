"""Bước 3: gom cụm góc máy thành ĐỊA ĐIỂM thật bằng khớp điểm ORB + RANSAC (chịu được xoay/zoom camera).
Lý do: đặc trưng cạnh 48x40 (bước 2) không nhận ra cùng địa điểm khi camera lệch góc/zoom
  (xem mắt 22/11: test máy 7/2 trùng địa điểm với train máy 15/6/18, cos chỉ 0,66–0,79).
Cách làm: mỗi cụm lấy K ảnh ngẫu nhiên (seed 0, chỉ ảnh 1280x1024); với mỗi cặp cụm (a, b) so K x K cặp ảnh:
  số inlier homography RANSAC; điểm cặp cụm = trung vị của (max inlier theo từng ảnh của a).
Ra: dia_diem_cap.csv (điểm từng cặp cụm), in ma trận.
Chạy: python b3_dia_diem.py [K]
"""
import sys, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, cv2

DAY = Path(__file__).resolve().parent
GOC = DAY.parent / "semanticrt" / "SemanticRT_dataset"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def dac_trung(ten):
    g = cv2.imread(str(GOC / "rgb" / f"{ten}.jpg"), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (640, 512))
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)   # ảnh đêm tối -> tăng tương phản cục bộ
    kp, des = cv2.ORB_create(2000).detectAndCompute(g, None)
    return np.float32([k.pt for k in kp]), des


def inlier(a, b):
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    tot = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(tot) < 8:
        return len(tot) // 2
    H, mask = cv2.findHomography(pa[[t.queryIdx for t in tot]], pb[[t.trainIdx for t in tot]], cv2.RANSAC, 5.0)
    return 0 if mask is None else int(mask.sum())


def so_cap(args):
    fa, fb = args
    return np.array([[inlier(x, y) for y in fb] for x in fa])


if __name__ == "__main__":
    ti = pd.read_csv(DAY / "thong_tin.csv")
    ti = ti[ti.rong == 1280]
    rng = np.random.default_rng(0)
    mau = {m: rng.choice(g.ten.values, min(K, len(g)), replace=False) for m, g in ti.groupby("may")}
    F = {m: [dac_trung(t) for t in v] for m, v in mau.items()}
    cum = sorted(F)
    # cùng cụm (ảnh khác nhau trong cụm) để có mức tham chiếu "chắc chắn cùng địa điểm"
    cap = [(a, b) for a, b in itertools.combinations_with_replacement(cum, 2)]
    with ProcessPoolExecutor(6) as ex:
        kq = list(ex.map(so_cap, [(F[a], F[b]) for a, b in cap]))
    rows = []
    for (a, b), M in zip(cap, kq):
        if a == b:
            M = M.astype(float); np.fill_diagonal(M, np.nan)
        diem = float(np.nanmedian(np.nanmax(M, 1)))
        rows.append(dict(a=a, b=b, diem=diem))
    d = pd.DataFrame(rows)
    d.to_csv(DAY / "dia_diem_cap.csv", index=False)
    mat = d.pivot(index="a", columns="b", values="diem")
    pd.set_option("display.width", 250)
    print(mat.round(0).fillna(-1).astype(int).to_string())
