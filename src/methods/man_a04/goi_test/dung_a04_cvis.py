"""a04_cvis = a04 + cổng VIS: Y = Ya04 + x(Y_vis − Ya04), x = 0,8·(1 − tối)·clip((m − 0,2)/0,3, 0, 1),
m = VIS trội chi tiết × không nóng (s = 8, a = 0,2 chọn trên Train 24/11, không dò lại). CrCb của VIS. Không nhãn, không SAM.
Cần ảnh tbsua_a04 dựng trước (dung_alpha.py). Kiểm: mau_a04_cvis.npz (Y của vài ảnh dựng ở máy nhà) phải trùng tuyệt đối."""
from pathlib import Path
import cv2, numpy as np, pandas as pd
from chung import ANH, MAN, duong, thu_muc
from xuat_anh import cong_vnc
NEN, TEN, S, A, X_MAX = "tbsua_a04", "a04_cvis", 8.0, 0.2, 0.8


def nl(a, s=4):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1)
    g = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)) / 8
    return cv2.GaussianBlur(g, (0, 0), s)


def cong_hep(y, ir, s, a):
    nong = cong_vnc(y, ir, g_min=0, toi_lo=-2, toi_hi=-1)
    toi = float(np.clip((110 - y.mean()) / 30, 0, 1))
    ev, ei = nl(y, s), nl(ir, s)
    m = ev / (ev + ei + 2) * np.clip(ev / 12, 0, 1) * (1 - nong)
    return X_MAX * (1 - toi) * np.clip((m - a) / .3, 0, 1)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(MAN / f"{t}.csv", dtype={"image_id": str}) for t in ("train", "dev")])
    assert len(df) == 1220 and df.image_id.is_unique
    mau = Path(__file__).with_name("mau_a04_cvis.npz"); M = dict(np.load(mau)) if mau.exists() else {}
    d, kiem = [], 0
    for r in df.itertuples():
        ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]; ir = cv2.imread(str(duong(r.ir_path)), 0)
        yt = cv2.cvtColor(cv2.imread(str(ANH / NEN / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        yf = (yt + cong_hep(y, ir, S, A) * (y.astype(np.float32) - yt)).round().clip(0, 255).astype(np.uint8)
        o = ANH / TEN / thu_muc(r) / f"{r.image_id}.png"; o.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(o), cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
        k = f"y_{r.image_id}"
        if k in M:
            yd = cv2.cvtColor(cv2.imread(str(o)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
            l = int(np.abs(yd.astype(int) - M[k].astype(int)).max()); print("kiểm", r.image_id, "lệch max", l); assert l <= 1; kiem += 1
        d.append(np.abs(yf - yt).mean())
    print(TEN, len(d), "|dY| so a04", round(float(np.mean(d)), 3), "| ảnh đối chứng", kiem, "/", len(M))
