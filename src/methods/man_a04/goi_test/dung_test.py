"""Vòng Test: dựng ảnh phương án nhóm cho đủ 1.500 ảnh (train 1.045 + dev 175 + test 280, manifest chính thức).
  tbsua_a04 : Y = TB + 0,4·(tbsua_992 − TB)          (y hệt dung_alpha.py; tbsua_992 ĐÓNG BĂNG, học không nhãn trên 992 ảnh train)
  a04_cvis  : Y = Ya04 + x·(Y_vis − Ya04)             (y hệt dung_a04_cvis.py, s = 8, a = 0,2)
Không nhãn, không SAM. Cần ảnh tbsua_992 dựng trước (xuat_anh.py --bien_the tbsua_992).
python dung_test.py tbsua_a04 [a04_cvis]"""
import sys
import cv2, numpy as np, pandas as pd
from chung import ANH, MAN, duong, thu_muc
from dung_a04_cvis import cong_hep, S, A


def main(bts):
    df = pd.concat([pd.read_csv(MAN / f"{t}.csv", dtype={"image_id": str}) for t in ("train", "dev", "test")])
    assert len(df) == 1500 and df.image_id.is_unique
    d = {b: [] for b in bts}
    for r in df.itertuples():
        ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb); y = ycc[..., 0]
        ir = cv2.imread(str(duong(r.ir_path)), 0)
        u = cv2.cvtColor(cv2.imread(str(ANH / "tbsua_992" / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        t = (y.astype(np.float32) + ir.astype(np.float32)) / 2
        ya = (t + .4 * (u - t)).round().clip(0, 255).astype(np.uint8)
        ra = {"tbsua_a04": ya}
        if "a04_cvis" in bts:   # như bản Dev: nền là Y đọc lại từ PNG a04 (qua YCrCb -> BGR -> YCrCb)
            yt = cv2.cvtColor(cv2.cvtColor(np.dstack([ya, ycc[..., 1:]]), cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
            ra["a04_cvis"] = (yt + cong_hep(y, ir, S, A) * (y.astype(np.float32) - yt)).round().clip(0, 255).astype(np.uint8)
        for b in bts:
            o = ANH / b / thu_muc(r) / f"{r.image_id}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(o), cv2.cvtColor(np.dstack([ra[b], ycc[..., 1:]]), cv2.COLOR_YCrCb2BGR))
            d[b].append((ra[b] >= 250).mean() * 100)
    for b in bts:
        print(b, len(d[b]), "ảnh, cháy trắng %", round(float(np.mean(d[b])), 4))


if __name__ == "__main__":
    bts = sys.argv[1:] or ["tbsua_a04"]
    assert set(bts) <= {"tbsua_a04", "a04_cvis"}, bts
    main(bts)
