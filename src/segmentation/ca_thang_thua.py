"""Ca thành công / thất bại của SAGE so với VIS (B3, 273 ảnh test sạch). CPU, mô tả (không có tiêu chí).
(Trong ghi chép riêng của Mân gọi là thí nghiệm E21.)

Xếp ảnh theo d = mIoU_ảnh(sage) - mIoU_ảnh(vis), ma trận nhầm lẫn cộng các seed B3 có sẵn.
Lấy N ảnh thắng nhiều nhất và N ảnh thua nhiều nhất. Mỗi ảnh: lớp đóng góp nhiều nhất vào d,
lớp đó bị nhầm thành gì (với vis và sage). Hình: VIS | IR | SAGE | nhãn | dự đoán vis | sage | tb (seed 0).
Cần: SAGE_RUNS chứa *_b3_s*_1220_15lop/test_nham_last.npz và pred_last/ của seed 0 (dataset Kaggle kết quả),
      ảnh SAGE đã xuất (xuat_anh.py --bien_the sage) ở SAGE_ANH, FMB gốc.
Chạy: python ca_thang_thua.py [thư_mục_ra]  -> ca_thang_thua.txt, sage_thang.png, sage_thua.png
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chung import ANH, MAN, RUNS, duong, thu_muc  # noqa: E402
from thong_ke import doc_thong_tin_anh  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
N = 4
LOP = ["unlabeled", "road", "sidewalk", "building", "traffic_light", "traffic_sign", "vegetation", "sky",
       "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole"]
MAU = np.array([(0, 0, 0), (179, 228, 228), (181, 57, 133), (67, 162, 177), (200, 178, 50), (132, 45, 199),
                (66, 172, 84), (179, 73, 79), (76, 99, 166), (66, 121, 253), (137, 165, 91), (155, 97, 152),
                (105, 153, 140), (255, 255, 255), (135, 113, 90)], np.uint8)  # RGB, theo 4_bang_mau_color.csv


def iou_anh(conf):
    """IoU từng ảnh từng lớp 1..14 (N,14), NaN nếu lớp không có trong GT ảnh; bỏ pixel GT unlabeled."""
    c = conf[:, 1:, :].astype(float)
    tp = np.stack([c[:, k, k + 1] for k in range(14)], 1)
    g, p = c.sum(2), c[:, :, 1:].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(g > 0, tp / (g + p - tp), np.nan) * 100


def pred_path(b):
    f = RUNS / f"{b}_b3_s0_1220_15lop" / "pred_last"
    if f.exists():
        return f
    raise FileNotFoundError(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anh = doc_thong_tin_anh()
    sach = ~anh.leak.to_numpy()
    conf = {}
    for b in ("vis", "sage", "tb"):
        cs = [np.load(f)["conf"] for f in sorted(RUNS.glob(f"{b}_b3_s*_1220_15lop/test_nham_last.npz"))]
        conf[b] = sum(cs)
        ns = len(cs)
    I = {b: iou_anh(conf[b]) for b in conf}
    M = {b: np.nanmean(I[b], 1) for b in I}
    df = anh.copy()
    df["d"] = M["sage"] - M["vis"]
    df["d_tb"] = M["tb"] - M["sage"]
    q = np.quantile(anh.y_mean[sach], [1 / 3, 2 / 3])
    df["sang"] = np.where(df.y_mean < q[0], "tối", np.where(df.y_mean < q[1], "vừa", "sáng"))
    df = df[sach]
    man = pd.read_csv(MAN / "test.csv", dtype={"image_id": str}).set_index("image_id")
    pv, ps, pt = pred_path("vis"), pred_path("sage"), pred_path("tb")
    idx_of = {iid: i for i, iid in enumerate(anh.image_id)}

    out = []
    p = lambda t="": (out.append(t), print(t))
    p(f"Ca thắng/thua của SAGE so với VIS (B3, cộng {ns} seed, 273 ảnh sạch)")
    p(f"d = mIoU_ảnh(sage) - mIoU_ảnh(vis): TB {df.d.mean():+.2f}, trung vị {df.d.median():+.2f}; "
      f"sage thắng {(df.d > 1).sum()} ảnh, thua {(df.d < -1).sum()} ảnh, hòa (|d|≤1) {(df.d.abs() <= 1).sum()}")
    for g in ("tối", "vừa", "sáng"):
        x = df[df.sang == g]
        p(f"  {g:4s}: thắng {(x.d > 1).sum():3d}  thua {(x.d < -1).sum():3d}  hòa {(x.d.abs() <= 1).sum():3d}")
    p()

    for ten, chon in (("THẮNG", df.nlargest(N, "d")), ("THUA", df.nsmallest(N, "d"))):
        p(f"== {N} ảnh SAGE {ten} nhiều nhất")
        fig, ax = plt.subplots(len(chon), 7, figsize=(21, 2.4 * len(chon)))
        for row, r in enumerate(chon.itertuples()):
            i = idx_of[r.image_id]
            dl = I["sage"][i] - I["vis"][i]
            k = int(np.nanargmax(np.abs(dl)))
            lop = LOP[k + 1]
            cv_, cs_ = conf["vis"][i][k + 1], conf["sage"][i][k + 1]
            nham = lambda c: ", ".join(f"{LOP[j]} {100 * c[j] / c.sum():.0f}%" for j in np.argsort(c)[::-1][:2])
            p(f"  {r.image_id} [{r.sang}] d {r.d:+.1f} (tb-sage {r.d_tb:+.1f}) | lớp chênh nhất: {lop} "
              f"{I['vis'][i][k]:.0f}->{I['sage'][i][k]:.0f} | {lop} thật đoán thành: vis[{nham(cv_)}] "
              f"sage[{nham(cs_)}]")
            m = man.loc[r.image_id]
            r2 = type("R", (), {"vis_path": m.vis_path})
            vis = cv2.cvtColor(cv2.imread(str(duong(m.vis_path))), cv2.COLOR_BGR2RGB)
            ir = cv2.imread(str(duong(m.ir_path)), cv2.IMREAD_GRAYSCALE)
            sg = cv2.cvtColor(cv2.imread(str(ANH / "sage" / thu_muc(r2) / f"{r.image_id}.png")), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(duong(m.label_path)), cv2.IMREAD_UNCHANGED)
            pr = [cv2.imread(str(d / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED) for d in (pv, ps, pt)]
            iv, isg, it = (np.nanmean(I[b][i]) for b in ("vis", "sage", "tb"))
            o = [(vis, "VIS"), (ir, "IR"), (sg, "ảnh SAGE"), (MAU[gt], "nhãn"),
                 (MAU[pr[0]], f"đoán từ VIS {iv:.0f}"), (MAU[pr[1]], f"đoán từ SAGE {isg:.0f}"),
                 (MAU[pr[2]], f"đoán từ TB {it:.0f}")]
            for col, (im, t) in enumerate(o):
                a = ax[row, col]
                a.imshow(im, cmap="gray" if im.ndim == 2 else None)
                a.set_xticks([]), a.set_yticks([])
                a.set_title(t, fontsize=9)
            ax[row, 0].set_ylabel(f"{r.image_id} [{r.sang}]\nd {r.d:+.1f}", fontsize=9)
        # chú giải màu
        hp = [plt.Rectangle((0, 0), 1, 1, color=MAU[j] / 255) for j in range(1, 15) if j != 13]
        fig.legend(hp, [LOP[j] for j in range(1, 15) if j != 13], loc="lower center", ncol=13, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fn = OUT / f"sage_{'thang' if ten == 'THẮNG' else 'thua'}.png"
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        p(f"  -> {fn.name}\n")

    # tổng hợp: lớp nào hay quyết định thắng/thua
    p("== Lớp đóng góp nhiều nhất vào d, đếm trên các ảnh |d| > 3")
    dem = {"thắng": {}, "thua": {}}
    for r in df[df.d.abs() > 3].itertuples():
        i = idx_of[r.image_id]
        dl = I["sage"][i] - I["vis"][i]
        k = int(np.nanargmax(np.abs(dl)))
        key = "thắng" if r.d > 0 else "thua"
        dem[key][LOP[k + 1]] = dem[key].get(LOP[k + 1], 0) + 1
    for key, v in dem.items():
        p(f"  {key}: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items(), key=lambda t: -t[1])))
    (OUT / "ca_thang_thua.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
