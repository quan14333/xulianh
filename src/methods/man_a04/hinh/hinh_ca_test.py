"""27/11: hình ca Test cho MeanRes (mã a04 / tbsua_a04) – docx "Yêu cầu chốt", mục 04 + dòng 19.

Không train, không chấm lại: chỉ dùng
  - ma trận nhầm lẫn từng ảnh của 5 seed Test (repo sage-a04/test_fmb/runs/*/test_nham_last.npz) để CHỌN ca;
  - mask dự đoán seed 0 (pred_last/*.png) của đúng lượt đã chấm: MeanRes tải từ output Kaggle sage-test-a04-nhom1,
    TB/SAGE từ output cũ trên máy (npz trùng md5 với repo);
  - ảnh ghép dựng lại trên máy: SAGE, TB (luu_tru, như lượt Test), MeanRes = TB + 0,4·(tbsua_992 − TB) (công thức dung_test.py).

KIỂM: mask seed 0 + nhãn → ma trận nhầm lẫn phải TRÙNG TUYỆT ĐỐI ma trận trong npz (280 ảnh, 3 bản). Sai → dừng.

LUẬT CHỌN CA (ghi trước khi xem kết quả từng ảnh):
  điểm ảnh = mIoU_co_mat của MỘT ảnh (chung.cac_miou, bỏ pixel nhãn 0), trung bình 5 seed; chỉ 273 ảnh sạch.
  d_TB = MeanRes − TB, d_SAGE = MeanRes − SAGE.
  - cải thiện (2 ca): min(d_TB, d_SAGE) lớn nhất  (MeanRes hơn cả hai)
  - suy giảm  (2 ca): max(d_TB, d_SAGE) nhỏ nhất  (MeanRes thua cả hai)
  - khó       (2 ca): 00089 (khói, chọn từ audit 3–4/11) + ảnh VIS tối nhất (trung bình Y nhỏ nhất)
  Không lấy hai ca cách nhau ≤ 10 id (tránh cùng cảnh). Điểm một ảnh nhiễu (lớp nhỏ lật IoU) → hình chỉ để minh họa,
  kết luận vẫn theo mIoU gộp toàn tập.
python scene_v1/hinh_ca_test.py   (từ tai_lap)"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TL / "b3"))
from chung import cac_miou, doc_nhan, duong, nham  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = TL.parent / "sage-a04" / "test_fmb" / "runs"
ANH_CU = TL.parent / "luu_tru" / "tim_gap_b0_0811" / "seg" / "anh"
RA = TL / "hinh_ca_test"
BT = {"MeanRes": "tbsua_a04", "TB": "tb", "SAGE": "sage"}
PRED0 = {"MeanRes": TL / "_kaggle_out/test_a04_nhom1/runs/tbsua_a04_b3_s0_1220_15lop/pred_last",
         "TB": TL / "_kaggle_out/tbir/runs/tb_b3_s0_1220_15lop/pred_last",
         "SAGE": TL / "_kaggle_out/sv/runs/sage_b3_s0_1220_15lop/pred_last"}
# bảng màu FMB (fmb_audit_final/4_bang_mau_color.csv, RGB); bicycle không có trong test → xám
MAU = np.array([(0, 0, 0), (179, 228, 228), (181, 57, 133), (67, 162, 177), (200, 178, 50), (132, 45, 199),
                (66, 172, 84), (179, 73, 79), (76, 99, 166), (66, 121, 253), (137, 165, 91), (155, 97, 152),
                (105, 153, 140), (128, 128, 128), (135, 113, 90)], np.uint8)


def man():
    df = pd.read_csv(TL / "b3/manifests/test.csv", dtype={"image_id": str})
    df["leak"] = df.leak.astype(str).str.lower().eq("true")
    return df.set_index("image_id")


def anh_meanres(df):
    """Dựng MeanRes cho test (y hệt dung_test.py), từ tbsua_992 dựng bằng xuat_anh.py vào hinh_ca_test/anh."""
    o = RA / "anh/tbsua_a04/test"; o.mkdir(parents=True, exist_ok=True)
    for i, r in df.iterrows():
        f = o / f"{i}.png"
        if f.exists():
            continue
        ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb); y = ycc[..., 0]
        ir = cv2.imread(str(duong(r.ir_path)), 0)
        u = cv2.cvtColor(cv2.imread(str(RA / "anh/tbsua_992/test" / f"{i}.png")), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        t = (y.astype(np.float32) + ir.astype(np.float32)) / 2
        ya = (t + .4 * (u - t)).round().clip(0, 255).astype(np.uint8)
        assert cv2.imwrite(str(f), cv2.cvtColor(np.dstack([ya, ycc[..., 1:]]), cv2.COLOR_YCrCb2BGR))


def duong_ghep(ten, i):
    return (RA / "anh/tbsua_a04/test" if ten == "MeanRes" else ANH_CU / BT[ten] / "test") / f"{i}.png"


def diem_anh(df):
    """mIoU từng ảnh, trung bình 5 seed; kèm seed 0."""
    bang = {}
    for ten, bt in BT.items():
        v = []
        for s in range(5):
            z = np.load(REPO / f"{bt}_b3_s{s}_1220_15lop" / "test_nham_last.npz")
            ids = z["image_id"].astype(str)
            v.append(pd.Series([cac_miou(c)[0]["mIoU_co_mat"] for c in z["conf"]], index=ids))
        bang[ten] = pd.concat(v, axis=1).mean(1)
        bang[ten + "_s0"] = v[0]
    d = pd.DataFrame(bang).loc[df.index]
    d["leak"] = df.leak
    d["d_TB"] = d.MeanRes - d.TB
    d["d_SAGE"] = d.MeanRes - d.SAGE
    d["sang_vis"] = [cv2.cvtColor(cv2.imread(str(duong(df.loc[i, "vis_path"]))), cv2.COLOR_BGR2YCrCb)[..., 0].mean()
                     for i in d.index]
    return d


def kiem_mask(df):
    for ten, bt in BT.items():
        z = np.load(REPO / f"{bt}_b3_s0_1220_15lop" / "test_nham_last.npz")
        sai = 0
        for i, c in zip(z["image_id"].astype(str), z["conf"]):
            p = cv2.imread(str(PRED0[ten] / f"{i}.png"), cv2.IMREAD_UNCHANGED)
            assert p is not None, (ten, i)
            sai += not np.array_equal(nham(doc_nhan(duong(df.loc[i, "label_path"]), lop0=True), p.astype(np.int64), 15), c)
        print(f"kiểm mask {ten} s0: {len(z['conf'])} ảnh, lệch npz {sai}")
        assert sai == 0, f"mask {ten} không khớp npz"


def chon(d):
    c = d[~d.leak].copy()
    da = []

    def lay(thu_tu, n, nhom):
        for i in thu_tu:
            if len([x for x in da if x[1] == nhom]) == n:
                break
            if all(abs(int(i) - int(x[0])) > 10 for x in da):
                da.append((i, nhom))
    lay(c[["d_TB", "d_SAGE"]].min(1).sort_values(ascending=False).index, 2, "cai_thien")
    lay(c[["d_TB", "d_SAGE"]].max(1).sort_values().index, 2, "suy_giam")
    lay(["00089"], 1, "kho_khoi")
    lay(c.sang_vis.sort_values().index, 1, "kho_toi")
    return da


def ve(df, d, i, nhom, k):
    r = df.loc[i]
    lab = doc_nhan(duong(r.label_path), lop0=True)
    vis = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2RGB)
    ir = cv2.imread(str(duong(r.ir_path)), 0)
    fig, ax = plt.subplots(3, 3, figsize=(15, 11.6))
    o = [(vis, "VIS"), (ir, "IR"), (MAU[lab], "Nhãn GT (đen = không chấm)")]
    for ten in BT:
        o.append((cv2.cvtColor(cv2.imread(str(duong_ghep(ten, i))), cv2.COLOR_BGR2RGB), f"Ảnh ghép {ten}"))
    for ten in BT:
        p = cv2.imread(str(PRED0[ten] / f"{i}.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)
        m = MAU[p].copy()
        m[(lab > 0) & (p != lab)] = 255          # sai → trắng
        m[lab == 0] = (m[lab == 0] * .35).astype(np.uint8)
        o.append((m, f"Dự đoán {ten} s0 (trắng = sai)\nmIoU ảnh s0 {d.loc[i, ten + '_s0']:.1f} · TB 5 seed {d.loc[i, ten]:.1f}"))
    for a, (im, t) in zip(ax.flat, o):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=11); a.axis("off")
    ten_nhom = {"cai_thien": "Cải thiện", "suy_giam": "Suy giảm", "kho_khoi": "Khó – khói", "kho_toi": "Khó – tối"}[nhom]
    fig.suptitle(f"{ten_nhom}: test {i}  |  MeanRes − TB {d.loc[i, 'd_TB']:+.1f}, MeanRes − SAGE {d.loc[i, 'd_SAGE']:+.1f}"
                 " (mIoU một ảnh, TB 5 seed)", fontsize=13)
    fig.tight_layout()
    f = RA / "figures" / f"ca{k}_{nhom}_{i}.png"; f.parent.mkdir(exist_ok=True)
    fig.savefig(f, dpi=90); plt.close(fig)
    return f


def main():
    df = man()
    kiem_mask(df)
    anh_meanres(df)
    d = diem_anh(df)
    d.round(3).to_csv(RA / "diem_tung_anh.csv")
    da = chon(d)
    print(d.loc[[i for i, _ in da], ["MeanRes", "TB", "SAGE", "d_TB", "d_SAGE", "sang_vis"]].round(1).assign(nhom=[n for _, n in da]))
    for k, (i, n) in enumerate(da, 1):
        print("ghi", ve(df, d, i, n, k))


if __name__ == "__main__":
    main()
