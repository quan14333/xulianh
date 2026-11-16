"""Phần dùng chung cho xuat_anh.py, train_seg.py, eval_seg.py, tong_hop.py.

Đường dẫn đổi bằng biến môi trường (để chạy được cả trên Kaggle lẫn máy nhà):
  SAGE_FMB  : thư mục chứa FMB (tự dò nếu không đặt)
  SAGE_ANH  : nơi ghi ảnh đầu vào cho SegFormer, anh/{bien_the}/{train|test}/{id}.png
  SAGE_RUNS : nơi ghi kết quả train
  SAGE_MAN  : thư mục manifest (mặc định: manifests/ cạnh file này nếu có – kiểu gói code Kaggle –,
              không thì data/manifests/ của repo)
  SAGE_STUDENT: thư mục có model.py + checkpoints.pt của SAGE Student (model_sub/model.py và
              checkpoints/checkpoints.pt của RollingPlain/SAGE_IVIF, md5 checkpoint 241396a51a3c5e85e360a409d9e8fe4d)
"""
import functools
import os
from pathlib import Path

import numpy as np

DAY = Path(__file__).resolve().parent
ROOT = DAY.parents[1]  # gốc repo (khi chạy trong repo)


def _mac_dinh(ten_env, canh, trong_repo):
    if os.environ.get(ten_env):
        return Path(os.environ[ten_env])
    return DAY / canh if (DAY / canh).exists() else trong_repo


MAN = _mac_dinh("SAGE_MAN", "manifests", ROOT / "data/manifests")
ANH = Path(os.environ.get("SAGE_ANH", ROOT / "outputs/b3_anh"))
RUNS = Path(os.environ.get("SAGE_RUNS", ROOT / "runs/b3_fmb_seg"))
STUDENT = _mac_dinh("SAGE_STUDENT", "sage_student", ROOT / "checkpoints/pretrained/sage_student")

NC = 14  # lớp 1..14 của FMB -> chỉ số 0..13; nhãn 0 (unlabeled) -> 255 (bỏ qua)
IGNORE = 255
TEN_LOP = ["road", "sidewalk", "building", "traffic_light", "traffic_sign", "vegetation", "sky",
           "person", "car", "truck", "bus", "motorcycle", "bicycle", "pole"]
# 7 lớp bài SAGE in ra ở Bảng 1 (Road, Bldg., Sign, Veg., Bus, Per., Car), đúng thứ tự của bài
LOP_BAI = ["road", "building", "traffic_sign", "vegetation", "bus", "person", "car"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@functools.lru_cache(maxsize=1)
def _tim_fmb():
    """{'train': thư mục có Visible/Infrared/Label của train, 'test': ...}. Chịu được cả FMB/train/train/..."""
    if os.environ.get("SAGE_FMB"):
        goc = Path(os.environ["SAGE_FMB"])
    elif Path("/kaggle/input").is_dir():
        goc = Path("/kaggle/input")
    else:
        goc = ROOT / "data/raw/FMB"
    kq = {}
    for p in sorted(goc.rglob("Visible")):
        if p.is_dir() and p.parent.name in ("train", "test") and (p.parent / "Label").is_dir():
            kq.setdefault(p.parent.name, p.parent)
    assert set(kq) == {"train", "test"}, f"không thấy FMB (train/Visible, test/Visible) dưới {goc}"
    return kq


def duong(rel):
    """Đường dẫn thật của một ảnh FMB từ đường dẫn trong manifest, vd 'train/Visible/00001.png'."""
    s, con = rel.split("/", 1)
    return _tim_fmb()[s] / con


def thu_muc(r):
    """'train' hoặc 'test' – thư mục gốc FMB của ảnh (dev nằm trong train)."""
    return r.vis_path.split("/")[0]


TEN_LOP15 = ["unlabeled"] + TEN_LOP


def doc_nhan(path, lop0=False):
    """lop0=False: lớp 1..14 -> 0..13, nhãn 0 -> IGNORE.  lop0=True: giữ nguyên 0..14 (unlabeled là một lớp)."""
    import cv2  # import muộn để phần tính mIoU chạy được mà không cần OpenCV (test CPU)

    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert lab is not None and lab.ndim == 2 and lab.max() <= NC, path
    lab = lab.astype(np.int64)
    if lop0:
        return lab
    lab -= 1
    lab[lab < 0] = IGNORE
    return lab


def nham(gt, pred, n=NC):
    """Ma trận nhầm lẫn n x n (hàng = GT, cột = dự đoán), bỏ pixel IGNORE."""
    m = gt != IGNORE
    return np.bincount(gt[m] * n + pred[m], minlength=n * n).reshape(n, n)


def iou_lop(c):
    """IoU % từng lớp; NaN nếu lớp không có trong GT."""
    tp = np.diag(c).astype(np.float64)
    g, p = c.sum(1), c.sum(0)
    iou = np.full(len(c), np.nan)
    ok = g > 0
    iou[ok] = tp[ok] / (g[ok] + p[ok] - tp[ok])
    return iou * 100


def cac_miou(c):
    """mIoU theo nhiều quy ước (bài SAGE không nói dùng quy ước nào).

    Vào: ma trận 14x14 (model 14 lớp, đã bỏ pixel unlabeled) hoặc 15x15 (model 15 lớp, unlabeled là lớp 0).
    Trả (dict, IoU 14 lớp khi BỎ pixel unlabeled, IoU 15 lớp khi TÍNH unlabeled hoặc None).

    Bỏ pixel unlabeled (cách của nhóm):
      mIoU_co_mat : trung bình các lớp có trong GT (test FMB: 13 lớp, không có bicycle)
      mIoU_14     : chia cho đủ 14 lớp, lớp không có trong GT tính 0
      mIoU_7      : trung bình 7 lớp bài in ra ở Bảng 1
    Tính unlabeled là một lớp (cách của code SegMiF cho MFNet, cùng nhóm tác giả với SAGE; chỉ có khi 15x15):
      mIoU_15_nan0    : trung bình đủ 15 lớp, lớp không có trong GT tính 0 (cách in ra của SegMiF)
      mIoU_15_mrfs    : 15 lớp, bỏ lớp vừa không có trong GT vừa không bị dự đoán (cách của code MRFS cho FMB)
      mIoU_7_ca_nhan0 : 7 lớp bài in ra, IoU lấy từ ma trận 15x15
    """
    j7 = [TEN_LOP.index(t) for t in LOP_BAI]
    iou15 = None
    if len(c) == NC:
        iou = iou_lop(c)
        acc = np.diag(c).sum() / max(1, c.sum()) * 100
    else:
        r = c[1:]                                   # chỉ các pixel có nhãn thật 1..14
        tp = np.diag(r[:, 1:]).astype(np.float64)
        g, p = r.sum(1), r[:, 1:].sum(0)            # dự đoán "unlabeled" trên pixel có nhãn vẫn tính là sai
        iou = np.full(NC, np.nan)
        ok = g > 0
        iou[ok] = tp[ok] / (g[ok] + p[ok] - tp[ok]) * 100
        acc = tp.sum() / max(1, r.sum()) * 100
        iou15 = iou_lop(c)
    kq = {
        "mIoU_co_mat": float(np.nanmean(iou)),
        "mIoU_14": float(np.nan_to_num(iou).mean()),
        "mIoU_7": float(np.mean(iou[j7])),
        "so_lop_co_mat": int(np.isfinite(iou).sum()),
        "pixel_acc": float(acc),
    }
    if iou15 is not None:
        kq["mIoU_15_nan0"] = float(np.nan_to_num(iou15).mean())
        # Cách của code MRFS cho FMB (utils/metric.py): 15 lớp, nanmean, NaN chỉ khi lớp không có trong GT
        # VÀ không bị dự đoán lần nào; lớp vắng trong GT nhưng có bị dự đoán thì IoU = 0 và vẫn tính.
        hop = c.sum(1) + c.sum(0) - np.diag(c)
        kq["mIoU_15_mrfs"] = float(np.mean((np.diag(c) / np.maximum(hop, 1) * 100)[hop > 0]))
        kq["mIoU_7_ca_nhan0"] = float(np.mean(iou15[[j + 1 for j in j7]]))
        kq["IoU_unlabeled"] = float(iou15[0])
    return kq, iou, iou15
