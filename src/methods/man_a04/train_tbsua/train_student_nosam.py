"""E17 – Train lại Student của SAGE KHÔNG có SAM (và không có Teacher), chỉ dùng loss nội dung.

Câu hỏi: phần SAM/Teacher của SAGE có làm ảnh ghép tốt hơn cho phân đoạn không?
Bài (Bảng 3): bỏ SAM thì mIoU FMB 61,2 -> 56,5 (-4,7). Nhóm kiểm lại bằng cùng thiết lập B3.

Giữ nguyên theo external/SAGE_IVIF/train.py + model_sub/losses.py:
  - kiến trúc Student (sage_student/model.py, 0,136M tham số)
  - loss nội dung: loss_fuse = 3*MSE(o,Y) + 2*MSE(o,IR); loss_grad = 9*Grad(o,Y,IR); loss = 3*loss_fuse + 3*loss_grad
  - Adam lr 2e-3, CosineAnnealingLR(T_max = 100 epoch * số batch, eta_min 1e-5), batch 4, crop 192x256 + lật ngang
  - 100 epoch; code gốc chỉ cập nhật Student ở batch chẵn (xen kẽ với Teacher) -> số bước cập nhật = nửa số batch,
    và scheduler cũng chỉ bước theo Student -> lr chỉ giảm nửa chu kỳ cosine. Nhóm làm GIỐNG HỆT điều này.
BỎ: các số hạng có ref (ảnh Teacher), loss tương phản theo mask SAM, loss cosine đặc trưng trung gian.
Dữ liệu: 1.220 ảnh train chính thức (train + dev của nhóm), KHÔNG đụng test.

Ra: ../b3/sage_student/nosam.pt (+ nosam_log.csv). Sau đó: python ../b3/xuat_anh.py --bien_the nosam
Ví dụ: python train_student_nosam.py           (chạy thật)
       python train_student_nosam.py --thu 50  (chạy thử 50 bước: đo tốc độ)
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_DAY = Path(__file__).resolve().parent
B3 = _DAY if (_DAY / "chung.py").exists() else _DAY.parent / "b3"  # gói Kaggle: cùng thư mục
sys.path.insert(0, str(B3))
from chung import ANH, MAN, STUDENT, duong, thu_muc  # noqa: E402

sys.path.insert(0, str(STUDENT))
from model import Network  # noqa: E402
from model_xa import NetworkXa  # noqa: E402  (S-xa 21/11)
from model_mat import NetworkMat  # noqa: E402  (S-mắt 23/11)

SOBEL_X = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
SOBEL_Y = torch.tensor([[1., 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None]


def grad(x, kx, ky):
    return F.conv2d(x, kx, padding=1).abs() + F.conv2d(x, ky, padding=1).abs()


def grad_loss(o, s1, s2, kx, ky):
    """Như GradientLoss của SAGE: L1(|∇o|, max(|∇s1|, |∇s2|))."""
    return F.l1_loss(grad(o, kx, ky), torch.maximum(grad(s1, kx, ky), grad(s2, kx, ky)))


VAT = (8, 9, 10, 11, 12, 13)   # nhãn FMB: person, car, truck, bus, motorcycle, bicycle
KER_VANH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))   # vành ~7 px quanh vật
NGUONG_KHAC = 8.0   # (23/11) |vhn − nosam| ≥ 8 mức xám -> trọng số tối đa 1 + w_khac


def vat_the(lab, toi_da=16, dt_min=15, lop=VAT):
    """A2: tách từng vật (thành phần liên thông của mỗi lớp vật) trong crop nhãn -> list (mask vật, mask vành).
    Vành = vùng nở ra quanh vật, bỏ chính lớp đó và bỏ nhãn 0 (unlabeled). Chỉ dùng lúc TRAIN."""
    ra = []
    for c in lop:
        m = (lab == c).astype(np.uint8)
        if m.sum() < dt_min:
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < dt_min:
                continue
            v = (cc == k).astype(np.uint8)
            vanh = (cv2.dilate(v, KER_VANH) > 0) & (m == 0) & (lab != 0)
            if vanh.sum() >= dt_min:
                ra.append((v.astype(bool), vanh))
    if len(ra) > toi_da:   # quá nhiều vật: giữ các vật NHỎ nhất (người nhỏ là chỗ nosam thua nặng nhất)
        ra = sorted(ra, key=lambda t: t[0].sum())[:toi_da]
    return ra


def loss_vung(o, y, ir, ds):
    """A2: phạt khi tương phản vật–vành trên ảnh ghép THẤP hơn tương phản mạnh nhất của 2 nguồn.
    c(x) = |TB_vật(x) - TB_vành(x)|; loss = TB_vật relu(max(c(Y), c(IR)) - c(o))."""
    if not ds:
        return o.new_zeros(())
    b, mv, mr = zip(*ds)
    b = torch.tensor(b, device=o.device)
    mv = torch.from_numpy(np.stack(mv)).to(o.device, torch.float32)
    mr = torch.from_numpy(np.stack(mr)).to(o.device, torch.float32)

    def c(x):
        x = x[b, 0]
        return ((x * mv).sum((1, 2)) / mv.sum((1, 2)) - (x * mr).sum((1, 2)) / mr.sum((1, 2))).abs()
    with torch.no_grad():
        dich = torch.maximum(c(y), c(ir))
    return F.relu(dich - c(o)).mean()


def loss_den(o, y, ds):
    """S-đèn (21/11): GIỮ ĐÈN THEO VIS. Phạt khi tương phản đèn–vành trên ảnh ghép CAO hơn của VIS.
    Lý do (ma trận nhầm lẫn val151, 3 seed): A2 tô đậm đèn (8,0 -> 14,2 mức xám; VIS 7,6, IR 22,8) dù đèn không
    có trong loss -> B0 báo nhầm đèn->biển 2,0k -> 5,1k px. loss = TB_đèn relu(c(o) − c(Y))."""
    if not ds:
        return o.new_zeros(())
    b, mv, mr = zip(*ds)
    b = torch.tensor(b, device=o.device)
    mv = torch.from_numpy(np.stack(mv)).to(o.device, torch.float32)
    mr = torch.from_numpy(np.stack(mr)).to(o.device, torch.float32)

    def c(x):
        x = x[b, 0]
        return ((x * mv).sum((1, 2)) / mv.sum((1, 2)) - (x * mr).sum((1, 2)) / mr.sum((1, 2))).abs()
    with torch.no_grad():
        dich = c(y)
    return F.relu(c(o) - dich).mean()


KER_DAI = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))   # A4: dải ~10 px mỗi bên biên 2 vùng


def cap_vung(lab, toi_da=8, px_min=50):
    """A4 (21/11): mọi cặp vùng KỀ NHAU của 2 lớp bất kỳ trong crop nhãn (bỏ nhãn 0), không gắn lớp cụ thể.
    Mỗi cặp (a, b): chỉ lấy phần của a và của b nằm trong dải quanh biên chung -> (mask phía a, mask phía b).
    Quá nhiều cặp: giữ các cặp có dải lớn nhất. Chỉ dùng lúc TRAIN."""
    lop = [int(c) for c in np.unique(lab) if c != 0]
    no = {c: cv2.dilate((lab == c).astype(np.uint8), KER_DAI) > 0 for c in lop}
    ra = []
    for i, a in enumerate(lop):
        for b in lop[i + 1:]:
            dai = no[a] & no[b]
            ma, mb = dai & (lab == a), dai & (lab == b)
            if ma.sum() >= px_min and mb.sum() >= px_min:
                ra.append((ma, mb))
    if len(ra) > toi_da:
        ra = sorted(ra, key=lambda t: -(t[0].sum() + t[1].sum()))[:toi_da]
    return ra


def loss_tach(o, y, ir, ds):
    """A4: phạt khi độ TÁCH 2 vùng kề nhau trên ảnh ghép thấp hơn nguồn tách tốt nhất.
    s(x) = |TB_a − TB_b| / sqrt((var_a + var_b)/2 + eps); loss = TB_cặp relu(max(s(Y), s(IR)) − s(o)).
    Bằng chứng (kiem_a3/do_tuong_phan_lop.py, ảnh train): tách đường–vỉa hè VIS 0,45 / IR 1,06 / SAGE 0,83 / nosam 0,49."""
    if not ds:
        return o.new_zeros(())
    b, ma, mb = zip(*ds)
    b = torch.tensor(b, device=o.device)
    ma = torch.from_numpy(np.stack(ma)).to(o.device, torch.float32)
    mb = torch.from_numpy(np.stack(mb)).to(o.device, torch.float32)

    def s(x):
        x = x[b, 0]
        na, nb = ma.sum((1, 2)), mb.sum((1, 2))
        ta, tb = (x * ma).sum((1, 2)) / na, (x * mb).sum((1, 2)) / nb
        va = ((x - ta[:, None, None]) ** 2 * ma).sum((1, 2)) / na
        vb = ((x - tb[:, None, None]) ** 2 * mb).sum((1, 2)) / nb
        return (ta - tb).abs() / torch.sqrt((va + vb) / 2 + 1e-4)
    with torch.no_grad():
        dich = torch.maximum(s(y), s(ir))
    return F.relu(dich - s(o)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thu", type=int, default=0, help="chỉ chạy N bước để đo tốc độ, không lưu")
    ap.add_argument("--ten", default="nosam")
    ap.add_argument("--chi_train", action="store_true",
                    help="chỉ 1.045 ảnh train (bỏ Dev) – dùng làm điểm khởi tạo cho pp chọn trên Dev (M11), tránh rò Dev")
    ap.add_argument("--w_ir", type=float, default=None,
                    help="tỉ lệ IR trong MSE cường độ: l_fuse = 5*((1-w)*MSE(o,Y) + w*MSE(o,IR)); bỏ trống = gốc 3*Y + 2*IR (w 0,4)")
    ap.add_argument("--them", nargs="*", default=[],
                    help="thư mục dữ liệu NGOÀI (không nhãn) có vi/ và ir/ cùng tên, vd MSRS train; chỉ dùng cho loss ghép")
    ap.add_argument("--lam_vung", type=float, default=0.0,
                    help="A2: hệ số loss tương phản vật–vành (mask lấy từ NHÃN train, chỉ lúc train); 0 = tắt (y hệt cũ)")
    ap.add_argument("--lop_vat", nargs="+", type=int, default=list(VAT),
                    help="A2: nhãn FMB coi là vật (mặc định 8–13 người+xe); mở rộng: thêm 4 đèn, 5 biển báo, 14 cột")
    ap.add_argument("--kien_truc", default="goc", choices=["goc", "xa", "mat"],
                    help="goc = Student SAGE 0,136M; xa = S-xa (thêm nhánh ngữ cảnh x16, model_xa.py)")
    ap.add_argument("--dim", type=int, default=32, help="số kênh Student (gốc 32; S-rộng 64)")
    ap.add_argument("--xa_them", type=int, default=2, help="S-xa: số tầng thu nhỏ thêm (2 -> x16, 3 -> x32)")
    ap.add_argument("--lam_tach", type=float, default=0.0,
                    help="A4: hệ số loss giữ độ tách 2 vùng kề nhau (mọi cặp lớp, mask từ NHÃN train); 0 = tắt")
    ap.add_argument("--lam_den", type=float, default=0.0,
                    help="S-đèn: hệ số loss giữ tương phản đèn không vượt VIS (mask từ NHÃN train); 0 = tắt")
    ap.add_argument("--lop_den", nargs="+", type=int, default=[4], help="S-đèn: nhãn giữ theo VIS (mặc định 4 đèn)")
    ap.add_argument("--w_lop", default=None,
                    help="L_int theo lớp: file json {nhãn: w_IR} (scene_v1/w_lop.json); MSE cường độ = 5·TB[(1−w)(o−Y)² + w(o−IR)²],"
                         " w theo nhãn từng điểm ảnh (chỉ lúc train). w 0,4 mọi nơi = y hệt gốc")
    ap.add_argument("--ref", default=None,
                    help="CHƯNG CẤT kiểu SAGE (23/11): ảnh Teacher = ảnh ghép đã xuất ANH/<ref>/... (vd vhn). Như loss gốc SAGE "
                         "(model_sub/losses.py): loss_fuse += w_ref·MSE(o,ref), loss_grad += w_ref_grad·Grad(o,ref,ref); "
                         "tổng vẫn 3·fuse + 3·grad. Không bật -> y hệt bản cũ")
    ap.add_argument("--w_ref", type=float, default=4.0, help="hệ số MSE(o,ref) trong loss_fuse (gốc SAGE 4)")
    ap.add_argument("--w_ref_grad", type=float, default=3.0, help="hệ số Grad(o,ref) trong loss_grad (gốc SAGE 3)")
    ap.add_argument("--w_khac", type=float, default=0.0,
                    help="(23/11) tăng trọng số MSE(o,ref) ở chỗ ref khác ảnh nền --ref_nen: w = 1 + w_khac·min(|ref−nen|/NGUONG_KHAC, 1), "
                         "chuẩn hóa chia TB(w) để giữ thang loss. Chỉ dùng lúc train; 0 = tắt (y hệt bản cũ).")
    ap.add_argument("--ref_nen", default="nosam_kp841", help="ảnh nền để tìm vùng thầy đã chỉnh (vhn = nosam_kp841 + chỉnh vùng)")
    ap.add_argument("--chi_ref", action="store_true",
                    help="CHƯNG CẤT THUẦN (23/11): bỏ loss gốc (Y, IR, grad max), chỉ học theo ảnh ref: "
                         "l_fuse = 5·MSE(o,ref), l_grad = 9·Grad(o,ref,ref) (cùng thang hệ số loss gốc 3+2 và 9). Cần --ref.")
    ap.add_argument("--khoi_tao", default=None,
                    help="(23/11) KHỞI TẠO từ checkpoint Student b3/sage_student/<ten>.pt (vd nosam_kp841 = nền của thầy vhn) "
                         "thay vì ngẫu nhiên: chỉ học PHẦN DƯ thầy chỉnh, không mang nhiễu train lại (đo: train lại nosam lệch "
                         "person RMS 33 mức xám > thầy chỉnh 15). Với --kien_truc mat: nhánh mat_ khởi tạo mới (conv cuối = 0).")
    ap.add_argument("--w_grad_chi_ref", type=float, default=9.0,
                    help="(23/11) hệ số Grad(o,ref) khi --chi_ref (mặc định 9 = như cũ). Đo: với 9, grad chiếm ~95-99%% loss, "
                         "mà thầy vhn chỉnh MƯỢT (xác suất mềm) nên grad gần như không thấy -> hạ xuống để MSE(o,ref) có tiếng nói.")
    ap.add_argument("--dong_bang_goc", action="store_true",
                    help="(23/11) đóng băng toàn bộ Student gốc (cả thống kê BN), chỉ học nhánh mat_. Cần --khoi_tao và --kien_truc mat.")
    ap.add_argument("--lr_du", action="store_true",
                    help="lr cosine đi ĐỦ chu kỳ theo số bước Student thật (gốc SAGE chỉ đi nửa); khác SAGE gốc – ghi rõ")
    a = ap.parse_args()
    assert not a.chi_ref or a.ref, "--chi_ref cần --ref"
    assert not a.w_khac or a.ref, "--w_khac cần --ref"
    assert not a.dong_bang_goc or (a.khoi_tao and a.kien_truc == "mat"), "--dong_bang_goc cần --khoi_tao và --kien_truc mat"
    W_LOP = None
    if a.w_lop:
        import json
        W_LOP = np.full(256, 0.4, np.float32)
        for k, v in json.loads(Path(a.w_lop).read_text(encoding="utf-8")).items():
            W_LOP[int(k)] = v
        print("L_int theo lớp, w_IR:", {k: float(W_LOP[k]) for k in range(15)}, flush=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", dev, flush=True)

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str})
                    for s in (("train",) if a.chi_train else ("train", "dev"))])
    assert len(df) == (int(os.environ.get("SO_TRAIN", 1045)) if a.chi_train else 1220) and not df.vis_path.str.startswith("test").any()
    print(f"Nạp {len(df)} ảnh train vào RAM ...", flush=True)
    # danh sách (không stack) để nhận ảnh khác kích thước; không có --them thì y hệt bản cũ (cùng dãy ngẫu nhiên)
    Y = [cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2YCrCb)[:, :, 0] for p in df.vis_path]
    IR = [cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2GRAY) for p in df.ir_path]
    # A2: nhãn chỉ nạp khi bật loss vùng; ảnh ngoài (--them) không có nhãn -> None
    LAB = ([cv2.imread(str(duong(p)), cv2.IMREAD_GRAYSCALE) for p in df.label_path]
           if a.lam_vung > 0 or a.lam_tach > 0 or a.lam_den > 0 or a.w_lop else [None] * len(df))
    REF, WK = None, []
    if a.ref:
        assert not a.them, "--ref chưa hỗ trợ ảnh ngoài (--them)"
        REF = []
        for r in df.itertuples():
            f = ANH / a.ref / thu_muc(r) / f"{r.image_id}.png"
            im = cv2.imread(str(f))
            assert im is not None, f"thiếu ảnh Teacher {f}"
            REF.append(cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0])
            if a.w_khac:             # bản đồ trọng số (float16 cho nhẹ RAM)
                nen = cv2.imread(str(ANH / a.ref_nen / thu_muc(r) / f"{r.image_id}.png"))
                assert nen is not None, f"thiếu ảnh nền {a.ref_nen} {r.image_id}"
                d = np.abs(REF[-1].astype(np.float32) - cv2.cvtColor(nen, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32))
                WK.append((1 + a.w_khac * np.minimum(d / NGUONG_KHAC, 1)).astype(np.float16))
        print(f"Chưng cất: ảnh Teacher {a.ref} ({len(REF)} ảnh), w_ref {a.w_ref}, w_ref_grad {a.w_ref_grad}", flush=True)
    for d in a.them:
        d = Path(d)
        assert "test" not in d.as_posix().lower(), f"không train trên tập test: {d}"
        ten = sorted(p.name for p in (d / "vi").iterdir())
        assert ten and all((d / "ir" / t).exists() for t in ten), f"{d}: vi/ và ir/ không khớp"
        for t in ten:
            v, r = cv2.imread(str(d / "vi" / t)), cv2.imread(str(d / "ir" / t), cv2.IMREAD_GRAYSCALE)
            assert v.shape[:2] == r.shape, f"{d}/{t}: VIS và IR khác kích thước"
            Y.append(cv2.cvtColor(v, cv2.COLOR_BGR2YCrCb)[:, :, 0]); IR.append(r); LAB.append(None)
        print(f"  + {len(ten)} cặp ngoài từ {d}", flush=True)
    N = len(Y)
    print(f"Tổng {N} cặp train cho mạng ghép", flush=True)
    ch, cw = 192, 256

    net = (NetworkXa(a.dim, a.xa_them) if a.kien_truc == "xa" else
           NetworkMat(a.dim, pretrained=True) if a.kien_truc == "mat" else Network(a.dim)).to(dev).train()
    if a.khoi_tao:
        sd = torch.load(STUDENT / f"{a.khoi_tao}.pt", map_location="cpu", weights_only=True)
        assert not any(k.startswith(("_tb_", "dau.", "xa_")) for k in sd), "checkpoint khởi tạo phải là Student gốc"
        thieu, thua = net.load_state_dict(sd, strict=False)
        assert not thua and all(k.startswith("mat_") for k in thieu), f"khởi tạo lệch khóa: thiếu {thieu[:5]}, thừa {thua[:5]}"
        print(f"Khởi tạo từ {a.khoi_tao}.pt ({len(sd)} khóa; {len(thieu)} khóa mat_ mới)", flush=True)
    if a.dong_bang_goc:
        for k, p in net.named_parameters():
            if not k.startswith("mat_"):
                p.requires_grad = False
        bn_goc = [m for k, m in net.named_modules() if isinstance(m, torch.nn.BatchNorm2d) and not k.startswith("mat_")]
        for m in bn_goc:
            m.eval()                 # giữ thống kê BN của nền (net.train() chỉ gọi 1 lần lúc tạo)
        print(f"Đóng băng Student gốc ({len(bn_goc)} BN ở eval)", flush=True)
    print(f"Student: {sum(p.numel() for p in net.parameters()) / 1e6:.3f}M tham số, "
          f"học {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.3f}M", flush=True)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    so_batch = math.ceil(N / a.batch)
    # Gốc SAGE: T_max = mọi batch, nhưng Student chỉ bước ở batch chẵn -> lr chỉ đi NỬA chu kỳ (cuối = ½ lr đầu).
    # --lr_du (22/11): T_max = số bước Student thật -> lr giảm đủ về lr_min.
    t_max = a.epochs * math.ceil(so_batch / 2) if a.lr_du else a.epochs * so_batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t_max, eta_min=a.lr_min)
    kx, ky = SOBEL_X.to(dev), SOBEL_Y.to(dev)

    log, buoc, t0 = [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        thu_tu = rng.permutation(N)
        tong = dict(loss=0., fuse=0., grad=0., vung=0., tach=0., den=0., n=0)
        for bi in range(so_batch):
            if bi % 2 == 1:      # code gốc: batch lẻ dành cho Teacher, Student chỉ học ở batch chẵn
                continue
            idx = thu_tu[bi * a.batch:(bi + 1) * a.batch]
            ys, irs, ds, dt, dd, ws, rs, wks = [], [], [], [], [], [], [], []
            for bb, k in enumerate(idx):
                H, W = Y[k].shape
                i0, j0 = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
                y, ir = Y[k][i0:i0 + ch, j0:j0 + cw], IR[k][i0:i0 + ch, j0:j0 + cw]
                lab = None if LAB[k] is None else LAB[k][i0:i0 + ch, j0:j0 + cw]
                rf = None if REF is None else REF[k][i0:i0 + ch, j0:j0 + cw]
                wk = WK[k][i0:i0 + ch, j0:j0 + cw] if WK else None
                if rng.random() < 0.5:   # cùng một lần rút ngẫu nhiên như bản cũ -> lam_vung 0 cho dãy y hệt
                    y, ir = y[:, ::-1], ir[:, ::-1]
                    lab = None if lab is None else lab[:, ::-1]
                    rf = None if rf is None else rf[:, ::-1]
                    wk = None if wk is None else wk[:, ::-1]
                ys.append(y.copy()), irs.append(ir.copy())
                if rf is not None:
                    rs.append(rf.copy())
                if wk is not None:
                    wks.append(wk.astype(np.float32))
                if W_LOP is not None:   # ảnh ngoài không nhãn -> 0,4 (gốc)
                    ws.append(np.full((ch, cw), 0.4, np.float32) if lab is None else W_LOP[lab])
                if lab is not None:
                    lab = np.ascontiguousarray(lab)
                    if a.lam_vung > 0:
                        ds += [(bb, v, r) for v, r in vat_the(lab, lop=a.lop_vat)]
                    if a.lam_tach > 0:
                        dt += [(bb, p, q) for p, q in cap_vung(lab)]
                    if a.lam_den > 0:
                        dd += [(bb, v, r) for v, r in vat_the(lab, lop=a.lop_den)]
            y = torch.from_numpy(np.stack(ys)).float().div(255)[:, None].to(dev)
            ir = torch.from_numpy(np.stack(irs)).float().div(255)[:, None].to(dev)
            o = net(y, ir)[0]
            if W_LOP is not None:
                wm = torch.from_numpy(np.stack(ws))[:, None].to(dev)
                l_fuse = 5 * ((1 - wm) * (o - y) ** 2 + wm * (o - ir) ** 2).mean()
            elif a.w_ir is None:
                l_fuse = 3 * F.mse_loss(o, y) + 2 * F.mse_loss(o, ir)
            else:
                l_fuse = 5 * ((1 - a.w_ir) * F.mse_loss(o, y) + a.w_ir * F.mse_loss(o, ir))
            l_grad = 9 * grad_loss(o, y, ir, kx, ky)
            if REF is not None:      # chưng cất kiểu SAGE: Teacher = ảnh ref
                rf = torch.from_numpy(np.stack(rs)).float().div(255)[:, None].to(dev)
                if wks:              # trọng số vùng thầy đã chỉnh
                    wt = torch.from_numpy(np.stack(wks))[:, None].to(dev)
                    mse_ref = (wt * (o - rf) ** 2).mean() / wt.mean()
                else:
                    mse_ref = F.mse_loss(o, rf)
                if a.chi_ref:        # chưng cất thuần: thay hẳn loss gốc
                    l_fuse = 5 * mse_ref
                    l_grad = a.w_grad_chi_ref * grad_loss(o, rf, rf, kx, ky)
                else:
                    l_fuse = l_fuse + a.w_ref * mse_ref
                    l_grad = l_grad + a.w_ref_grad * grad_loss(o, rf, rf, kx, ky)
            loss = 3 * l_fuse + 3 * l_grad
            l_vung = loss_vung(o, y, ir, ds) if a.lam_vung > 0 else o.new_zeros(())
            if a.lam_vung > 0:
                loss = loss + a.lam_vung * l_vung
            l_tach = loss_tach(o, y, ir, dt) if a.lam_tach > 0 else o.new_zeros(())
            if a.lam_tach > 0:
                loss = loss + a.lam_tach * l_tach
            l_den = loss_den(o, y, dd) if a.lam_den > 0 else o.new_zeros(())
            if a.lam_den > 0:
                loss = loss + a.lam_den * l_den
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở bước {buoc}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            buoc += 1
            tong["loss"] += loss.item(); tong["fuse"] += l_fuse.item(); tong["grad"] += l_grad.item()
            tong["vung"] += l_vung.item(); tong["tach"] += l_tach.item(); tong["den"] += l_den.item(); tong["n"] += 1
            if a.thu and buoc >= a.thu:
                dt = time.time() - t0
                n = tong["n"]
                print(f"Thử {buoc} bước: {dt:.1f}s -> ước tính cả bài {dt / buoc * a.epochs * so_batch / 2 / 60:.0f} phút,"
                      f" GPU {torch.cuda.max_memory_allocated() / 2**20 if dev == 'cuda' else 0:.0f} MB, loss {loss.item():.4f}"
                      f" | TB: 3*fuse {3 * tong['fuse'] / n:.4f}, 3*grad {3 * tong['grad'] / n:.4f},"
                      f" lam*vung {a.lam_vung * tong['vung'] / n:.4f} (vung {tong['vung'] / n:.4f}),"
                      f" lam*tach {a.lam_tach * tong['tach'] / n:.4f} (tach {tong['tach'] / n:.4f}),"
                      f" lam*den {a.lam_den * tong['den'] / n:.4f} (den {tong['den'] / n:.4f})")
                return
        n = tong.pop("n")
        dong = {"epoch": ep, "buoc": buoc, "lr": sch.get_last_lr()[0], **{k: v / n for k, v in tong.items()},
                "phut": (time.time() - t0) / 60}
        log.append(dong)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} bước {buoc} lr {dong['lr']:.2e} loss {dong['loss']:.4f} "
                  f"(fuse {dong['fuse']:.4f}, grad {dong['grad']:.4f}, vung {dong['vung']:.4f}, tach {dong['tach']:.4f}, den {dong['den']:.4f}) {dong['phut']:.1f} phút", flush=True)
    torch.save(net.state_dict(), STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    print(f"Lưu {STUDENT / (a.ten + '.pt')}")


if __name__ == "__main__":
    main()
