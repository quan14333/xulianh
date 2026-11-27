"""Bước 1: xuất ảnh đầu vào cho SegFormer theo từng biến thể, cho cả train, dev, test.

  vis  : ảnh VIS gốc
  ir   : IR xám, nhân 3 kênh
  tb   : Y = (Y_vis + IR)/2, màu CrCb của VIS  (ghép trung bình – đối chứng đơn giản, bài không có)
  clahe: Y = CLAHE(Y_vis), màu CrCb của VIS     (đối chứng "chỉ tăng tương phản", bài không có)
  sage : Y = SAGE Student pretrained, màu CrCb của VIS (giống output thật của SAGE)
  nosam: Y = Student train lại KHÔNG SAM/Teacher (E17, tai_lap/e17), checkpoint sage_student/nosam.pt
  tn   : M5 – trộn thích nghi từng điểm ảnh (CS406, không train):
         Y = (1-w)*Y_vis + w*IR,  w = clip(0,4 + 0,3*(1 - L) + 0,3*S, 0,3; 0,85)
         L = độ sáng cục bộ của VIS (Gauss sigma 21), S = độ nổi IR cục bộ = clip(4*(IR - Gauss(IR)), 0, 1)
         Ý: chỗ tối và chỗ IR nổi (người, xe nóng) lấy nhiều IR hơn; chỗ sáng gần TB/VIS.
  tbct : M8 – nền TB + chi tiết max (không train): nền = (blur31(Y_vis) + blur31(IR))/2,
         chi tiết = phần (ảnh − nền) có |.| lớn hơn giữa VIS và IR, màu CrCb của VIS.
  tbcd : M8b – như tbct nhưng chi tiết = chi tiết VIS + chi tiết IR (cộng, không chọn).
  vci  : M8c – Y = Y_vis + (IR − mean(IR)), cắt [0, 255] (nền cũng cộng).
  vcim : M8d – như M8c nhưng nén mềm (tanh) ngoài [40, 200] thay vì cắt cứng (chống cháy trắng).

Ví dụ: python xuat_anh.py --bien_the sage
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from chung import ANH, MAN, STUDENT, duong, thu_muc


CKPT = {"sage": "checkpoints.pt", "nosam": "nosam.pt", "nosam_s1": "nosam_s1.pt", "nosam_s2": "nosam_s2.pt"}  # biến thể chạy bằng mạng Student


def trong_so_tn(y, ir):
    """Trọng số IR từng điểm ảnh cho M5 (y, ir: uint8 HxW)."""
    yv, iv = y.astype(np.float32) / 255, ir.astype(np.float32) / 255
    L = cv2.GaussianBlur(yv, (0, 0), 21)
    S = np.clip(4 * (iv - cv2.GaussianBlur(iv, (0, 0), 21)), 0, 1)
    return np.clip(0.4 + 0.3 * (1 - L) + 0.3 * S, 0.3, 0.85)


def tb_chi_tiet(y, ir, k=31):
    """M8: ghép 2 tầng. Nền = trung bình nền VIS và nền IR (như TB, giữ IR cho phân đoạn);
    chi tiết = ở từng điểm lấy chi tiết có |.| lớn hơn (giữ cạnh nét, như loss gradient max của TDFusion/ReFusion).
    Nền = lọc trung bình k x k (k = 31 như GFF, Li 2013)."""
    yv, iv = y.astype(np.float32), ir.astype(np.float32)
    bv, bi = cv2.blur(yv, (k, k)), cv2.blur(iv, (k, k))
    dv, di = yv - bv, iv - bi
    d = np.where(np.abs(dv) >= np.abs(di), dv, di)
    return ((bv + bi) / 2 + d).round().clip(0, 255).astype(np.uint8)


def tb_cong_chi_tiet(y, ir, k=31):
    """M8b: như M8 nhưng CỘNG chi tiết hai nguồn thay vì chọn max (M8 rớt SCD 87,5% vì bỏ chi tiết của nguồn yếu hơn)."""
    yv, iv = y.astype(np.float32), ir.astype(np.float32)
    bv, bi = cv2.blur(yv, (k, k)), cv2.blur(iv, (k, k))
    return ((bv + bi) / 2 + (yv - bv) + (iv - bi)).round().clip(0, 255).astype(np.uint8)


def vis_cong_ir(y, ir):
    """M8c: nền cũng CỘNG (không lấy trung bình): F = (nền VIS + nền IR) + (chi tiết VIS + chi tiết IR) − mean(IR)
    = Y_vis + (IR − mean(IR)). Giữ độ sáng chung của VIS, cộng toàn bộ độ lệch của IR (M8, M8b rớt SCD vì F giống
    trung bình; SCD lớn nhất khi F ≈ VIS + IR). Cắt về [0, 255]."""
    yv, iv = y.astype(np.float32), ir.astype(np.float32)
    return (yv + iv - iv.mean()).round().clip(0, 255).astype(np.uint8)


def nen_mem(x, lo=40.0, hi=200.0):
    """Nén mềm hai đầu thay cho cắt cứng: giữ nguyên trong [lo, hi], ngoài khoảng đó ép dần (tanh) vào [0, lo] / [hi, 255].
    Đơn điệu, liên tục, giữ được chênh lệch (cửa sổ, biển báo) ở vùng sáng thay vì cắt phẳng về 255."""
    y = x.copy()
    t = x > hi
    y[t] = hi + (255 - hi) * np.tanh((x[t] - hi) / (255 - hi))
    t = x < lo
    y[t] = lo - lo * np.tanh((lo - x[t]) / lo)
    return y


def vis_cong_ir_mem(y, ir):
    """M8d: như M8c (Y_vis + IR − mean(IR)) nhưng NÉN MỀM vùng sáng/tối thay vì cắt [0, 255]
    (M8c cháy trắng mặt tiền nhà ban ngày, tới 14% điểm ảnh; biển báo bạc màu)."""
    yv, iv = y.astype(np.float32), ir.astype(np.float32)
    return nen_mem(yv + iv - iv.mean()).round().clip(0, 255).astype(np.uint8)


W_LOP = os.environ.get("SAGE_W_LOP", "D:/CS406/DoAn/Sage/tai_lap/scene_v1/w_lop.json")


def oracle_lop(y, ir, lab, nguoc=False, sigma=4.0):
    """ORACLE 22/11 (chỉ chẩn đoán, dùng NHÃN THẬT cả train lẫn val – không phải phương pháp):
    F = (1-w)Y + w IR, w theo lớp từ w_lop.json (luật lint), làm mờ Gauss sigma px cho bớt đường nối.
    nguoc=True: luật NGƯỢC w' = 0,8 - w (cùng đường nối, sai luật) – đối chứng phần lợi do lộ ranh giới nhãn."""
    import json
    w = json.load(open(W_LOP))
    lut = np.array([w[str(c)] for c in range(15)], np.float32)
    if nguoc:
        lut = 0.8 - lut
    wm = cv2.GaussianBlur(lut[lab], (0, 0), sigma)
    return ((1 - wm) * y.astype(np.float32) + wm * ir).round().clip(0, 255).astype(np.uint8)


def ghep_giu_mau(vis_bgr, yf, sigma=2.0, bu=8.0, k_min=0.5, k_max=3.0):
    """GIỮ MÀU (22/11): thay vì Y mới + CrCb VIS nguyên (bạc màu khi Y bị đẩy sáng: person S chỉ 77% VIS ở nosam),
    co giãn độ lệch màu (CrCb − 128) theo k = Y_ghép / Y_vis  ->  RGB ra ≈ k · RGB_vis: giữ sắc độ (hue) và
    độ bão hòa của VIS, như cách nhân hệ số chung của FCDFusion (arXiv 2408.01080) / Brovey trong pansharpening.
    k tính trên ảnh làm mờ Gauss sigma px, cộng 'bu' mức xám để không khuếch đại nhiễu màu ở vùng VIS gần đen, kẹp [k_min, k_max].
    Kênh Y GIỮ NGUYÊN -> chỉ số ghép (tính trên Y) không đổi."""
    ycc = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    yv = cv2.GaussianBlur(ycc[:, :, 0], (0, 0), sigma)
    yb = cv2.GaussianBlur(yf.astype(np.float32), (0, 0), sigma)
    k = np.clip((yb + bu) / (yv + bu), k_min, k_max)
    c = 128 + (ycc[:, :, 1:] - 128) * k[..., None]
    out = np.dstack([yf.astype(np.float32), c]).round().clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


VNC_GOC = "nosam_992"   # nền ảnh ghép của vnc (Student bỏ SAM train 992 ảnh, cùng nền với mốc B3 Dev228)


def cong_vnc(y, ir, k=31, a=15.0, b=30.0, no=9, sm=4.0, toi_lo=80.0, toi_hi=110.0, g_min=0.2):
    """VNC (23/11) – cổng g trong [g_min, 1] cho Y = Y_vis + g·(Y_nosam − Y_vis). KHÔNG nhãn, KHÔNG mạng phụ.
    - Cổng nhiệt: s = top-hat trắng (elip k x k) của IR đã làm mịn 2 px = vật NHỎ ấm hơn nền (người); vùng lớn
      (tòa nhà giáp trời lạnh) bị loại – bản đầu dùng IR − Gauss25(IR) mở sai ở mép nhà/trời (xem hinh_cai_tien/vnc_train_00341).
      g = clip((s − a)/b), nới vùng (dilate no x no), làm mờ Gauss sm px. Tham số CHỌN TRÊN ẢNH TRAIN SÁNG
      (496 ảnh, scene_v1 train 992): tách người tốt nhất (g người 0,76; tòa nhà 0,17, truck 0,15, đèn 0,31). Không nhìn Dev.
    - Cổng tối (theo ảnh): mean(Y_vis) <= toi_lo -> 1 (dùng nosam), >= toi_hi -> 0, tuyến tính ở giữa.
      toi_lo ≈ P10 độ sáng ảnh train (79), toi_hi dưới trung vị train (115). Ghi rõ: mốc 100 ở giữa có tham khảo
      phân tích Dev228 (ảnh sáng/tối) trước đó.
    - g_min = 0,2: mức nhỏ nhất qua rào MS-SSIM >= 99% SAGE trên 83 ảnh train (0: 98,6%; 0,1: 99,0%; 0,2: 99,3%).
    Lý do (phân tích B3 Dev228 3 seed): ở ảnh sáng VIS thắng ảnh ghép ở đèn/truck/bus/vỉa hè (lớp hiếm, diện mạo bị
    ảnh ghép đổi), còn person cần IR; ở ảnh tối ảnh ghép thắng."""
    yv = y.astype(np.float32)
    irs = cv2.GaussianBlur(ir, (0, 0), 2)
    s = cv2.morphologyEx(irs, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    g = np.clip((s - a) / b, 0, 1)
    g = cv2.GaussianBlur(cv2.dilate(g, np.ones((no, no), np.uint8)), (0, 0), sm)
    toi = float(np.clip((toi_hi - yv.mean()) / (toi_hi - toi_lo), 0, 1))
    return np.maximum(np.maximum(g, toi), g_min)


def vis_neo_cong(y, ir, yn):
    g = cong_vnc(y, ir)
    yv = y.astype(np.float32)
    return (yv + g * (yn.astype(np.float32) - yv)).round().clip(0, 255).astype(np.uint8)


def tron_thich_nghi(y, ir):
    w = trong_so_tn(y, ir)
    return ((1 - w) * y.astype(np.float32) + w * ir).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", required=True,
                    help="vis|ir|tb|sage|clahe|nosam*|tn, hoặc tên checkpoint sage_student/<tên>.pt (M4/M6, có đầu 'dau.')")
    ap.add_argument("--gioi_han", type=int, default=0, help="chỉ để thử: xuất N ảnh đầu")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in ("train", "dev", "test")])
    if a.gioi_han:
        df = df.head(a.gioi_han)
    goc = a.bien_the
    # <biến thể>_mau  = Y của <biến thể> + ghép màu giữ bão hòa, k kẹp [0,5; 3] (bản 22/11 chiều, đã chấm B0 3 seed)
    # <biến thể>_mau2 = như trên nhưng k kẹp [1; 3]: CHỈ trả lại màu, không bao giờ làm nhạt màu dưới mức VIS.
    #   Lý do (đo 22/11 tối, val151): ở _mau, ảnh sáng có 69% điểm ảnh k < 0,9 (trung vị k 0,83) -> _mau làm NHẠT màu
    #   ảnh sáng, ngược ý định; nhóm sáng thua nosam cả 3 seed (−0,64), vỉa hè/road/car thua cả 3 seed.
    k_min = None
    if a.bien_the.endswith("_mau2"):
        goc, k_min = a.bien_the[:-5], 1.0
    elif a.bien_the.endswith("_mau"):
        goc, k_min = a.bien_the[:-4], 0.5
    giu_mau = k_min is not None
    la_vnc = goc == "vnc"
    if la_vnc:
        goc = VNC_GOC                                   # chạy Student nền rồi trộn với VIS theo cổng
    if goc not in CKPT and goc not in ("vis", "ir", "tb", "clahe", "tn", "tbct", "tbcd", "vci", "vcim", "orc_dung", "orc_nguoc"):
        assert (STUDENT / f"{goc}.pt").exists(), f"không có biến thể/checkpoint {goc}"
        CKPT[goc] = f"{goc}.pt"
    if goc in CKPT:
        import torch
        sys.path.insert(0, str(STUDENT))
        from model import Network
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = Network()
        sd = torch.load(STUDENT / CKPT[goc], map_location="cpu", weights_only=True)
        s_tb = sd.pop("_tb_scale", None)                # Nền 2 (cai_tien/tb_sua.py): Y = TB + s * Student
        s_tb = None if s_tb is None else float(s_tb)
        la_vung = any(k.startswith("vung_") for k in sd)   # S-vùng (23/11): nosam đóng băng + mắt + đầu vùng
        if la_vung:
            from model_vung import NetworkVung
            net = NetworkVung(sd["goc.module1.0.weight"].shape[0], rong=sd["vung_dau.0.weight"].shape[0])
        dim = sd["goc.module1.0.weight" if la_vung else "module1.0.weight"].shape[0]   # S-rộng (21/11): số kênh từ checkpoint
        if dim != 32 and not la_vung:
            net = Network(dim)
        if any(k.startswith("mat_") for k in sd):          # S-mắt (23/11): Student + ResNet18 đóng băng
            from model_mat import NetworkMat
            net = NetworkMat(dim)
        if any(k.startswith("xa_") for k in sd):           # S-xa (21/11): Student + nhánh ngữ cảnh rộng
            from model_xa import NetworkXa
            them = len({k.split(".")[1] for k in sd if k.startswith("xa_xuong.")})
            net = NetworkXa(dim, them)
        if any(k.startswith("dau.") for k in sd):          # M4/M6: Student + đầu trộn (DauTron)
            day = Path(__file__).resolve().parent      # Kaggle: train_student_seg.py nằm cùng thư mục
            sys.path.insert(0, str(day if (day / "train_student_seg.py").exists() else day.parent / "cai_tien"))
            from train_student_seg import DauTron
            net = DauTron(net)
        net.load_state_dict(sd, strict=True)
        net.to(dev).eval()
        t2 = lambda z: torch.from_numpy(z).float().div(255)[None, None].to(dev)

    n = 0
    for r in df.itertuples():
        out = ANH / a.bien_the / thu_muc(r) / f"{r.image_id}.png"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        vis = cv2.imread(str(duong(r.vis_path)))
        ir = cv2.cvtColor(cv2.imread(str(duong(r.ir_path))), cv2.COLOR_BGR2GRAY)
        if goc == "vis":
            img = vis
        elif goc == "ir":
            img = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        else:
            ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            y = ycc[:, :, 0]
            if goc == "clahe":
                yf = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
            elif goc == "tb":
                yf = ((y.astype(np.float64) + ir) / 2).round().astype(np.uint8)
            elif goc == "vcim":
                yf = vis_cong_ir_mem(y, ir)
            elif goc == "vci":
                yf = vis_cong_ir(y, ir)
            elif goc == "tbcd":
                yf = tb_cong_chi_tiet(y, ir)
            elif goc == "tbct":
                yf = tb_chi_tiet(y, ir)
            elif goc in ("orc_dung", "orc_nguoc"):
                lab = cv2.imread(str(duong(r.label_path)), cv2.IMREAD_GRAYSCALE)
                yf = oracle_lop(y, ir, lab, nguoc=goc == "orc_nguoc")
            elif goc == "tn":
                yf = tron_thich_nghi(y, ir)
            else:
                with torch.inference_mode():
                    o = net(t2(y), t2(ir))[0]
                    if s_tb is not None:
                        o = (t2(y) + t2(ir)) / 2 + s_tb * o
                    yf = (o.clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
                if la_vnc:
                    yf = vis_neo_cong(y, ir, yf)
            img = (ghep_giu_mau(vis, yf, k_min=k_min) if giu_mau
                   else cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR))
        cv2.imwrite(str(out), img)
        n += 1
    print(f"{a.bien_the}: ghi {n} ảnh mới vào {ANH / a.bien_the}", flush=True)


if __name__ == "__main__":
    main()
