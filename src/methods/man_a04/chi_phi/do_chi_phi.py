"""E18 – Đo chi phí: SAGE có SAM, SAGE bỏ SAM (E17), Ghép TB (và sau này: phương pháp nhóm).

Thầy yêu cầu: so có SAM / không SAM / phương pháp nhóm; chỉ ra SAM nặng ở BƯỚC NÀO; có đáng không;
phải tự chạy lại baseline và bấm giờ. B3 (train SegFormer chấm mIoU) chỉ là thước đo, KHÔNG tính vào chi phí.

Bám đúng external/SAGE_IVIF:
  - SAM tạo mask: SamAutomaticMaskGenerator(points_per_side=128, crop_n_layers=1, ...) trên CẢ IR và VIS
    (dataset/dataset_teacher_FMB.py), chạy 1 lần cho 1.220 ảnh train rồi lưu cache.
  - Train xen kẽ (train.py): batch chẵn -> bước Student, batch lẻ -> bước Teacher. 1.220 ảnh, batch 4
    -> 305 batch/epoch = 153 bước Student + 152 bước Teacher; 100 epoch.
      + Bước Student: Teacher forward (no_grad) + Student fwd/bwd + Losses.cal (L_cs gọi encoder SAM 2 tầng)
        + cosine đặc trưng.
      + Bước Teacher: Teacher fwd/bwd + X-Decoder Focal-L (L_seg) + Student forward + Losses.cal (L_cs).
  - Bỏ SAM (E17): chỉ bước Student với loss nội dung, 153 bước/epoch.
  - Không đo: ghi ảnh/tensorboard mỗi bước (report_freq=1 trong code gốc) và đọc dữ liệu – giống nhau
    cho mọi phương pháp, chỉ đo phần tính trên GPU. Ghi rõ điều này khi báo cáo.

Các phần (chạy riêng được, kết quả gộp vào ket_qua/<may>.json):
  tinh      : số tham số + GFLOPs (fvcore, 1 phép nhân-cộng = 1 FLOP) + dung lượng checkpoint
  sam_mask  : thời gian + VRAM đỉnh khi SAM tạo mask, N ảnh -> ước tính 1.220 ảnh; số mask & dung lượng cache
  train     : giây/bước + VRAM đỉnh cho bước Student, bước Teacher (có/không X-Decoder), bước bỏ SAM
              -> ước tính tổng 100 epoch
  chay      : ms/ảnh lúc chạy (600x800, batch 1): Student, Ghép TB, a04 (27/11, nếu có tbsua_992.pt)

27/11 – thêm phương án nhóm a04 = TB + 0,4·(tbsua_992 − TB), tbsua_992 = clamp(TB + s·Student(Y, IR)), s đọc từ checkpoint:
  chay  : a04_fp32 (Student tbsua_992 + phép trộn), cùng tiến trình với Student SAGE để so công bằng
  train : --buoc tbsua = bước train tbsua (như cai_tien/tb_sua.py: 150·MSE(F, TB) + 27·Grad, batch 4, crop 192x256);
          lượng tính y hệt bước bỏ SAM, chỉ khác loss. Ước tính cho 1.220 ảnh (như mọi dòng) và cho 992 ảnh (số thật dùng).

Ví dụ:
  python do_chi_phi.py --phan tinh chay
  python do_chi_phi.py --phan sam_mask --so_anh 5
  python do_chi_phi.py --phan sam_mask --so_anh 5 --sam_cauhinh macdinh
  python do_chi_phi.py --phan train --buoc nosam --so_buoc 20        (mỗi loại bước 1 tiến trình)
  python do_chi_phi.py --phan train --buoc teacher --batch 4 2 1
  python do_chi_phi.py --phan train --buoc teacher --batch 2 1 --xdecoder  (cần detectron2 + checkpoint X-Decoder)
Đường dẫn đổi bằng biến môi trường: SAGE_CODE (repo SAGE_IVIF), SAGE_CKPT (thư mục checkpoint ngoài),
SAGE_FMB (dữ liệu FMB, xem b3/chung.py).
"""
import argparse
import json
import math
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DAY = Path(__file__).resolve().parent
CODE = Path(os.environ.get("SAGE_CODE", DAY.parents[1] / "external" / "SAGE_IVIF"))
CKPT = Path(os.environ.get("SAGE_CKPT", DAY.parent / "_ckpt_ngoai"))
B3 = DAY.parent / "b3" if (DAY.parent / "b3" / "chung.py").exists() else DAY
sys.path.insert(0, str(B3))
sys.path.insert(0, str(CODE))
from chung import MAN, duong  # noqa: E402

DEV = "cuda"
N_TRAIN, BATCH, EPOCH = 1220, 4, 100
SO_BATCH = math.ceil(N_TRAIN / BATCH)            # 305
BUOC_S = (SO_BATCH + 1) // 2                     # 153 bước Student / epoch (batch chẵn)
BUOC_T = SO_BATCH // 2                           # 152 bước Teacher / epoch (batch lẻ)
CROP = (192, 256)
MASK_NUM = 4
TBSUA = next((p for p in (DAY / "tbsua_992.pt", DAY.parent / "b3" / "sage_student" / "tbsua_992.pt") if p.exists()), None)
ALPHA_A04, N_TBSUA = 0.4, 992


# ----------------------------------------------------------------------------- tiện ích
def dong_bo():
    torch.cuda.synchronize()


def bat_dau_do():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def vram_dinh_mb():
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def mb(p):
    p = Path(p)
    return round(p.stat().st_size / 2**20, 1) if p.exists() else None


def dem_tham_so(m):
    return sum(p.numel() for p in m.parameters())


def gflops(m, dau_vao):
    """GFLOPs theo fvcore (1 nhân-cộng = 1 FLOP, như cách đếm phổ biến trong các bài ghép ảnh)."""
    from fvcore.nn import FlopCountAnalysis
    fa = FlopCountAnalysis(m, dau_vao)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return round(fa.total() / 1e9, 3)


def bam_gio(ham, so_lan, khoi_dong):
    """Trả về (giây trung bình mỗi lần, độ lệch chuẩn) sau khi chạy khởi động."""
    for _ in range(khoi_dong):
        ham()
    dong_bo()
    t = []
    for _ in range(so_lan):
        t0 = time.perf_counter()
        ham()
        dong_bo()
        t.append(time.perf_counter() - t0)
    return float(np.mean(t)), float(np.std(t))


def anh_train(k, rng):
    """k cặp (Y, IR) uint8 600x800 lấy từ manifest train (không đụng test)."""
    import pandas as pd
    df = pd.read_csv(MAN / "train.csv", dtype={"image_id": str})
    chon = df.iloc[rng.choice(len(df), size=k, replace=False)]
    ys, irs, vis_bgr, ir_bgr = [], [], [], []
    for _, r in chon.iterrows():
        v = cv2.imread(str(duong(r.vis_path)))
        i = cv2.imread(str(duong(r.ir_path)))
        vis_bgr.append(v), ir_bgr.append(i)
        ys.append(cv2.cvtColor(v, cv2.COLOR_BGR2YCrCb)), irs.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    return list(chon.image_id), ys, irs, vis_bgr, ir_bgr


# ----------------------------------------------------------------------------- mô hình
def tao_student():
    from model_sub.model import Network
    return Network()


def tao_teacher():
    from model_main.model import Network
    return Network(mask_num=MASK_NUM)


def tao_sam_encoder_lcs():
    """Encoder SAM dùng trong L_cs: ViT-B nhưng chỉ 2 tầng (model_sub/common.py, load_sam)."""
    from SAM import ImageEncoderViT
    enc = ImageEncoderViT(depth=2, embed_dim=768, img_size=1024, mlp_ratio=4,
                          norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12, patch_size=16,
                          qkv_bias=True, use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11], window_size=14,
                          out_chans=256)
    sd = torch.load(CKPT / "sam_vit_b_01ec64.pth", map_location="cpu")
    sd = {k[len("image_encoder."):]: v for k, v in sd.items() if k.startswith("image_encoder.")}
    md = enc.state_dict()
    md.update({k: v for k, v in sd.items() if k in md})
    enc.load_state_dict(md)
    return enc


def tao_losses():
    """Losses của Student như code gốc, nhưng nạp encoder SAM từ CKPT thay cho đường dẫn cứng /mnt/..."""
    import model_sub.common as common
    common.load_sam = tao_sam_encoder_lcs
    from model_sub.losses import Losses
    return Losses().to(DEV)


def cosine_loss(a, b):
    return sum(1 - F.cosine_similarity(x.flatten(1), y.flatten(1), dim=1).mean() for x, y in zip(a, b))


def tao_xdecoder():
    """X-Decoder Focal-L như train.py (process_teacher_batch). Cần detectron2; chạy với cwd = CODE/xdecoder."""
    try:
        import mpi4py  # noqa: F401
    except ImportError:                                    # chỉ dùng khi train nhiều máy; 1 GPU không cần
        import types
        sys.modules["mpi4py"] = types.SimpleNamespace(MPI=None)
    cu = os.getcwd()
    os.chdir(CODE / "xdecoder")
    sys.path.insert(0, str(CODE / "xdecoder"))
    try:
        from xdecoder.infer_semseg import set_sys_args
        from xdecoder.utils.arguments import load_opt_command
        from xdecoder.modeling.BaseModel import BaseModel
        from xdecoder.modeling import build_model
        set_sys_args()
        opt, _ = load_opt_command([])
        opt["local_rank"] = 0
        opt["device"] = "cuda"
        opt.setdefault("MODEL", {}).setdefault("TEXT", {})
        opt["MODEL"]["TEXT"]["TOKENIZER"] = "clip"
        opt["MODEL"]["TEXT"]["PRETRAINED_TOKENIZER"] = "./local_clip_model"
        m = BaseModel(opt, build_model(opt)).from_pretrained("./xdecoder_focall_last.pt").eval().to(DEV)
        for p in m.parameters():
            p.requires_grad = False
        m.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            ["road", "sidewalk", "building", "lamp", "sign", "vegetation", "sky", "person", "car",
             "truck", "bus", "motocycle", "bicycle", "pole", "background"], is_eval=True)
        m.model.sem_seg_head.num_classes = 14
        return m
    finally:
        os.chdir(cu)


# ----------------------------------------------------------------------------- phần: tinh
def phan_tinh(a, kq):
    st, te = tao_student().to(DEV).eval(), tao_teacher().to(DEV).eval()
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(CKPT / "sam_vit_b_01ec64.pth")).to(DEV).eval()
    enc2 = tao_sam_encoder_lcs().to(DEV).eval()
    h, w = CROP
    o = {}
    o["tham_so"] = {
        "student": dem_tham_so(st),
        "teacher": dem_tham_so(te),
        "sam_vit_b_tong": dem_tham_so(sam),
        "sam_vit_b_image_encoder": dem_tham_so(sam.image_encoder),
        "sam_vit_b_prompt_mask_decoder": dem_tham_so(sam.prompt_encoder) + dem_tham_so(sam.mask_decoder),
        "sam_encoder_2_tang_Lcs": dem_tham_so(enc2),
    }
    with torch.no_grad():
        y1 = lambda hh, ww: torch.rand(1, 1, hh, ww, device=DEV)  # noqa: E731
        o["gflops"] = {
            "student_600x800": gflops(st, (y1(600, 800), y1(600, 800))),
            "student_768x1024_(co_anh_M3FD,_bai_ghi_52,06)": gflops(st, (y1(768, 1024), y1(768, 1024))),
            "student_crop_192x256": gflops(st, (y1(h, w), y1(h, w))),
            "teacher_crop_192x256": gflops(te, (y1(h, w), y1(h, w), torch.rand(1, MASK_NUM, h, w, device=DEV),
                                                torch.rand(1, MASK_NUM, h, w, device=DEV))),
            "sam_vit_b_image_encoder_1024x1024_(1_lan)": gflops(sam.image_encoder,
                                                                (torch.rand(1, 3, 1024, 1024, device=DEV),)),
            "sam_encoder_2_tang_Lcs_crop_192x256_(1_lan)": gflops(enc2, (torch.rand(1, 3, h, w, device=DEV),)),
        }
    # L_cs: 1 lần compute_fake_C(output) + mỗi mask (4 VIS + 4 IR): 1 compute_fake_C + 2 compute_vgg_loss
    o["so_lan_goi_encoder_SAM_moi_Losses_cal"] = 1 + 2 * MASK_NUM * 3
    xd = CODE / "xdecoder" / "xdecoder_focall_last.pt"
    if xd.exists():                                        # đếm thẳng từ checkpoint, không cần detectron2
        sd = torch.load(xd, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        o["tham_so"]["xdecoder_focall_checkpoint"] = int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
        del sd
    o["dung_luong_mb"] = {
        "student_checkpoint": mb(CODE / "checkpoints" / "checkpoints.pt"),
        "sam_vit_b": mb(CKPT / "sam_vit_b_01ec64.pth"),
        "xdecoder_focall_last": mb(CODE / "xdecoder" / "xdecoder_focall_last.pt"),
    }
    kq["tinh"] = o
    print(json.dumps(o, indent=1, ensure_ascii=False))


# ----------------------------------------------------------------------------- phần: sam_mask
def phan_sam_mask(a, kq):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    rng = np.random.default_rng(a.seed)
    _, _, _, vis_bgr, ir_bgr = anh_train(a.so_anh + 1, rng)
    sam = sam_model_registry["vit_b"](checkpoint=str(CKPT / "sam_vit_b_01ec64.pth")).to(DEV)
    if a.sam_cauhinh == "goc":                             # đúng dataset_teacher_FMB.py
        gen = SamAutomaticMaskGenerator(model=sam, points_per_side=128, pred_iou_thresh=0.86,
                                        stability_score_thresh=0.92, crop_n_layers=1,
                                        crop_n_points_downscale_factor=2, min_mask_region_area=100,
                                        output_mode="binary_mask")
        mo_ta = "points_per_side=128, crop_n_layers=1, như dataset_teacher_FMB.py"
    else:                                                  # mặc định của segment_anything: mức rẻ hợp lý
        gen = SamAutomaticMaskGenerator(model=sam, output_mode="binary_mask")
        mo_ta = "mặc định segment_anything: points_per_side=32, crop_n_layers=0"
    gen.generate(ir_bgr[-1])                                   # khởi động (không tính)
    bat_dau_do()
    t_ir, t_vis, so_mask, byte_cache = [], [], [], []
    for v, i in zip(vis_bgr[:-1], ir_bgr[:-1]):
        dong_bo(); t0 = time.perf_counter()
        mi = gen.generate(i)
        dong_bo(); t1 = time.perf_counter()
        mv = gen.generate(v)
        dong_bo(); t2 = time.perf_counter()
        t_ir.append(t1 - t0), t_vis.append(t2 - t1)
        n = min(20, len(mi), len(mv))                          # cache_mask_num = 20 trong train.py
        so_mask.append((len(mi), len(mv), n))
        byte_cache.append(2 * n * mi[0]["segmentation"].nbytes if n else 0)
        print(f"  ảnh: IR {t1 - t0:.1f}s ({len(mi)} mask), VIS {t2 - t1:.1f}s ({len(mv)} mask)", flush=True)
    moi_anh = float(np.mean(t_ir) + np.mean(t_vis))
    o = {
        "so_anh_do": a.so_anh,
        "giay_moi_anh_IR": round(float(np.mean(t_ir)), 2),
        "giay_moi_anh_VIS": round(float(np.mean(t_vis)), 2),
        "giay_moi_cap": round(moi_anh, 2),
        "uoc_tinh_gio_1220_cap": round(moi_anh * N_TRAIN / 3600, 2),
        "vram_dinh_mb": vram_dinh_mb(),
        "so_mask_IR_VIS_luu": so_mask,
        "uoc_tinh_cache_GB_1220_cap": round(float(np.mean(byte_cache)) * N_TRAIN / 2**30, 2),
        "tham_so_generator": mo_ta,
    }
    kq["sam_mask" + ("" if a.sam_cauhinh == "goc" else "_macdinh")] = o
    print(json.dumps(o, indent=1, ensure_ascii=False))


# ----------------------------------------------------------------------------- phần: train
def lo_du_lieu(rng, ys, irs, batch=BATCH):
    """Một batch crop 192x256 từ ảnh thật. Mảnh SAM giả lập bằng vùng chữ nhật ngẫu nhiên:
    nội dung mask không đổi lượng tính toán (code chỉ bỏ qua mask rỗng)."""
    h, w = CROP
    Y, IR, YM, IRM, CB, CR = [], [], [], [], [], []
    for _ in range(batch):
        k = rng.integers(len(ys))
        i0, j0 = rng.integers(0, 600 - h + 1), rng.integers(0, 800 - w + 1)
        ycc = ys[k][i0:i0 + h, j0:j0 + w].astype(np.float32) / 255
        ir = irs[k][i0:i0 + h, j0:j0 + w].astype(np.float32) / 255
        # giữ đúng cách đặt tên của dataset_teacher_FMB.py: 'cb' = kênh 1, 'cr' = kênh 2 của cv2 YCrCb
        Y.append(ycc[..., 0]), CB.append(ycc[..., 1]), CR.append(ycc[..., 2]), IR.append(ir)
        ym, irm = [], []
        for _ in range(MASK_NUM):
            m = np.zeros((h, w), np.float32)
            a0, b0 = rng.integers(0, h // 2), rng.integers(0, w // 2)
            m[a0:a0 + h // 2, b0:b0 + w // 2] = 1
            ym.append(ycc[..., 0] * m), irm.append(ir * m)
        YM.append(np.stack(ym)), IRM.append(np.stack(irm))
    t = lambda x, c=True: torch.from_numpy(np.stack(x))[:, None if c else slice(None)].to(DEV)  # noqa: E731
    return t(Y), t(IR), t(YM, False), t(IRM, False), t(CB), t(CR)


def phan_train(a, kq):
    """Mỗi lần gọi chỉ đo MỘT loại bước (--buoc) để VRAM đỉnh không lẫn mô hình khác.
    Notebook gọi mỗi loại bước trong một tiến trình python riêng."""
    assert len(a.buoc) == 1, "train: mỗi tiến trình chỉ đo 1 loại bước (--buoc student|teacher|nosam)"
    loai = a.buoc[0]
    rng = np.random.default_rng(a.seed)
    _, ys, irs, _, _ = anh_train(16, rng)
    o = kq.setdefault("train", {})
    o.update({"so_buoc_do": a.so_buoc, "crop": CROP, "batch_goc": BATCH, "buoc_student_moi_epoch": BUOC_S,
              "buoc_teacher_moi_epoch": BUOC_T, "epoch": EPOCH,
              "cach_do": "mỗi loại bước một tiến trình riêng (VRAM đỉnh không lẫn)"})

    if loai in ("nosam", "tbsua"):                          # = E17 / tb_sua.py: chỉ loss nội dung, không nạp SAM/Teacher
        nosam = tao_student().to(DEV).train()
        if loai == "tbsua":                                 # tb_sua.tao_mang: lớp cuối khởi tạo 0
            torch.nn.init.zeros_(nosam.final_decoder[0].weight); torch.nn.init.zeros_(nosam.final_decoder[0].bias)
        opt_n = torch.optim.Adam(nosam.parameters(), lr=2e-3)
        from model_sub.common import GradientLoss
        gl = GradientLoss().to(DEV)

        def ham(b):
            def buoc():
                y, ir, *_ = lo_du_lieu(rng, ys, irs, b)
                if loai == "tbsua":
                    tb = (y + ir) / 2
                    out = (tb + 0.5 * nosam(y, ir)[0]).clamp(0, 1)
                    l = 150 * F.mse_loss(out, tb) + 27 * gl(out, y, ir)
                else:
                    out = nosam(y, ir)[0]
                    l = 3 * (3 * F.mse_loss(out, y) + 2 * F.mse_loss(out, ir)) + 3 * 9 * gl(out, y, ir)
                opt_n.zero_grad(); l.backward(); opt_n.step()
            return buoc
        opts = [opt_n]
    else:
        import utils
        st, te = tao_student().to(DEV).train(), tao_teacher().to(DEV).train()
        opt_s = torch.optim.Adam(st.parameters(), lr=2e-3)
        opt_t = torch.optim.Adam(te.parameters(), lr=5e-4)
        losses = tao_losses()
        opts = [opt_s, opt_t]
        xdec = tao_xdecoder() if (a.xdecoder and loai == "teacher") else None
        from model_main.segment_loss import calculate_loss

        def ham(b):
            def buoc_student():                            # = process_student_batch (bỏ ghi ảnh/log)
                y, ir, ym, irm, cb, cr = lo_du_lieu(rng, ys, irs, b)
                with torch.no_grad():
                    o_t, f_t = te(y, ir, ym, irm)
                o_s, f_s = st(y, ir)
                l, *_ = losses.cal(o_s, y, ir, o_t, ym, irm)
                l = l + 0.3 * cosine_loss(f_s, f_t)
                opt_s.zero_grad(); l.backward(); opt_s.step()

            def buoc_teacher():                            # = process_teacher_batch (bỏ ghi ảnh/log)
                y, ir, ym, irm, cb, cr = lo_du_lieu(rng, ys, irs, b)
                o_t, f_t = te(y, ir, ym, irm)
                l1, l2 = te.loss_cal(o_t, y, ir)
                l = l1 + l2
                if xdec is not None:
                    mau = utils.YCrCb2RGB(torch.cat((o_t, cb, cr), dim=1))
                    ra = []
                    for j in range(mau.size(0)):
                        x = F.interpolate(torch.clamp(mau[j:j + 1], 0, 1), size=(384, 384), mode="bicubic",
                                          align_corners=False) * 255.0
                        s = xdec([{"image": x.squeeze(0), "height": 384, "width": 384}])[-1]["sem_seg"]
                        ra.append(F.interpolate(s[None] if s.dim() == 3 else s, size=CROP, mode="bilinear",
                                                align_corners=False))
                    nhan = torch.randint(0, 14, (b, *CROP), device=DEV)
                    l = l + 0.3 * calculate_loss(torch.cat(ra), nhan, device=DEV)
                with torch.no_grad():
                    o_s, f_s = st(y, ir)
                ls, *_ = losses.cal(o_t, y, ir, o_s, ym, irm)
                l = l + 0.1 * ls + 0.01 * cosine_loss(f_t, f_s)
                opt_t.zero_grad(); l.backward(); opt_t.step()
            return buoc_student if loai == "student" else buoc_teacher

    ten_goc = {"student": "sage_buoc_student", "teacher": "sage_buoc_teacher", "nosam": "nosam_buoc",
               "tbsua": "tbsua_buoc"}[loai]
    if loai == "teacher" and a.xdecoder:
        ten_goc += "_xdecoder"
    for b in a.batch:
        ten = ten_goc + ("" if b == BATCH else f"_b{b}")
        bat_dau_do()
        try:
            tb, sd = bam_gio(ham(b), a.so_buoc, 3)
            o[ten] = {"batch": b, "giay_moi_buoc": round(tb, 4), "sd": round(sd, 4), "vram_dinh_mb": vram_dinh_mb()}
        except torch.cuda.OutOfMemoryError:
            o[ten] = {"batch": b, "loi": "HẾT VRAM", "vram_dinh_mb": vram_dinh_mb()}
            for m in opts:
                m.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(ten, o[ten], flush=True)
    tong_hop_train(o)
    print(json.dumps(o, indent=1, ensure_ascii=False))


def gio_buoc_b4(o, goc):
    """Giây/bước ở batch 4: đo thẳng nếu có; nếu hết VRAM thì ngoại suy tuyến tính từ batch nhỏ lớn nhất đo được."""
    r = o.get(goc, {})
    if r.get("giay_moi_buoc"):
        return r["giay_moi_buoc"], "đo thẳng"
    for b in (3, 2, 1):
        r = o.get(f"{goc}_b{b}", {})
        if r.get("giay_moi_buoc"):
            return r["giay_moi_buoc"] * BATCH / b, f"ngoại suy từ batch {b} (x{BATCH}/{b})"
    return None, None


def tong_hop_train(o):
    gs, _ = gio_buoc_b4(o, "sage_buoc_student")
    gn, _ = gio_buoc_b4(o, "nosam_buoc")
    if gn:
        o["uoc_tinh_gio_train_nosam"] = round(EPOCH * BUOC_S * gn / 3600, 3)
    gtb, _ = gio_buoc_b4(o, "tbsua_buoc")
    if gtb:
        o["uoc_tinh_gio_train_tbsua_1220"] = round(EPOCH * BUOC_S * gtb / 3600, 3)
        o["uoc_tinh_gio_train_tbsua_992"] = round(EPOCH * ((math.ceil(N_TBSUA / BATCH) + 1) // 2) * gtb / 3600, 3)
    if gs:
        o["uoc_tinh_gio_buoc_student_sage"] = round(EPOCH * BUOC_S * gs / 3600, 2)
    for goc, nhan in (("sage_buoc_teacher", "chua_xdecoder"), ("sage_buoc_teacher_xdecoder", "co_xdecoder")):
        gt, cach = gio_buoc_b4(o, goc)
        if gs and gt:
            o[f"uoc_tinh_gio_train_sage_{nhan}"] = {
                "gio": round(EPOCH * (BUOC_S * gs + BUOC_T * gt) / 3600, 2), "buoc_teacher": cach}


# ----------------------------------------------------------------------------- phần: chay
def phan_chay(a, kq):
    rng = np.random.default_rng(a.seed)
    _, ys, irs, _, _ = anh_train(4, rng)
    st = tao_student().to(DEV).eval()
    st.load_state_dict(torch.load(CODE / "checkpoints" / "checkpoints.pt", map_location=DEV))
    y = torch.from_numpy(ys[0][..., 0]).float().div(255)[None, None].to(DEV)
    ir = torch.from_numpy(irs[0]).float().div(255)[None, None].to(DEV)
    o = {"kich_thuoc": [600, 800], "batch": 1, "so_lan": a.so_lan}
    with torch.no_grad():
        cac = [("student_sage_fp32", lambda: st(y, ir)), ("ghep_tb", lambda: (y + ir) / 2)]
        if TBSUA is not None:                               # 27/11: a04 = TB + 0,4·(tbsua_992 − TB)
            sd = torch.load(TBSUA, map_location=DEV, weights_only=True)
            s_tb = float(sd.pop("_tb_scale"))
            ts = tao_student().to(DEV).eval()
            ts.load_state_dict(sd)
            o["a04"] = {"checkpoint": TBSUA.name, "s": s_tb, "alpha": ALPHA_A04}

            def a04():
                tb = (y + ir) / 2
                u = (tb + s_tb * ts(y, ir)[0]).clamp(0, 1)
                return tb + ALPHA_A04 * (u - tb)
            cac.append(("a04_fp32", a04))
        for ten, ham in cac:
            bat_dau_do()
            tb, sd = bam_gio(ham, a.so_lan, 20)
            o[ten] = {"ms_moi_anh": round(tb * 1000, 3), "sd_ms": round(sd * 1000, 3), "vram_dinh_mb": vram_dinh_mb()}
            print(ten, o[ten], flush=True)
    kq["chay"] = o


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", nargs="+", default=["tinh", "chay"],
                    choices=["tinh", "sam_mask", "train", "chay"])
    ap.add_argument("--so_anh", type=int, default=5, help="sam_mask: số cặp ảnh đo")
    ap.add_argument("--so_buoc", type=int, default=20, help="train: số bước đo mỗi loại")
    ap.add_argument("--so_lan", type=int, default=200, help="chay: số lần đo")
    ap.add_argument("--xdecoder", action="store_true", help="train: gồm X-Decoder trong bước Teacher")
    ap.add_argument("--buoc", nargs="+", default=["nosam"], choices=["student", "teacher", "nosam", "tbsua"],
                    help="train: loại bước đo (mỗi tiến trình 1 loại)")
    ap.add_argument("--batch", nargs="+", type=int, default=[BATCH],
                    help="train: các batch thử, vd 4 2 1 (hết VRAM ở 4 thì ngoại suy từ batch nhỏ)")
    ap.add_argument("--sam_cauhinh", default="goc", choices=["goc", "macdinh"],
                    help="sam_mask: goc = như code SAGE; macdinh = mặc định segment_anything")
    ap.add_argument("--may", default=None, help="tên máy cho file kết quả (mặc định: tên GPU)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "cần GPU"
    torch.manual_seed(a.seed)
    may = a.may or torch.cuda.get_device_name(0).replace(" ", "_")
    ra = DAY / "ket_qua" / f"{may}.json"
    ra.parent.mkdir(exist_ok=True)
    kq = json.loads(ra.read_text(encoding="utf-8")) if ra.exists() else {}
    kq["moi_truong"] = {"gpu": torch.cuda.get_device_name(0),
                        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                        "torch": torch.__version__, "python": platform.python_version(),
                        "ngay": time.strftime("%Y-%m-%d %H:%M")}
    for p in a.phan:
        print(f"=== {p} ===", flush=True)
        try:
            {"tinh": phan_tinh, "sam_mask": phan_sam_mask, "train": phan_train, "chay": phan_chay}[p](a, kq)
        except torch.cuda.OutOfMemoryError as e:
            kq[p + "_loi"] = f"HẾT VRAM: {str(e)[:300]}"
            print(kq[p + "_loi"], flush=True)
        ra.write_text(json.dumps(kq, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Ghi", ra)


if __name__ == "__main__":
    main()
