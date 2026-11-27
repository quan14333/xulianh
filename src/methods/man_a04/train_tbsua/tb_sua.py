"""Nền 2 – "TB + phần sửa": ảnh ghép Y = clamp(TB + s * Student(Y_vis, IR), 0, 1), TB = (Y_vis + IR) / 2.

Student = khung mạng của SAGE (sage_student/model.py, 0,136M, đầu ra Tanh trong [-1, 1]), train từ đầu, KHÔNG SAM.
Lớp cuối khởi tạo 0 -> lúc đầu ảnh ghép đúng bằng TB; Student chỉ học PHẦN SỬA.

Bước 2 (bản này, chưa có CE): loss = w_tb * MSE(F, TB) + w_g * Grad(F, Y_vis, IR)
  - Grad: như SAGE, L1(|∇F|, max(|∇Y_vis|, |∇IR|)) -> trả lại cạnh (Q_ABF).
  - MSE(F, TB): giữ gần TB (giữ ưu điểm mIoU của TB). w_tb = 15 = tổng trọng số MSE của SAGE (9·MSE_Y + 6·MSE_IR).
  - w_g = 27 = trọng số grad của SAGE (3 * 9).
Công thức train giống E17 nosam (Adam 2e-3, cosine, batch 4, crop 192x256 + lật, 100 epoch, chỉ học ở batch chẵn).

Checkpoint lưu thêm khóa "_tb_scale" (= s) để xuat_anh.py biết đây là kiểu TB + phần sửa.

Ví dụ (tập khám phá):
  SAGE_MAN=.../scene_v1/kham_pha SO_TRAIN=841 python tb_sua.py --chi_train --ten tbsua_kp841
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

DAY = Path(__file__).resolve().parent
sys.path.insert(0, str(DAY.parent / "b3"))
sys.path.insert(0, str(DAY.parent / "e17"))
from chung import MAN, STUDENT, duong  # noqa: E402
from train_student_nosam import SOBEL_X, SOBEL_Y, grad, grad_loss  # noqa: E402

sys.path.insert(0, str(STUDENT))
from model import Network  # noqa: E402


def tao_mang():
    net = Network()
    lop_cuoi = net.final_decoder[0]
    torch.nn.init.zeros_(lop_cuoi.weight)
    torch.nn.init.zeros_(lop_cuoi.bias)
    return net


def ghep(net, y, ir, s):
    """Ảnh ghép = TB + s * phần sửa. Trả (F, phần sửa)."""
    r = net(y, ir)[0]
    return ((y + ir) / 2 + s * r).clamp(0, 1), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--s", type=float, default=0.5, help="biên độ phần sửa (Y trong [0,1])")
    ap.add_argument("--w_tb", type=float, default=15.0)
    ap.add_argument("--w_g", type=float, default=27.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thu", type=int, default=0, help="chỉ chạy N bước để đo tốc độ, không lưu")
    ap.add_argument("--ten", default="tbsua")
    ap.add_argument("--chi_train", action="store_true", help="chỉ tập train của SAGE_MAN (bỏ dev)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", dev, flush=True)

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str})
                    for s in (("train",) if a.chi_train else ("train", "dev"))])
    assert len(df) == (int(os.environ.get("SO_TRAIN", 1045)) if a.chi_train else 1220)
    assert not df.vis_path.str.startswith("test").any()
    print(f"Nạp {len(df)} ảnh train vào RAM ...", flush=True)
    Y = np.stack([cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2YCrCb)[:, :, 0] for p in df.vis_path])
    IR = np.stack([cv2.cvtColor(cv2.imread(str(duong(p))), cv2.COLOR_BGR2GRAY) for p in df.ir_path])
    N, H, W = Y.shape
    ch, cw = 192, 256

    net = tao_mang().to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_batch = math.ceil(N / a.batch)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * so_batch, eta_min=a.lr_min)
    kx, ky = SOBEL_X.to(dev), SOBEL_Y.to(dev)

    log, buoc, t0 = [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        thu_tu = rng.permutation(N)
        tong = dict(loss=0., tb=0., grad=0., sua_tb=0., n=0)
        for bi in range(so_batch):
            if bi % 2 == 1:      # như code gốc SAGE: Student chỉ học ở batch chẵn
                continue
            idx = thu_tu[bi * a.batch:(bi + 1) * a.batch]
            ys, irs = [], []
            for k in idx:
                i0, j0 = rng.integers(0, H - ch + 1), rng.integers(0, W - cw + 1)
                y, ir = Y[k, i0:i0 + ch, j0:j0 + cw], IR[k, i0:i0 + ch, j0:j0 + cw]
                if rng.random() < 0.5:
                    y, ir = y[:, ::-1], ir[:, ::-1]
                ys.append(y.copy()), irs.append(ir.copy())
            y = torch.from_numpy(np.stack(ys)).float().div(255)[:, None].to(dev)
            ir = torch.from_numpy(np.stack(irs)).float().div(255)[:, None].to(dev)
            f, r = ghep(net, y, ir, a.s)
            l_tb = F.mse_loss(f, (y + ir) / 2)
            l_g = grad_loss(f, y, ir, kx, ky)
            loss = a.w_tb * l_tb + a.w_g * l_g
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở bước {buoc}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            buoc += 1
            tong["loss"] += loss.item(); tong["tb"] += l_tb.item(); tong["grad"] += l_g.item()
            tong["sua_tb"] += (a.s * r).abs().mean().item() * 255; tong["n"] += 1
            if a.thu and buoc >= a.thu:
                dt = time.time() - t0
                print(f"Thử {buoc} bước: {dt:.1f}s -> ước tính cả bài {dt / buoc * a.epochs * so_batch / 2 / 60:.0f} phút, "
                      f"loss {loss.item():.4f}, |phần sửa| {a.s * r.abs().mean().item() * 255:.2f} mức xám")
                return
        n = tong.pop("n")
        dong = {"epoch": ep, "buoc": buoc, "lr": sch.get_last_lr()[0], **{k: v / n for k, v in tong.items()},
                "phut": (time.time() - t0) / 60}
        log.append(dong)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} loss {dong['loss']:.4f} (MSE_TB {dong['tb']:.5f}, grad {dong['grad']:.4f}) "
                  f"|phần sửa| TB {dong['sua_tb']:.2f} mức xám  {dong['phut']:.1f} phút", flush=True)
    sd = net.state_dict()
    sd["_tb_scale"] = torch.tensor(a.s)
    torch.save(sd, STUDENT / f"{a.ten}.pt")
    pd.DataFrame(log).to_csv(STUDENT / f"{a.ten}_log.csv", index=False)
    print(f"Lưu {STUDENT / (a.ten + '.pt')}")


if __name__ == "__main__":
    main()
