"""Bước 2: train SegFormer trên ảnh (đã ghép) của FMB. Mỗi biến thể ảnh train một model riêng, như bài SAGE.

Hai giao thức:
  --gop_dev  : train trên đủ 1.220 ảnh train chính thức (train + dev của nhóm), 100 epoch, lấy last.pt.
               Đây là giao thức SÁT BÀI nhất (bài: "split chính thức, 100 epoch"). Không chọn checkpoint
               nên không đụng tới test.
  (mặc định) : train 1.045 ảnh, chọn best.pt bằng 175 ảnh dev. Giao thức sạch của nhóm.

Những chỗ bài KHÔNG ghi, nhóm tự chọn theo cấu hình SegFormer gốc: khởi tạo MiT ImageNet-1k,
AdamW lr 6e-5 (head x10), warmup + poly, crop 512, scale 0.5-2.0, lật ngang, CE bỏ nhãn 0.

Ví dụ:
  python train_seg.py --bien_the sage --gop_dev            # B3, 100 epoch
  python train_seg.py --bien_the sage --gop_dev --thu 40   # chạy thử 40 iter: đo tốc độ, bộ nhớ
"""
import argparse
import functools
import json
import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chung import ANH, IGNORE, MAN, MEAN, NC, RUNS, STD, cac_miou, doc_nhan, duong, nham, thu_muc

MODEL = {"b0": "nvidia/mit-b0", "b1": "nvidia/mit-b1", "b2": "nvidia/mit-b2", "b3": "nvidia/mit-b3"}

# Đo TRẦN 4 kênh (20/11): biến thể đặc biệt, không ghép ảnh – SegFormer nhận thẳng RGB của VIS + IR từ FMB gốc.
# Kênh IR chuẩn hóa bằng TB của MEAN/STD ImageNet; tầng đầu thêm kênh thứ 4 = TB trọng số 3 kênh RGB pretrained.
BT_4KENH = "vis_ir4"
# Cận trên v2 (21/11), cùng dữ liệu 4 kênh như vis_ir4:
#   vis_ir4_lr10 : như vis_ir4, conv đầu vào lr x10 (kênh IR mới phải học từ đầu, lr encoder 6e-5 quá chậm).
#   hai_nhanh    : 2 encoder MiT pretrained riêng (VIS RGB | IR nhân 3 kênh), mỗi tầng: f = v + i + conv1x1([v, i])
#                  (conv khởi tạo 0 -> lúc đầu là cộng), chung decoder SegFormer. Kiểu CMX đơn giản, gấp đôi encoder.
BT_LR10 = "vis_ir4_lr10"
BT_HAI = "hai_nhanh"
# Đối chứng (21/11): y hệt hai_nhanh nhưng kênh 4 = ảnh xám của VIS thay cho IR -> tách "thông tin IR" khỏi "2 encoder".
BT_HAI_VV = "hai_nhanh_visvis"
BT_VIS_IR = (BT_4KENH, BT_LR10, BT_HAI, BT_HAI_VV)   # đọc thẳng VIS + IR gốc (visvis: không dùng IR)


def so_kenh(bien_the):
    return 4 if bien_the in BT_VIS_IR else 3


class HaiNhanh(torch.nn.Module):
    """SegFormer 2 nhánh: nhận x 4 kênh (RGB đã chuẩn hóa ImageNet + IR chuẩn hóa TB MEAN/STD như vis_ir4)."""

    def __init__(self, m):
        super().__init__()
        import copy
        self.segformer = m.segformer                    # nhánh VIS (tên giữ nguyên để nhóm lr encoder)
        self.segformer_ir = copy.deepcopy(m.segformer)  # nhánh IR, cùng trọng số ImageNet ban đầu
        self.decode_head = m.decode_head
        self.config = m.config
        self.tron = torch.nn.ModuleList(
            torch.nn.Conv2d(2 * c, c, 1) for c in m.config.hidden_sizes)
        for t in self.tron:
            torch.nn.init.zeros_(t.weight)
            torch.nn.init.zeros_(t.bias)
        # IR (chuẩn hóa TB) -> 3 kênh chuẩn hóa đúng từng kênh ImageNet: ir3 = (ir*s + m - MEAN_c) / STD_c
        self.register_buffer("a", torch.tensor(STD.mean() / STD, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("b", torch.tensor((MEAN.mean() - MEAN) / STD, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, pixel_values):
        v = pixel_values[:, :3]
        ir = pixel_values[:, 3:4] * self.a + self.b
        hv = self.segformer(v, output_hidden_states=True).hidden_states
        hi = self.segformer_ir(ir, output_hidden_states=True).hidden_states
        f = [a + b + t(torch.cat([a, b], 1)) for a, b, t in zip(hv, hi, self.tron)]
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.decode_head(f))


def nhom_tham_so(model, bien_the, lr):
    """encoder: lr; decoder + khối trộn: lr x10; vis_ir4_lr10: conv đầu vào lr x10."""
    dau = set()
    if bien_the == BT_LR10:
        dau = {n + ".weight" for n, mod in model.named_modules()
               if isinstance(mod, torch.nn.Conv2d) and mod.in_channels == 4}
        dau |= {n.replace(".weight", ".bias") for n in dau}
        assert len(dau) == 2, dau
    enc, head, d = [], [], []
    for n, p in model.named_parameters():
        if n in dau:
            d.append(p)
        elif n.startswith("segformer.") or n.startswith("segformer_ir."):
            enc.append(p)
        else:
            head.append(p)
    g = [{"params": enc, "lr": lr}, {"params": head, "lr": lr * 10}]
    if d:
        g.append({"params": d, "lr": lr * 10})
    return g


def ten_run(bien_the, model, seed, gop_dev, lop0=False):
    return f"{bien_the}_{model}_s{seed}" + ("_1220" if gop_dev else "") + ("_15lop" if lop0 else "")


class FMBSeg(Dataset):
    def __init__(self, splits, bien_the, train, crop=512, scale=(0.5, 2.0), lop0=False):
        self.lop0 = lop0
        self.df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in splits],
                            ignore_index=True)
        self.bt, self.train, self.crop, self.scale = bien_the, train, crop, scale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.bt in BT_VIS_IR:
            v = cv2.imread(str(duong(r.vis_path)))
            ir = cv2.imread(str(duong(r.ir_path)), cv2.IMREAD_GRAYSCALE)
            if self.bt == BT_HAI_VV:
                ir = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            assert v is not None and ir is not None and v.shape[:2] == ir.shape, f"lỗi đọc VIS/IR {r.image_id}"
            img = np.dstack([cv2.cvtColor(v, cv2.COLOR_BGR2RGB), ir])
        else:
            img = cv2.imread(str(ANH / self.bt / thu_muc(r) / f"{r.image_id}.png"))
            assert img is not None, f"thiếu ảnh {self.bt}/{r.image_id}, chạy xuat_anh.py trước"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = doc_nhan(duong(r.label_path), self.lop0)
        if self.train:
            s = random.uniform(*self.scale)
            h, w = int(img.shape[0] * s + 0.5), int(img.shape[1] * s + 0.5)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            ph, pw = max(self.crop - h, 0), max(self.crop - w, 0)
            if ph or pw:
                img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
                lab = cv2.copyMakeBorder(lab.astype(np.uint8), 0, ph, 0, pw, cv2.BORDER_CONSTANT,
                                         value=IGNORE).astype(np.int64)
            y0 = random.randint(0, img.shape[0] - self.crop)
            x0 = random.randint(0, img.shape[1] - self.crop)
            img = img[y0:y0 + self.crop, x0:x0 + self.crop]
            lab = lab[y0:y0 + self.crop, x0:x0 + self.crop]
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
        if img.shape[2] == 4:
            x = (img.astype(np.float32) / 255 - np.append(MEAN, MEAN.mean())) / np.append(STD, STD.mean())
        else:
            x = (img.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(lab.copy())


def _seed_worker(w, seed):
    cv2.setNumThreads(0)
    random.seed(seed * 1000 + w)
    np.random.seed(seed * 1000 + w)


def tao_model(ten, nc=NC, kenh=3, bien_the=None):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(
        MODEL[ten], num_labels=nc, id2label={i: str(i) for i in range(nc)},
        label2id={str(i): i for i in range(nc)}, ignore_mismatched_sizes=True)
    if bien_the in (BT_HAI, BT_HAI_VV):
        return HaiNhanh(m)
    if kenh == 4:   # thêm kênh IR vào tầng đầu: giữ trọng số RGB pretrained, kênh 4 = TB 3 kênh RGB
        # conv nhận ảnh đầu vào (tên khác nhau theo bản transformers: encoder.patch_embeddings.0.proj | stages.0.patch_embeddings.proj)
        ten_cu = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d) and mod.in_channels == 3]
        assert len(ten_cu) == 1, f"không xác định được conv đầu vào: {ten_cu}"
        cha, _, la = ten_cu[0].rpartition(".")
        cu = m.get_submodule(ten_cu[0])
        moi = torch.nn.Conv2d(4, cu.out_channels, cu.kernel_size, cu.stride, cu.padding, bias=cu.bias is not None)
        with torch.no_grad():
            moi.weight[:, :3] = cu.weight
            moi.weight[:, 3:] = cu.weight.mean(1, keepdim=True)
            if cu.bias is not None:
                moi.bias.copy_(cu.bias)
        setattr(m.get_submodule(cha), la, moi)
        m.config.num_channels = 4
    return m


@torch.inference_mode()
def du_doan(model, x):
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(pixel_values=x).logits
    return F.interpolate(lg.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)


def danh_gia(model, loader, nc=NC):
    model.eval()
    c = np.zeros((nc, nc), np.int64)
    for x, y in loader:
        p = du_doan(model, x.cuda()).argmax(1).cpu().numpy()
        for pi, yi in zip(p, y.numpy()):
            c += nham(yi, pi, nc)
    model.train()
    return cac_miou(c)[0]["mIoU_co_mat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", required=True)
    ap.add_argument("--model", default="b3", choices=list(MODEL))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tich_luy", type=int, default=1,
                    help="cộng dồn gradient N bước (batch 4 x tich_luy 2 ~ batch 8 khi thiếu bộ nhớ)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--gop_dev", action="store_true", help="train trên train + dev = 1.220 ảnh chính thức")
    ap.add_argument("--lop0", action="store_true",
                    help="coi unlabeled (nhãn 0) là một lớp: model 15 lớp, như code SegMiF cho MFNet")
    ap.add_argument("--eval_moi", type=int, default=500, help="ghi log (và chấm dev nếu không gop_dev) mỗi N iter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tiep", action="store_true", help="train tiếp từ resume.pt nếu có")
    ap.add_argument("--thu", type=int, default=0, help="chỉ chạy N iter để đo tốc độ, bộ nhớ; không lưu gì")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    run = RUNS / ten_run(a.bien_the, a.model, a.seed, a.gop_dev, a.lop0)
    nc = NC + 1 if a.lop0 else NC
    if not a.thu:
        run.mkdir(parents=True, exist_ok=True)

    tr = FMBSeg(["train", "dev"] if a.gop_dev else ["train"], a.bien_the, True, a.crop, lop0=a.lop0)
    g = torch.Generator().manual_seed(a.seed)
    lt = DataLoader(tr, a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0, generator=g,
                    worker_init_fn=functools.partial(_seed_worker, seed=a.seed))
    ld = None
    if not a.gop_dev:
        ld = DataLoader(FMBSeg(["dev"], a.bien_the, False, lop0=a.lop0), 2, num_workers=a.workers)

    model = tao_model(a.model, nc, so_kenh(a.bien_the), a.bien_the).cuda().train()
    opt = torch.optim.AdamW(nhom_tham_so(model, a.bien_the, a.lr), lr=a.lr, weight_decay=0.01)
    tong = a.epochs * len(lt)                 # số iter (số batch)
    buoc = tong // a.tich_luy                 # số lần cập nhật trọng số
    warm = min(1500, buoc // 10)
    # warmup tuyến tính rồi giảm poly (power 1), giống mmseg/SegFormer
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda b: (b + 1) / warm if b < warm else max(0.0, 1 - (b - warm) / max(1, buoc - warm)))
    scaler = torch.amp.GradScaler("cuda")

    cfg = vars(a) | {"iters": tong, "buoc_cap_nhat": buoc, "warmup": warm, "n_train": len(tr),
                     "n_dev": 0 if a.gop_dev else len(ld.dataset), "init": MODEL[a.model],
                     "so_lop": nc, "loss": "CE, nhãn 0 là một lớp" if a.lop0 else "CE, bỏ nhãn 0", "optimizer": "AdamW wd 0.01, head lr x10",
                     "aug": "scale 0.5-2.0, random crop, hflip",
                     "chon_ckpt": "last (không chọn)" if a.gop_dev else "best theo dev",
                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    for k in ("tiep", "thu", "workers"):
        cfg.pop(k, None)

    best, it, t0, dl, ep0, da_chay = -1.0, 0, time.time(), [], 0, 0.0
    rp = run / "resume.pt"
    log = None
    if a.thu:
        print(f"CHẠY THỬ {a.thu} iter: {a.model}, batch {a.batch} x tích lũy {a.tich_luy}, "
              f"{len(tr)} ảnh train, {tong} iter nếu chạy thật", flush=True)
    elif a.tiep and rp.exists():
        # train tiếp từ cuối epoch đã lưu (không khớp từng bit: worker augmentation khởi động lại RNG)
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        bo = lambda c: {k: v for k, v in c.items() if k not in ("gpu", "torch")}
        if bo(ck["cfg"]) != bo(cfg):
            raise RuntimeError(f"cấu hình khác lần trước, không train tiếp được:\n{ck['cfg']}\n{cfg}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        scaler.load_state_dict(ck["scaler"])
        best, it, ep0, dl, da_chay = ck["best"], ck["it"], ck["ep"], ck["dl"], ck["giay"]
        g.set_state(ck["rng_gen"])
        torch.set_rng_state(ck["rng_torch"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        random.setstate(ck["rng_py"])
        np.random.set_state(ck["rng_np"])
        log = open(run / "log.csv", "a", encoding="utf-8")
        print(f"{run.name}: TRAIN TIẾP từ epoch {ep0}, iter {it}/{tong}", flush=True)
    else:
        (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        log = open(run / "log.csv", "w", encoding="utf-8")
        log.write("iter,epoch,loss,lr,dev_miou,giay\n")
        print(f"{run.name}: {len(tr)} train, {cfg['n_dev']} dev, {tong} iter", flush=True)
    t0 -= da_chay  # cột "giay" tính cả thời gian các lần chạy trước

    opt.zero_grad(set_to_none=True)
    for ep in range(ep0, a.epochs):
        for x, y in lt:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(pixel_values=x).logits
            lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(lg, y, ignore_index=IGNORE)
            if not math.isfinite(loss.item()):
                raise RuntimeError("loss NaN/inf")
            scaler.scale(loss / a.tich_luy).backward()
            it += 1
            if it % a.tich_luy == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
            dl.append(loss.item())

            if a.thu:
                if it == 5:
                    t5 = time.time()
                if it >= a.thu:
                    toc = (it - 5) / (time.time() - t5)
                    mem = torch.cuda.max_memory_allocated() / 2 ** 30
                    print(f"Tốc độ {toc:.2f} iter/s | bộ nhớ GPU cao nhất {mem:.1f} GB | "
                          f"loss {np.mean(dl):.3f}\nƯớc tính chạy thật: {tong / toc / 3600:.1f} giờ "
                          f"cho {a.epochs} epoch (chưa tính chấm điểm)", flush=True)
                    return
                continue

            if it % a.eval_moi == 0 or it == tong:
                m = danh_gia(model, ld, nc) if ld is not None else float("nan")
                l = float(np.mean(dl))
                dl = []
                log.write(f"{it},{ep + 1},{l:.4f},{sch.get_last_lr()[0]:.2e},{m:.2f},{time.time() - t0:.0f}\n")
                log.flush()
                tot = ld is not None and m > best
                if tot:
                    best = m
                    torch.save(model.state_dict(), run / "best.pt")
                con = (time.time() - t0) / it * (tong - it) / 3600
                print(f"iter {it}/{tong} ep {ep + 1} loss {l:.3f}"
                      + (f" dev mIoU {m:.2f}{' *' if tot else ''}" if ld is not None else "")
                      + f"  {time.time() - t0:.0f}s, còn ~{con:.1f} giờ", flush=True)
        # lưu trạng thái cuối mỗi epoch để train tiếp nếu bị ngắt
        tmp = run / "resume.tmp"
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "it": it, "ep": ep + 1, "dl": dl,
                    "giay": time.time() - t0, "rng_gen": g.get_state(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(), "rng_py": random.getstate(),
                    "rng_np": np.random.get_state()}, tmp)
        os.replace(tmp, rp)
    torch.save(model.state_dict(), run / "last.pt")
    cfg["best_dev_miou"] = None if a.gop_dev else best
    cfg["gio_train"] = round((time.time() - t0) / 3600, 2)
    (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    rp.unlink(missing_ok=True)   # train xong thì bỏ resume.pt cho nhẹ (last.pt đã có trọng số)
    print(f"Xong {run.name} sau {cfg['gio_train']} giờ. Checkpoint: {run / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
