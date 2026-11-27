"""Bước 3: chấm trên FMB test. Chỉ chạy SAU khi đã train xong.

Run train kiểu --gop_dev: chấm last.pt. Run chọn bằng dev: chấm cả best.pt và last.pt.

Ra (trong thư mục run), với {ck} là best hoặc last:
  test_nham_{ck}.npz    : ma trận nhầm lẫn 14x14 từng ảnh (để bootstrap, chia theo điều kiện)
  test_ket_qua_{ck}.json: mIoU theo 3 quy ước, trên 280 và 273 ảnh (bỏ 7 ảnh rò)
  test_iou_lop_{ck}.csv : IoU đủ 14 lớp
  pred_{ck}/{id}.png    : nhãn dự đoán (1..14 như FMB)

Ví dụ: python eval_seg.py --run runs/sage_b3_s0_1220
      SAGE_MAN=.../scene_v1/kham_pha python eval_seg.py --run runs/tb_b0_s0 --ckpt last --tap dev   (val 151, ra dev_nham_last.npz)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from chung import NC, TEN_LOP15, cac_miou, nham
from train_seg import FMBSeg, du_doan, so_kenh, tao_model


def cham(run, cfg, ck, luu_anh, tap="test"):
    lop0 = bool(cfg.get("lop0"))
    nc = NC + 1 if lop0 else NC
    model = tao_model(cfg["model"], nc, so_kenh(cfg["bien_the"]), cfg["bien_the"])
    model.load_state_dict(torch.load(run / f"{ck}.pt", map_location="cpu", weights_only=True))
    model.cuda().eval()

    ds = FMBSeg([tap], cfg["bien_the"], False, lop0=lop0)
    assert len(ds.df), f"tập {tap} rỗng"
    df = ds.df
    conf = np.zeros((len(df), nc, nc), np.int64)
    if luu_anh:
        (run / f"pred_{ck}").mkdir(exist_ok=True)   # chỉ ghi ảnh dự đoán khi tap = test
    for i in range(len(df)):
        x, y = ds[i]
        p = du_doan(model, x[None].cuda()).argmax(1)[0].cpu().numpy()
        conf[i] = nham(y.numpy(), p, nc)
        if luu_anh:
            cv2.imwrite(str(run / f"pred_{ck}" / f"{df.image_id.iloc[i]}.png"), (p if lop0 else p + 1).astype(np.uint8))

    leak = df.leak.astype(str).str.lower().eq("true").to_numpy()
    np.savez_compressed(run / f"{tap}_nham_{ck}.npz", conf=conf,
                        image_id=df.image_id.to_numpy().astype(str), leak=leak)
    kq = {k: cfg.get(k) for k in ("bien_the", "model", "seed", "gop_dev", "lop0", "epochs", "n_train", "best_dev_miou")}
    kq["ckpt"] = ck
    bang = {}
    for ten, chon in [("280", np.ones(len(df), bool)), ("273", ~leak)]:
        kq[ten], iou, iou15 = cac_miou(conf[chon].sum(0))
        kq[ten]["so_anh"] = int(chon.sum())
        bang[f"IoU_{ten}"] = [np.nan] + list(iou)   # bỏ pixel unlabeled
        if iou15 is not None:
            bang[f"IoU15_{ten}"] = iou15             # tính unlabeled là một lớp
    (run / f"{tap}_ket_qua_{ck}.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(bang, index=TEN_LOP15).round(2).to_csv(run / f"{tap}_iou_lop_{ck}.csv")
    print(f"== {run.name} [{ck}] tập {tap} ==")
    for ten in ("280", "273"):
        k = kq[ten]
        print(f"  {ten} ảnh: mIoU có mặt ({k['so_lop_co_mat']} lớp) {k['mIoU_co_mat']:.2f} | "
              f"14 lớp {k['mIoU_14']:.2f} | 7 lớp của bài {k['mIoU_7']:.2f} | pixel acc {k['pixel_acc']:.2f}"
              + (f" || tính unlabeled: 15 lớp {k['mIoU_15_nan0']:.2f} | 7 lớp {k['mIoU_7_ca_nhan0']:.2f}"
                 if "mIoU_15_nan0" in k else ""))
    print(pd.DataFrame(bang, index=TEN_LOP15).round(1).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", nargs="*", help="best và/hoặc last; mặc định tự chọn theo giao thức")
    ap.add_argument("--khong_luu_anh", action="store_true")
    ap.add_argument("--tap", default="test", help="tập chấm trong SAGE_MAN (test | dev); file ra có tiền tố tên tập")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for ck in a.ckpt or (["last"] if cfg.get("gop_dev") else ["best", "last"]):
        cham(run, cfg, ck, not a.khong_luu_anh and a.tap == "test", a.tap)


if __name__ == "__main__":
    main()
