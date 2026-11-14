"""Bước 1: xuất ảnh đầu vào cho SegFormer theo từng biến thể, cho cả train, dev, test.

  vis  : ảnh VIS gốc
  ir   : IR xám, nhân 3 kênh
  tb   : Y = (Y_vis + IR)/2, màu CrCb của VIS  (ghép trung bình – đối chứng đơn giản, bài không có)
  clahe: Y = CLAHE(Y_vis), màu CrCb của VIS     (đối chứng "chỉ tăng tương phản", bài không có)
  sage : Y = SAGE Student pretrained, màu CrCb của VIS (giống output thật của SAGE)

Ví dụ: python xuat_anh.py --bien_the sage
"""
import argparse
import sys

import cv2
import numpy as np
import pandas as pd

from chung import ANH, MAN, STUDENT, duong, thu_muc


CKPT = {"sage": "checkpoints.pt"}  # biến thể chạy bằng mạng Student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", required=True, choices=["vis", "ir", "tb", "sage", "clahe"])
    ap.add_argument("--gioi_han", type=int, default=0, help="chỉ để thử: xuất N ảnh đầu")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(MAN / f"{s}.csv", dtype={"image_id": str}) for s in ("train", "dev", "test")])
    if a.gioi_han:
        df = df.head(a.gioi_han)
    if a.bien_the in CKPT:
        import torch
        sys.path.insert(0, str(STUDENT))
        from model import Network
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = Network()
        net.load_state_dict(torch.load(STUDENT / CKPT[a.bien_the], map_location="cpu", weights_only=True),
                            strict=True)
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
        if a.bien_the == "vis":
            img = vis
        elif a.bien_the == "ir":
            img = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        else:
            ycc = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            y = ycc[:, :, 0]
            if a.bien_the == "clahe":
                yf = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
            elif a.bien_the == "tb":
                yf = ((y.astype(np.float64) + ir) / 2).round().astype(np.uint8)
            else:
                with torch.inference_mode():
                    yf = (net(t2(y), t2(ir))[0].clamp(0, 1) * 255).round().byte().cpu().numpy()[0, 0]
            img = cv2.cvtColor(np.dstack([yf, ycc[:, :, 1:]]), cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(str(out), img)
        n += 1
    print(f"{a.bien_the}: ghi {n} ảnh mới vào {ANH / a.bien_the}", flush=True)


if __name__ == "__main__":
    main()
