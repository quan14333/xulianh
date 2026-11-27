"""Bước 1: bảng thông tin từng ảnh SemanticRT (11.371 cặp) để chia lại theo góc máy.
Cột: ten, split_goc, may (góc máy, từ semanticrt_may.json), rong, cao, sang_vis (TB mức xám VIS),
     sang_ir, ir_3kenh_giong (IR lưu 3 kênh nhưng giống nhau?), px_<lop> (số điểm ảnh mỗi lớp 0–12).
Chạy: python -I b1_thong_tin.py <thư mục SemanticRT_dataset> <semanticrt_may.json>  -> thong_tin.csv
"""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

GOC = Path(sys.argv[1])
NLOP = 13


def mot(x):
    v = Image.open(GOC / "rgb" / f"{x}.jpg")
    i = Image.open(GOC / "thermal" / f"{x}.jpg")
    lab = np.asarray(Image.open(GOC / "labels" / f"{x}.png"))
    ia = np.asarray(i)
    giong = True if ia.ndim == 2 else bool((ia[..., 0] == ia[..., 1]).all() and (ia[..., 0] == ia[..., 2]).all())
    px = np.bincount(lab.ravel(), minlength=256)
    return dict(ten=x, rong=v.size[0], cao=v.size[1], ir_rong=i.size[0], ir_cao=i.size[1], lab_rong=lab.shape[1],
                lab_cao=lab.shape[0], sang_vis=float(np.asarray(v.convert("L")).mean()),
                sang_ir=float(np.asarray(i.convert("L")).mean()), ir_3kenh_giong=giong,
                nhan_ngoai=int(px[NLOP:].sum()), **{f"px_{c}": int(px[c]) for c in range(NLOP)})


if __name__ == "__main__":
    import pandas as pd
    m = json.load(open(sys.argv[2], encoding="utf-8"))
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(mot, m["ten"], chunksize=64))
    df = pd.DataFrame(rows)
    df.insert(1, "split_goc", m["split_goc"])
    df.insert(2, "may", m["may"])
    df.to_csv(Path(__file__).with_name("thong_tin.csv"), index=False)
    print(df.shape)
    print(df[["rong", "cao"]].value_counts())
    print("IR 3 kênh giống nhau:", df.ir_3kenh_giong.mean(), "| nhãn ngoài 0–12:", (df.nhan_ngoai > 0).sum())
    print("Kích thước VIS/IR/nhãn khớp:", ((df.rong == df.ir_rong) & (df.rong == df.lab_rong) & (df.cao == df.ir_cao) & (df.cao == df.lab_cao)).mean())
