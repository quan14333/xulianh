"""So 4 biến thể B3 (vis/ir/sage/tb, từng seed + TB các seed, last.pt), 273 ảnh sạch, mIoU_co_mat.
CI95 bootstrap theo khối id (thong_ke.py, thong_tin_anh_test.csv). Chia tối/vừa/sáng theo tam phân vị độ sáng VIS.
Chạy: SAGE_RUNS=<thư mục chứa các run> python so_sanh_b3.py > so_sanh_b3.txt"""
import numpy as np, pandas as pd
from chung import RUNS, TEN_LOP, cac_miou
from thong_ke import boot_idx, ci, SEED, doc_thong_tin_anh
anh = doc_thong_tin_anh()
BT = ["vis", "ir", "sage", "tb"]
SEEDS = sorted({int(p.name.split("_s")[1].split("_")[0]) for p in RUNS.glob("*_b3_s*_1220_15lop")
                 if all((RUNS / f"{b}_b3_s{p.name.split('_s')[1].split('_')[0]}_1220_15lop" / "test_nham_last.npz").exists() for b in BT)})
CS = {}
for s in SEEDS:
    for b in BT:
        z = np.load(RUNS / f"{b}_b3_s{s}_1220_15lop" / "test_nham_last.npz")
        assert [str(i) for i in z["image_id"]] == list(anh.image_id), (b, s)
        CS[(s, b)] = z["conf"]
print("Seed đủ 4 biến thể:", SEEDS)
def m(c): d, iou, _ = cac_miou(c); return d["mIoU_co_mat"], iou
sach = ~anh.leak.to_numpy()
q = np.quantile(anh.y_mean[sach], [1/3, 2/3])
nhom = np.where(anh.y_mean < q[0], "toi", np.where(anh.y_mean < q[1], "vua", "sang"))
CAP = [("sage", "vis"), ("tb", "vis"), ("tb", "sage"), ("sage", "ir")]
P = TEN_LOP.index("person")
TOM = []
for s_seed in SEEDS:
    print(f"\n################ SEED {s_seed} ################")
    C = {b: CS[(s_seed, b)] for b in BT}
    for g, chon in [("273 ảnh", sach), ("tối", sach & (nhom == "toi")),
                    ("vừa", sach & (nhom == "vua")), ("sáng", sach & (nhom == "sang"))]:
        khoi = anh.khoi[chon].to_numpy()
        mm = {b: m(C[b][chon].sum(0)) for b in BT}
        print(f"\n== [{g}] {chon.sum()} ảnh | " + "  ".join(f"{b} {mm[b][0]:.2f}" for b in BT)
              + " | person " + "  ".join(f"{b} {mm[b][1][P]:.1f}" for b in BT))
        rng = np.random.default_rng(SEED)
        ds = {c: [] for c in CAP}
        for i in boot_idx(khoi, rng):
            v = {b: m(C[b][chon][i].sum(0))[0] for b in BT}
            for a, b in CAP: ds[(a, b)].append(v[a] - v[b])
        for a, b in CAP:
            c = ci(ds[(a, b)]); s = "*" if c[0] > 0 or c[1] < 0 else " "
            print(f"  {a:4s}−{b:4s} {mm[a][0]-mm[b][0]:+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]{s}")
            TOM.append((s_seed, g, a, b, mm[a][0]-mm[b][0], c[0], c[1]))
    for b in BT:
        TOM.append((s_seed, "mIoU", b, "", m(C[b][sach].sum(0))[0], np.nan, np.nan))

t = pd.DataFrame(TOM, columns=["seed", "nhom", "a", "b", "gia_tri", "ci_lo", "ci_hi"])
print("\n################ TÓM TẮT THEO SEED (273 ảnh) ################")
mi = t[t.nhom == "mIoU"].pivot(index="a", columns="seed", values="gia_tri").loc[BT]
mi["TB"] = mi[SEEDS].mean(1); mi["SD"] = mi[SEEDS].std(1, ddof=1) if len(SEEDS) > 1 else np.nan
print(mi.round(2).to_string())
d = t[t.nhom != "mIoU"].copy()
d["o"] = d.apply(lambda r: f"{r.gia_tri:+.2f}{'*' if r.ci_lo > 0 or r.ci_hi < 0 else ''}", axis=1)
d["cap"] = d.a + "−" + d.b
print(d.pivot_table(index=["nhom", "cap"], columns="seed", values="o", aggfunc="first", sort=False).to_string())

# GỘP SEED: mỗi lần bootstrap dùng CÙNG bộ ảnh cho mọi seed, lấy TB chênh lệch qua seed
# (chỉ tính biến thiên theo ảnh; biến thiên seed xem ở bảng trên). Tiêu chí ghi trong reports/experiments/b3_fmb_seg/README.md mục 9.1.
if len(SEEDS) > 1:
    print(f"\n################ GỘP {len(SEEDS)} SEED (273 ảnh, TB chênh lệch qua seed) ################")
    for g, chon in [("273 ảnh", sach), ("tối", sach & (nhom == "toi")),
                    ("vừa", sach & (nhom == "vua")), ("sáng", sach & (nhom == "sang"))]:
        khoi = anh.khoi[chon].to_numpy()
        goc = {(s, b): m(CS[(s, b)][chon].sum(0))[0] for s in SEEDS for b in BT}
        rng = np.random.default_rng(SEED)
        ds = {c: [] for c in CAP}
        for i in boot_idx(khoi, rng):
            v = {(s, b): m(CS[(s, b)][chon][i].sum(0))[0] for s in SEEDS for b in BT}
            for a, b in CAP: ds[(a, b)].append(np.mean([v[(s, a)] - v[(s, b)] for s in SEEDS]))
        print(f"== [{g}] {chon.sum()} ảnh")
        for a, b in CAP:
            c = ci(ds[(a, b)]); s = "*" if c[0] > 0 or c[1] < 0 else " "
            print(f"  {a:4s}−{b:4s} {np.mean([goc[(x, a)] - goc[(x, b)] for x in SEEDS]):+.2f} [{c[0]:+.2f}; {c[1]:+.2f}]{s}")
