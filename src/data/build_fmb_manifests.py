"""Tạo train.csv, dev.csv, test.csv cho FMB từ kết quả audit, rồi kiểm tra chúng.

Chạy nhanh, không cần ảnh (chưa kiểm tra file, cột md5 của IR/label để trống):
    python src/data/build_fmb_manifests.py --audit-dir fmb_audit_final --out data/manifests

Chạy đầy đủ, có ảnh (bản chính thức để commit):
    python src/data/build_fmb_manifests.py --audit-dir <audit> --out <out> --root <FMB>

Cách chia dev: xem data/manifests/README.md. Kết quả không phụ thuộc seed: chạy lại
luôn ra cùng danh sách.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

N_TRAIN_GOC, N_TEST = 1220, 280
N_LEAK = 7
DEV_MIN, DEV_MAX = 122, 183  # 10–15% của 1.220 ảnh train
SO_KHOI_DEV = (3, 4, 5)  # ít nhất 3 khối để dev không chỉ có một cảnh
KHOI_NHO_NHAT = 25
MOC_DO_SANG = (60, 90, 130)  # chia vis_mean thành 4 nhóm: tối → sáng
NGUONG_DHASH = 5  # cùng ngưỡng tìm ứng viên trong audit
SIZE = (800, 600)
THU_MUC = {"vis": "Visible", "ir": "Infrared", "label": "Label"}  # không dùng "color"
COT = [
    "image_id", "split", "vis_path", "ir_path", "label_path", "width", "height",
    "vis_md5", "ir_md5", "label_md5", "leak",
]


def doc_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tach_khoi(ids: list[int]) -> list[tuple[int, int]]:
    """Gom các id liền số thành khối (đầu, cuối)."""
    khoi, dau, truoc = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != truoc + 1:
            khoi.append((dau, truoc))
            dau = i
        truoc = i
    khoi.append((dau, truoc))
    return khoi


def histogram(values: list[float]) -> list[float]:
    dem = [0] * (len(MOC_DO_SANG) + 1)
    for v in values:
        dem[sum(v >= moc for moc in MOC_DO_SANG)] += 1
    return [d / len(values) for d in dem]


def chon_khoi_dev(
    do_sang: dict[int, float], khoi_cam: set[int]
) -> tuple[tuple[tuple[int, int], ...], float]:
    """Chọn tổ hợp khối có phân bố độ sáng gần phần train còn lại nhất."""
    ung_vien = [
        k for k in tach_khoi(sorted(do_sang))
        if k[1] - k[0] + 1 >= KHOI_NHO_NHAT
        and not any(k[0] <= i <= k[1] for i in khoi_cam)
    ]
    tot_nhat, diem_tot = None, float("inf")
    for so_khoi in SO_KHOI_DEV:
        for to_hop in itertools.combinations(ung_vien, so_khoi):
            dev = {i for a, b in to_hop for i in range(a, b + 1)}
            if not DEV_MIN <= len(dev) <= DEV_MAX:
                continue
            h_dev = histogram([do_sang[i] for i in dev])
            h_con = histogram([v for i, v in do_sang.items() if i not in dev])
            diem = sum(abs(x - y) for x, y in zip(h_dev, h_con))
            if diem < diem_tot - 1e-12:
                tot_nhat, diem_tot = to_hop, diem
    if tot_nhat is None:
        raise SystemExit("Không có tổ hợp khối nào thoả điều kiện dev.")
    return tot_nhat, diem_tot


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def tao_dong(audit: list[dict[str, str]], dev_ids: set[int], leak: set[str]) -> dict[str, list]:
    tap: dict[str, list] = {"train": [], "dev": [], "test": []}
    for dong in sorted(audit, key=lambda d: int(d["name"])):
        name = dong["name"]
        thu_muc_goc = dong["split"].split("/")[0]  # audit ghi "train/train" hoặc "test/test"
        split = "test" if thu_muc_goc == "test" else "dev" if int(name) in dev_ids else "train"
        tap[split].append({
            "image_id": name,
            "split": split,
            "vis_path": f"{thu_muc_goc}/Visible/{name}.png",
            "ir_path": f"{thu_muc_goc}/Infrared/{name}.png",
            "label_path": f"{thu_muc_goc}/Label/{name}.png",
            "width": int(dong["vis_w"]),
            "height": int(dong["vis_h"]),
            "vis_md5": dong["vis_md5"],
            "ir_md5": "",
            "label_md5": "",
            "leak": name in leak,
        })
    return tap


def kiem_tra_danh_sach(tap: dict[str, list], leak: set[str], dev_ids: set[int]) -> list[str]:
    loi = []
    ids = {k: [d["image_id"] for d in v] for k, v in tap.items()}
    tat_ca = sum(ids.values(), [])

    def ky(ten: str, dat: bool, chi_tiet: str = "") -> None:
        print(f"  [{'OK' if dat else 'LỖI'}] {ten} {chi_tiet}".rstrip())
        if not dat:
            loi.append(ten)

    for ten, ds in ids.items():
        ky(f"không trùng id trong {ten}.csv", len(ds) == len(set(ds)), f"({len(ds)} dòng)")
    for a, b in itertools.combinations(ids, 2):
        ky(f"{a} ∩ {b} rỗng", not set(ids[a]) & set(ids[b]))
    ky("tổng 1.500 ảnh", len(tat_ca) == N_TRAIN_GOC + N_TEST, f"({len(tat_ca)})")
    ky("test đủ 280 ảnh", len(ids["test"]) == N_TEST)
    ky("dev 10–15% của train", DEV_MIN <= len(ids["dev"]) <= DEV_MAX,
       f"({len(ids['dev'])} ảnh, {len(ids['dev']) / N_TRAIN_GOC:.1%})")
    ky("dev chỉ lấy từ train", all(d["vis_path"].startswith("train/") for d in tap["dev"]))
    co_leak = {d["image_id"] for d in tap["test"] if d["leak"]}
    ky("đúng 7 ảnh test gắn cờ leak", co_leak == leak and len(leak) == N_LEAK, f"({len(co_leak)})")
    ky("train/dev không gắn cờ leak", not any(d["leak"] for d in tap["train"] + tap["dev"]))
    con_train = sorted(int(i) for i in ids["train"])
    gap = min(abs(a - b) for a in dev_ids for b in con_train)
    ky("dev không liền số với train", gap >= 2, f"(khoảng cách id nhỏ nhất: {gap})")
    return loi


def kiem_tra_file(tap: dict[str, list], root: Path, out: Path) -> list[str]:
    """Dò đuôi file thật, kiểm tra ảnh, điền md5, tìm ảnh gần trùng giữa dev và train."""
    import numpy as np
    from PIL import Image

    loi: list[str] = []
    chi_so: dict[str, dict[str, str]] = {}

    def tim(goc: str, loai: str, name: str) -> tuple[Path, str] | None:
        """Trả về (đường dẫn thật, đường dẫn ghi vào manifest); nhận cả train/train/... ."""
        ghi = f"{goc}/{THU_MUC[loai]}"
        if ghi not in chi_so:
            cac_cho = [root / ghi, root / goc / ghi]
            p = next((c for c in cac_cho if c.is_dir()), None)
            chi_so[ghi] = {f.stem: f for f in p.iterdir()} if p else {}
        f = chi_so[ghi].get(name)
        return (f, f"{ghi}/{f.name}") if f else None

    def dhash(path: Path) -> np.ndarray:
        g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BOX))
        return (g[:, 1:] > g[:, :-1]).flatten()

    so_lop: dict[str, dict[int, int]] = {k: {} for k in tap}
    hash_vis: dict[str, dict[str, np.ndarray]] = {"train": {}, "dev": {}}
    duong_dan_vis: dict[str, Path] = {}

    for split, dong_list in tap.items():
        print(f"  Đọc {split} ({len(dong_list)} cặp)...", flush=True)
        for d in dong_list:
            goc = "test" if split == "test" else "train"
            duong_dan = {}
            for loai in THU_MUC:
                tim_thay = tim(goc, loai, d["image_id"])
                if tim_thay is None:
                    loi.append(f"thiếu {loai} của {split}/{d['image_id']}")
                    continue
                duong_dan[loai], d[f"{loai}_path"] = tim_thay
            if len(duong_dan) < 3:
                continue
            if md5_file(duong_dan["vis"]) != d["vis_md5"]:
                loi.append(f"md5 VIS khác audit: {d['image_id']}")
            d["ir_md5"] = md5_file(duong_dan["ir"])
            d["label_md5"] = md5_file(duong_dan["label"])
            for p in duong_dan.values():
                with Image.open(p) as im:
                    if im.size != SIZE:
                        loi.append(f"sai kích thước {p.name}: {im.size}")
            nhan = np.asarray(Image.open(duong_dan["label"]))
            if nhan.ndim != 2 or nhan.max() > 14:
                loi.append(f"nhãn bất thường: {d['image_id']}")
            for lop in np.unique(nhan).tolist():
                so_lop[split][lop] = so_lop[split].get(lop, 0) + 1
            if split in hash_vis:
                hash_vis[split][d["image_id"]] = dhash(duong_dan["vis"])
                duong_dan_vis[d["image_id"]] = duong_dan["vis"]

    with (out / "kiem_tra_lop_theo_tap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["class_id", "so_anh_train", "so_anh_dev", "so_anh_test"])
        for lop in range(15):
            w.writerow([lop, *(so_lop[k].get(lop, 0) for k in ("train", "dev", "test"))])

    gan = []
    if hash_vis["train"] and hash_vis["dev"]:
        id_train = list(hash_vis["train"])
        ma_tran = np.stack([hash_vis["train"][i] for i in id_train])
        for id_dev, h in hash_vis["dev"].items():
            hamming = (ma_tran != h).sum(axis=1)
            for j in np.flatnonzero(hamming <= NGUONG_DHASH):
                gan.append((id_dev, id_train[j], int(hamming[j])))
    gan.sort(key=lambda t: t[2])
    with (out / "kiem_tra_dev_gan_train.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["anh_dev", "anh_train", "hamming_vis"])
        w.writerows(gan)
    print(f"  Cặp dev–train gần trùng (dHash VIS ≤ {NGUONG_DHASH}): {len(gan)} "
          "→ phải xem bằng mắt, xem kiem_tra_dev_gan_train.csv")
    if gan:
        hang = []
        for id_dev, id_train, _ in gan[:12]:
            cap = [Image.open(duong_dan_vis[i]).convert("RGB").resize((320, 240))
                   for i in (id_dev, id_train)]
            hang.append(np.hstack([np.asarray(a) for a in cap]))
        Image.fromarray(np.vstack(hang)).save(out / "kiem_tra_dev_gan_train.png")
    return loi


def ghi_csv(tap: dict[str, list], out: Path) -> None:
    for split, dong_list in tap.items():
        with (out / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COT, lineterminator="\n")
            w.writeheader()
            w.writerows(dong_list)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, help="thư mục FMB; có thì kiểm tra file và điền md5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    audit = doc_csv(args.audit_dir / "3_thong_tin_tung_anh.csv")
    leak = set((args.audit_dir / "5_anh_test_ro.txt").read_text(encoding="utf-8").split())
    do_sang = {int(d["name"]): float(d["vis_mean"]) for d in audit if d["split"].startswith("train")}
    khoi_cam = {int(d["anh_train"]) for d in doc_csv(args.audit_dir / "5_ung_vien_ro_vis_ir.csv")}

    khoi_dev, diem = chon_khoi_dev(do_sang, khoi_cam)
    dev_ids = {i for a, b in khoi_dev for i in range(a, b + 1)}
    print(f"Khối dev: {khoi_dev} | {len(dev_ids)} ảnh | lệch độ sáng L1 = {diem:.3f}")

    tap = tao_dong(audit, dev_ids, leak)
    print("Kiểm tra danh sách:")
    loi = kiem_tra_danh_sach(tap, leak, dev_ids)

    if args.root:
        print(f"Kiểm tra file trong {args.root}:")
        loi_file = kiem_tra_file(tap, args.root, args.out)
        print(f"  Lỗi file: {len(loi_file)}")
        print("\n".join(f"  - {x}" for x in loi_file[:20]))
        loi += loi_file
    else:
        print("CHƯA kiểm tra file: chưa có --root. Đuôi .png là giả định, md5 IR/label để trống.")

    ghi_csv(tap, args.out)
    print(f"Đã ghi 3 file vào {args.out}")
    if loi:
        print(f"KHÔNG ĐẠT: {len(loi)} lỗi. Chưa được dùng các file này.")
        return 1
    print("ĐẠT." if args.root else "ĐẠT phần danh sách. Vẫn phải chạy lại với --root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
