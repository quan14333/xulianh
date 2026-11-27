"""27/11: tạo tai_lap/FROZEN_MANIFEST.json cho phương án nhóm a04 (theo "Yêu cầu chốt phương pháp", mục 1 và 4).

Ngày khóa: 26/11 12:59 giờ VN = lần cập nhật cuối của gói code Test (dataset Kaggle sage-test-b3-code), trước các
lượt Test. File này tổng hợp ngày 27/11 và đối chiếu: gói ở máy nhà, gói trên Kaggle và manifest chép ra trong output
từng notebook Test phải có SHA256 trùng nhau. Script chỉ băm file, không chạy file nào trong gói tải về.

python scene_v1/tao_frozen_manifest.py <thư mục gói sage-test-b3-code tải về>   (từ tai_lap)
"""
import hashlib
import json
import sys
from pathlib import Path

TL = Path(__file__).resolve().parents[1]
GOI = TL / "_kaggle_test_code"
KAGGLE = Path(sys.argv[1])
RUNS = TL / "_kaggle_out" / "test_gop5"
NGUON_RUN = {"test_a04_nhom1": "thanhmn123/sage-test-a04-nhom1", "test_a04_nhom2": "thanhmn123/sage-test-a04-nhom2",
             "test_moc": "thanhmn123/sage-test-moc-nb"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(TL).as_posix()


def main():
    goi = {}
    for p in sorted(GOI.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "dataset-metadata.json":
            r = p.relative_to(GOI).as_posix()
            k = KAGGLE / r
            goi[r] = {"sha256": sha(p), "kaggle_sha256": sha(k) if k.exists() else None}
            goi[r]["khop_kaggle"] = goi[r]["sha256"] == goi[r]["kaggle_sha256"]
    assert all(v["khop_kaggle"] for v in goi.values()), [k for k, v in goi.items() if not v["khop_kaggle"]]

    # manifest mà từng notebook Test thực sự dùng (notebook chép ra output)
    man_chay = {}
    for thu_muc, slug in NGUON_RUN.items():
        for f in ("train.csv", "dev.csv", "test.csv"):
            p = TL / "_kaggle_out" / thu_muc / "code" / "manifests" / f
            if p.exists():
                man_chay[f"{slug}:{f}"] = {"sha256": sha(p), "khop_goi": sha(p) == goi[f"manifests/{f}"]["sha256"]}
    assert all(v["khop_goi"] for v in man_chay.values()), man_chay

    ngoai_goi = {}
    for f, vai in (("cai_tien/tb_sua.py", "train tbsua_992 (không nhãn, không SAM)"),
                   ("scene_v1/chay_tbsua992.sh", "lệnh train tbsua_992: w_tb 150, w_g 27, s 0,5, 100 epoch, seed 0, 992 ảnh"),
                   ("scene_v1/tbsua_992.log", "log train tbsua_992 (13,2 phút, RTX 3050)"),
                   ("scene_v1/dung_alpha.py", "dựng a04 cho Train/Dev (cùng công thức dung_test.py)"),
                   ("scene_v1/cham_a04_3s.py", "chấm Dev228 3 seed – căn cứ khóa a04"),
                   ("scene_v1/cham_a04_3s.txt", "kết quả Dev228 3 seed – căn cứ khóa a04 (26/11)"),
                   ("scene_v1/cham_a04cvis.txt", "a04_cvis rớt luật trên Dev → giữ a04"),
                   ("QUY_TAC_CHAM.md", "giao thức chấm + câu kết luận ghi trước khi mở số Test"),
                   ("b3/so_sanh_cap.py", "script chấm cặp seed dùng ở máy nhà"),
                   ("b3/sage_student/tbsua_992.pt", "checkpoint ở máy nhà (phải trùng bản trong gói)"),
                   ("chi_phi/do_chi_phi.py", "đo chi phí (27/11, có a04)"),
                   ("chi_phi/ket_qua/T4_a04.json", "chi phí a04 trên T4")):
        p = TL / f
        ngoai_goi[f] = {"vai_tro": vai, "sha256": sha(p) if p.exists() else None}
    assert ngoai_goi["b3/sage_student/tbsua_992.pt"]["sha256"] == goi["sage_student/tbsua_992.pt"]["sha256"]
    ngoai_goi["b3/so_sanh_cap.py"]["khop_goi"] = ngoai_goi["b3/so_sanh_cap.py"]["sha256"] == goi["so_sanh_cap.py"]["sha256"]

    runs = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        runs[d.name] = {"bien_the": c["bien_the"], "seed": c["seed"],
                        "nguon": next((s for t, s in NGUON_RUN.items() if (TL / "_kaggle_out" / t / "runs" / d.name).exists()),
                                      "_kaggle_out/runs_gop (mốc seed 0–2, chạy trước 26/11)"),
                        **{f: sha(d / f) for f in ("config.json", "test_ket_qua_last.json", "test_nham_last.npz")}}
    # mọi run cùng giao thức: config chỉ khác seed, bien_the, gio_train, torch/gpu
    bo_qua = {"seed", "bien_the", "gio_train"}
    cfg = [json.loads((RUNS / r / "config.json").read_text(encoding="utf-8")) for r in runs]
    khac = sorted({k for c in cfg for k in set(c) | set(cfg[0]) if k not in bo_qua and c.get(k) != cfg[0].get(k)})

    kq = {
        "ten": "a04 (tbsua_a04) – phương án nhóm, phần của Mân",
        "ngay_khoa": "2026-11-26 12:59 (giờ VN) – chọn trên Dev228, gói code Test đóng băng trên Kaggle",
        "ngay_tong_hop_file": "2026-11-27",
        "khong_chinh_theo_test": "code, config, checkpoint, split, metric giữ nguyên từ lúc khóa; không chạy lại để chọn bản",
        "phuong_phap": {
            "cong_thuc": "Y = TB + 0,4·(tbsua_992 − TB); TB = (Y_vis + IR)/2; tbsua_992 = clamp(TB + 0,5·Student(Y_vis, IR), 0, 1); "
                         "màu: ghép Y với CrCb của VIS (OpenCV)",
            "tham_so": {"alpha": 0.4, "s": 0.5, "w_tb": 150, "w_g": 27, "epoch": 100, "batch": 4, "crop": [192, 256],
                        "lr": "Adam 2e-3, cosine", "seed": 0, "so_anh_train": 992},
            "khong_dung": ["SAM", "X-Decoder", "nhãn phân đoạn (khi train mô hình ghép)"],
            "tham_so_mo_hinh": 135617,
        },
        "giao_thuc_cham": {
            "segformer": {k: cfg[0][k] for k in ("model", "init", "epochs", "batch", "crop", "lr", "optimizer", "aug",
                                                 "chon_ckpt", "loss", "so_lop", "n_train") if k in cfg[0]},
            "metric_chinh": "mIoU_co_mat: một ma trận nhầm lẫn 15x15 gộp toàn tập, bỏ pixel nhãn 0, trung bình các lớp có mặt "
                            "(13 lớp, không bicycle) trên 273 ảnh test sạch (bỏ 7 ảnh rò); kèm 280 ảnh",
            "so_sanh": "so_sanh_cap.py: hiệu theo cặp seed, CI bootstrap khối 10 ảnh id liền (B=2000); thang 3 mức trong QUY_TAC_CHAM.md",
            "seed": [0, 1, 2, 3, 4],
            "config_khac_nhau_ngoai_seed_bien_the": khac,
        },
        "bang_chung_khoa": {
            "goi_test_kaggle": "dataset thanhmn123/sage-test-b3-code, lastUpdated 2026-11-26 05:59 UTC (12:59 giờ VN), "
                               "trước các lượt Test tbsua_a04",
            "goi_khop_may_nha": all(v["khop_kaggle"] for v in goi.values()),
            "manifest_notebook_khop_goi": all(v["khop_goi"] for v in man_chay.values()),
            "notebook_assert_md5_tbsua_992": "fb0f5e8be8465ec98474c81a38938d3a",
        },
        "goi_test": goi,
        "manifest_trong_output_notebook": man_chay,
        "file_ngoai_goi": ngoai_goi,
        "runs_test": runs,
        "repo": "sage-huong-khong-sam, commit 7b5a063 (26/11)",
    }
    ra = TL / "FROZEN_MANIFEST.json"
    ra.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ghi", ra, "| gói", len(goi), "file khớp Kaggle | manifest notebook", len(man_chay), "| runs", len(runs),
          "| config khác ngoài seed/bien_the:", khac, "| so_sanh_cap khớp gói:", ngoai_goi["b3/so_sanh_cap.py"]["khop_goi"])


if __name__ == "__main__":
    main()
