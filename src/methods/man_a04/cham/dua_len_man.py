"""27/11: chép phần a04 + kết quả Test/Dev/SemanticRT/chi phí vào repo nhóm xulianh (nhánh `man`).
Chỉ chép file nhỏ (không last.pt SegFormer, không ảnh ghép). Gói code Test chép NGUYÊN BYTE để SHA256 trong
FROZEN_MANIFEST.json kiểm được. python scene_v1/dua_len_man.py   (từ tai_lap)"""
import shutil
from pathlib import Path

TL = Path(__file__).resolve().parents[1]
REPO = TL.parent / "xulianh"
M = REPO / "src" / "methods" / "man_a04"
R = REPO / "reports" / "experiments" / "man_a04"
NHO = ("config.json", "log.csv", "test_iou_lop_last.csv", "test_ket_qua_last.json", "test_nham_last.npz")


def cp(a, b):
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a, b)


def main():
    for p in (TL / "_kaggle_test_code").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            cp(p, M / "goi_test" / p.relative_to(TL / "_kaggle_test_code"))
    for f, d in (("cai_tien/tb_sua.py", "train_tbsua/tb_sua.py"),
                 ("e17/train_student_nosam.py", "train_tbsua/train_student_nosam.py"),
                 ("scene_v1/chay_tbsua992.sh", "train_tbsua/chay_tbsua992.sh"),
                 ("scene_v1/tbsua_992.log", "train_tbsua/tbsua_992.log"),
                 ("scene_v1/dung_alpha.py", "dung_alpha.py"),
                 ("scene_v1/cham_a04_3s.py", "cham/cham_a04_3s.py"),
                 ("scene_v1/tao_frozen_manifest.py", "cham/tao_frozen_manifest.py"),
                 ("scene_v1/dua_len_man.py", "cham/dua_len_man.py"),
                 ("scene_v1/xuat_bang_ket_qua.py", "cham/xuat_bang_ket_qua.py"),
                 ("chi_phi/do_chi_phi.py", "chi_phi/do_chi_phi.py")):
        cp(TL / f, M / d)

    for f in ("FROZEN_MANIFEST.json", "QUY_TAC_CHAM.md"):
        cp(TL / f, R / f)
    G = TL / "_kaggle_out" / "test_gop5"
    for d in G.iterdir():
        if d.is_dir():
            for f in NHO:
                if (d / f).exists():   # nosam s1, s2 (chạy 18/11) không có test_iou_lop_last.csv; IoU lớp tính lại từ npz
                    cp(d / f, R / "test_fmb" / "runs" / d.name / f)
    for f in ("so_sanh_test5.txt", "so_sanh_cap_test_b3.csv", "so_sanh_nosam_s012.csv"):
        cp(G / f, R / "test_fmb" / f)
    for f in ("cham_a04_3s.txt", "cham_a04cvis.txt", "rao_a04_cvis.txt"):
        cp(TL / "scene_v1" / f, R / "dev228" / f)
    S = TL / "_kaggle_out" / "srt_s0" / "runs"
    cp(S / "tong_hop_srt.csv", R / "semanticrt" / "s0" / "tong_hop_srt.csv")
    for d in S.iterdir():
        if d.is_dir():
            for f in NHO:
                if (d / f).exists():
                    cp(d / f, R / "semanticrt" / "s0" / "runs" / d.name / f)
    cp(TL / "scene_v1" / "srt_rao" / "rao.txt", R / "semanticrt" / "rao_anh_train300.txt")
    cp(TL / "data_ngoai_chia_README.md" if (TL / "data_ngoai_chia_README.md").exists()
       else TL.parent / "data_ngoai" / "chia_semanticrt" / "README.md", R / "semanticrt" / "chia_lai_README.md")
    for f in ("T4.json", "T4_a04.json"):
        cp(TL / "chi_phi" / "ket_qua" / f, R / "chi_phi" / f)
    cp(TL / "chi_phi" / "README.md", R / "chi_phi" / "README_E18.md")
    print("xong:", sum(1 for _ in M.rglob("*") if _.is_file()), "file code |", sum(1 for _ in R.rglob("*") if _.is_file()), "file kết quả")


if __name__ == "__main__":
    main()
