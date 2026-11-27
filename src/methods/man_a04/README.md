# `man_a04` – code phương án a04 (Mân)

Method card: [method card a04](../../../docs/methods/man_a04_METHOD_SUMMARY.md) ·
Kết quả: [kết quả a04](../../../reports/experiments/man_a04/README.md)

Code ở đây là **bản chép nguyên** từ thư mục làm việc của Mân (`tai_lap/`), chưa gom thành package của repo.
Đường dẫn bên trong script giữ như lúc chạy thật để SHA256 khớp `FROZEN_MANIFEST.json`, nên **không sửa trực tiếp**.
Khi đọc script, đổi đường dẫn theo bảng dưới.

| Thư mục / file | Nội dung |
|---|---|
| `goi_test/` | **Gói đã chạy Test FMB** (dataset Kaggle `sage-test-b3-code`, đóng băng 26/11 12:59), đúng từng byte. Gồm manifest chính thức (1.045/175/280, cờ 7 ảnh rò), `tbsua_992.pt`, checkpoint Student SAGE, code dựng ảnh / train / chấm SegFormer-B3 |
| `goi_test/dung_test.py` | dựng ảnh a04 cho 1.500 ảnh: `Y = TB + 0,4·(tbsua_992 − TB)`, màu = CrCb của VIS |
| `goi_test/xuat_anh.py` | xuất ảnh `tbsua_992`, `tb`, `sage` (Student SAGE) |
| `goi_test/chay_song_song.py`, `train_seg.py`, `eval_seg.py` | train SegFormer-B3 trên ảnh ghép (100 epoch, `last.pt`) và chấm Test |
| `goi_test/so_sanh_cap.py` | so theo cặp seed + CI bootstrap khối ảnh |
| `train_tbsua/` | train Student `tbsua_992` (không nhãn, không SAM): `tb_sua.py` + hàm gradient từ `train_student_nosam.py`, lệnh `chay_tbsua992.sh`, log |
| `dung_alpha.py` | dựng a04 cho Train/Dev (cùng công thức `dung_test.py`) |
| `cham/` | chấm Dev 3 seed, tạo `FROZEN_MANIFEST.json`, xuất bảng CSV, script chép sang repo |
| `chi_phi/do_chi_phi.py` | đo chi phí (E18 + a04): ms/ảnh, VRAM, giây/bước train |

## Đối chiếu đường dẫn trong script

| Trong script / `QUY_TAC_CHAM.md` (máy Mân) | Ở đây |
|---|---|
| `tai_lap/b3/` (`chung.py`, `xuat_anh.py`, `so_sanh_cap.py`, `sage_student/`) | `goi_test/` |
| `tai_lap/_kaggle_test_code/` (gói Kaggle `sage-test-b3-code`) | `goi_test/` |
| `tai_lap/_kaggle_vnc_code/manifests`, `_kaggle_dev228_code/manifests` (split Dev228) | `dev228/split_scene_v1/` trong thư mục kết quả |
| `tai_lap/cai_tien/tb_sua.py`, `train_student_nosam.py` | `train_tbsua/` |
| `tai_lap/scene_v1/chay_tbsua992.sh`, `tbsua_992.log` | `train_tbsua/` |
| `tai_lap/scene_v1/dung_alpha.py` | `dung_alpha.py` |
| `tai_lap/scene_v1/cham_a04_3s.py`, `tao_frozen_manifest.py`, `xuat_bang_ket_qua.py`, `dua_len_man.py` | `cham/` |
| `tai_lap/scene_v1/cham_a04_3s.txt`, `cham_a04cvis.txt` | `dev228/` trong thư mục kết quả |
| `tai_lap/chi_phi/do_chi_phi.py` | `chi_phi/do_chi_phi.py` |
| `tai_lap/_kaggle_out/dev228_gop/` (run Dev228) | không kèm (chỉ kèm bảng chấm `dev228/*.txt`) |
| `luu_tru/.../seg/anh` (ảnh ghép dựng sẵn), `D:/Miniconda3/...` | máy riêng, không kèm |

## Chạy lại Test trên Kaggle

1. Tạo dataset từ `goi_test/` (hoặc dùng `sage-test-b3-code`) + dataset FMB.
2. Notebook: `python xuat_anh.py --bien_the tbsua_992` → `python dung_test.py tbsua_a04` →
   `python chay_song_song.py --cap tbsua_a04:0 tbsua_a04:1 ... --gop_dev --lop0` (T4 x2, ~2,9 giờ/lượt).
3. Chấm ở máy: `SAGE_RUNS=<runs> python so_sanh_cap.py --bien_the tbsua_a04 tb sage --model b3 --hau _1220_15lop`.

## Train lại `tbsua_992` (máy nhà, ~13 phút RTX 3050)

Chạy trong `train_tbsua/`:

```bash
PYTHONPATH=../goi_test SAGE_MAN=<thư mục dev228/split_scene_v1> SO_TRAIN=992 SAGE_FMB=<thư mục FMB gốc> \
  python tb_sua.py --chi_train --ten tbsua_992 --w_tb 150
```

`PYTHONPATH` để import `chung` (script gốc tìm ở `tai_lap/b3`). Lệnh gốc: `train_tbsua/chay_tbsua992.sh`.
Script **ghi đè** `goi_test/sage_student/<ten>.pt` → dùng tên khác (vd. `--ten tbsua_992_lai`) để giữ bản đã khóa
(md5 `fb0f5e8b…`). Train lại trên GPU khác có thể lệch nhẹ.

## Lưu ý

- Hai checkpoint nhỏ trong `goi_test/sage_student/` (0,6 MB mỗi file) được đưa vào repo (`git add -f` ở repo nhóm vì
  `.gitignore` chặn `*.pt`) để kiểm SHA256 và chạy lại được. `checkpoints.pt` là trọng số Student SAGE công bố.
- Script có đường dẫn tuyệt đối kiểu `D:/CS406/...` là đường dẫn máy Mân lúc chạy thật; giữ nguyên để khớp hash.
