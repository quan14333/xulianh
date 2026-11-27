# `man_a04` – code phương án a04 (Mân)

Method card: [`docs/methods/man_a04_METHOD_SUMMARY.md`](../../../docs/methods/man_a04_METHOD_SUMMARY.md) ·
Kết quả: [`reports/experiments/man_a04/`](../../../reports/experiments/man_a04/README.md)

Code ở đây là **bản chép nguyên** từ thư mục làm việc của Mân (`tai_lap/`), chưa gom thành package của repo.
Đường dẫn bên trong script giữ như lúc chạy thật để SHA256 khớp `FROZEN_MANIFEST.json`.

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

## Chạy lại Test trên Kaggle

1. Tạo dataset từ `goi_test/` (hoặc dùng `sage-test-b3-code`) + dataset FMB.
2. Notebook: `python xuat_anh.py --bien_the tbsua_992` → `python dung_test.py tbsua_a04` →
   `python chay_song_song.py --cap tbsua_a04:0 tbsua_a04:1 ... --gop_dev --lop0` (T4 x2, ~2,9 giờ/lượt).
3. Chấm ở máy: `SAGE_RUNS=<runs> python so_sanh_cap.py --bien_the tbsua_a04 tb sage --model b3 --hau _1220_15lop`.

## Lưu ý

- `*.pt` bị `.gitignore` chặn; hai checkpoint nhỏ trong `goi_test/sage_student/` (0,6 MB mỗi file) được thêm bằng `git add -f`
  để kiểm SHA256 và chạy lại được. `checkpoints.pt` là trọng số Student SAGE công bố.
- `train_tbsua/tb_sua.py` import `chung` (bản `b3/` của tai_lap) và `train_student_nosam` theo đường dẫn tai_lap; muốn chạy
  trong repo cần chỉnh `sys.path`.
