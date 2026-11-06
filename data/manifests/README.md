# Manifest FMB

| File | Số ảnh | Ghi chú |
|---|---:|---|
| `train.csv` | 1.045 | Train chính thức trừ dev |
| `dev.csv` | 175 | 14,3% của train, gồm 5 khối id liền nhau |
| `test.csv` | 280 | Đủ test chính thức; 7 ảnh có `leak=True` |

Tổng 1.500 ảnh, không bỏ ảnh nào, không dùng ảnh đệm.

Đọc `image_id` dạng chuỗi để giữ số 0 đầu, ví dụ `pd.read_csv(path, dtype={"image_id": str})`.
Đường dẫn ảnh là tương đối so với thư mục FMB. Lọc `leak == False` trong `test.csv` để có 273 ảnh.

## Cách chia dev

Train và test chính thức xen kẽ thành 20 khối train và 20 khối test. Dev lấy **nguyên khối train**,
nên các khối test nằm giữa dev và train đóng vai trò đệm (khoảng cách id nhỏ nhất: 5).

Khối dev: `00052–00084`, `00091–00120`, `00172–00206`, `00591–00630`, `01383–01419`.

Script chọn tự động, không dùng seed và không dùng ảnh hay kết quả của test để so độ sáng:

1. Xét các khối train có từ 25 ảnh.
2. Bỏ khối chứa ảnh train nằm trong danh sách ứng viên rò của audit (`5_ung_vien_ro_vis_ir.csv`).
3. Xét tổ hợp 3–5 khối có tổng 122–183 ảnh (10–15% của 1.220).
4. Chọn tổ hợp có phân bố `vis_mean` gần phần train còn lại nhất (L1, mốc 60/90/130).

Id xa nhau chưa chắc khác cảnh, nên bước kiểm tra có ảnh sẽ tìm cặp dev–train gần trùng bằng dHash
VIS (Hamming ≤ 5) rồi xem bằng mắt. Cách chia này chưa được nhóm chốt.

## Tạo lại

```bash
python src/data/build_fmb_manifests.py --audit-dir <audit> --out data/manifests --root <FMB>
```

`--root` là thư mục chứa `train/` và `test/`. Script nhận cả kiểu `train/Visible` và kiểu
`train/train/Visible` (Colab), nhưng manifest luôn ghi `train/Visible/...`. Nếu dữ liệu ở kiểu
`train/train/`, cần đổi đường dẫn khi đọc. Không có `--root` thì script chỉ kiểm tra danh sách,
đuôi `.png` là giả định và cột `ir_md5`, `label_md5` để trống.

Kết quả kiểm tra trên FMB thật nằm ở `kiem_tra_dev_gan_train.csv` và `kiem_tra_lop_theo_tap.csv`.
