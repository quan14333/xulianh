# Audit FMB

Kết quả kiểm tra bộ FMB do Mân chạy trên Colab ngày 3–4/11/2026, bằng notebook
`notebooks/man/FMB_audit_gop_final.ipynb`. Mọi số dưới đây là số nhóm tự đo.

## Kết quả chính

- **Số lượng:** 1.220 train + 280 test = 1.500 cặp, 0 lỗi ghép cặp, 0 ảnh hỏng.
- **Định dạng:** tất cả 800×600. VIS 3 kênh uint8. IR lưu 3 kênh giống hệt nhau, nên đọc 1 kênh.
  Label 1 kênh, giá trị 0–14, không có 255.
- **Thư mục `color`:** là nhãn tô màu, không phải ảnh VIS.
- **Lớp hiếm:** bicycle có 12 ảnh train và 0 ảnh test. Dưới 0,5% số pixel gồm traffic_light,
  traffic_sign, truck, motorcycle, bicycle.
- **Căn chỉnh IR–VIS (300 cặp ngẫu nhiên):** nhóm 150 cặp tin cậy cao có trung vị dx 1,06 px,
  dy 0,96 px, |lệch| 3,15 px, P90 6,14 px, 1,33% lệch trên 10 px. Không có lệch hệ thống đáng kể.
  Phép đo này không thấy được lệch xoay hay lệch cục bộ.

## Rò dữ liệu train–test

**7/280 ảnh test (2,5%)** là khung liền kề cùng video với ảnh train. Không loại chúng khỏi benchmark.
Báo cáo chính dùng đủ 280 ảnh, có thể báo thêm trên 273 ảnh.

| Cụm | Ảnh test | Ảnh train cùng cảnh |
|---|---|---|
| A – đường đêm, toà nhà + vách đá | 00587, 00588, 00589, 00590 | 00489–00500 |
| B – phố chung cư có sương | 00631, 00632, 00633 | 00586 |

Tiêu chí:

1. Trùng tuyệt đối theo MD5: 0 cặp.
2. Tìm ứng viên bằng dHash 64 bit trên VIS và IR, Hamming ≤ 5.
3. Xem bằng mắt để chốt. dHash chỉ dùng tìm ứng viên: IR nhiễu ở ảnh đêm, còn ngưỡng 8–10 bit báo
   131–312 cặp, phần lớn là nhầm.

Đã loại khi xem bằng mắt vì khác cảnh: test 00632/00633 ↔ train 00442, test 00880 ↔ train 00571.

## File

| File | Nội dung |
|---|---|
| `2_so_anh_theo_split.csv`, `2_loi_ghep_cap.csv` | Số ảnh và lỗi ghép cặp |
| `3_thong_tin_tung_anh.csv` | Kích thước, kênh, md5 VIS, độ sáng từng ảnh. Script manifest cần |
| `4_phan_bo_lop.csv`, `4_phan_bo_lop.png`, `4_bang_mau_color.csv` | Phân bố lớp, bảng màu |
| `5_anh_test_ro.txt` | 7 ảnh rò đã chốt. Script manifest cần |
| `5_ung_vien_ro_vis_ir.csv` | Ứng viên rò. Script manifest cần |
| `5_anh_test_ro_da_chot.csv`, `5_ro_theo_nguong.csv` | Cặp đã chốt, số cặp theo ngưỡng |
| `6_can_chinh_tung_cap.csv`, `6_tom_tat_can_chinh.csv`, `6_can_chinh.png` | Đo căn chỉnh |
| `8_tom_tat.json` | Tóm tắt các số chính |

Không đưa lên 3 hình `5_xem_mat_anh_ro.png`, `6_cap_lech_nhat.png`, `7_bo_anh_minh_hoa.png` vì có
ảnh dataset và nặng 1–3 MB. Chạy lại notebook sẽ tạo lại các hình này.

## Khác biệt so với lúc chạy

Notebook trên repo đã xóa output và sửa đoạn giải nén để dữ liệu nằm ở `/content/FMB/train/Visible`.
Phần phân tích không đổi. Các file ở đây được tạo lúc dữ liệu còn ở `train/train/Visible`, nên cột
`split` và tiêu đề cột ghi `train/train`, `test/test`. Chạy lại sẽ thấy `train`, `test`.
