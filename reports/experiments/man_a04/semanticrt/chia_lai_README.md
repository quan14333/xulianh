# Chia lại SemanticRT – bộ kiểm tổng quát zero-shot (22/11)

**Mục đích:** kiểm mô hình ghép (train trên FMB, có thể thêm MSRS) trên dữ liệu **chưa từng thấy**.
Mô hình chấm (B0/B3) train trên ảnh ghép của `train.csv`, chấm trên `test.csv`. SAGE, TB, VIS và phương pháp nhóm dùng **cùng split này**.

## Kết quả

| | Train | Test |
|---|---|---|
| Số ảnh | 1.000 | 300 |
| Ảnh đêm (TB mức xám VIS < 60) | 78% | 75% |
| Cụm góc máy (mã cũ) | 18 cụm (thành phần G) | {5, 12, 19, 20} |

Số ảnh có mặt từng lớp 0–12:
- train: 1000, 904, 486, 382, 481, 484, 822, 81, 302, 302, 647, 989, 731
- test : 300, 293, 254, 75, 183, 100, 262, 19, 98, 105, 162, 286, 244

Lớp 7 hiếm (19 ảnh test). Tên lớp: 6 = car, 12 = person, 0 = background (gồm cả mặt đường). Tên các lớp còn lại cần đối chiếu bài SemanticRT trước khi ghi báo cáo.

md5: `train.csv` 4ff680f5…, `test.csv` eb78edc6…

## Vì sao không dùng split gốc
Split gốc chia **ngẫu nhiên theo khung hình**: 347/2.836 ảnh test trùng y hệt ảnh train, ~59% test có ảnh train gần cùng khung (kiểm 15/11).

## Các bước (chạy lại theo thứ tự)
1. `b1_thong_tin.py` → `thong_tin.csv`. Kích thước VIS/IR/nhãn khớp 100%; IR lưu 3 kênh nhưng giống nhau; nhãn chỉ 0–12. Bỏ 271 ảnh nhỏ (camera khác).
2. `b3_dia_diem.py` → `dia_diem_cap.csv`: khớp điểm ORB + RANSAC giữa các cụm góc máy (nền ~5 inlier; cùng địa điểm lệch góc/zoom 47–238).
3. `b2_chia.py` (hằng số trong `b2_chia_hang.py`) → `ung_vien_test.csv`, rồi `b5_kiem_orb.py 50 4` → `ro_orb.csv`, rồi chạy lại `b2_chia.py` → `train.csv`, `test.csv`, `chia_log.json`.
   - Gom ĐỊA ĐIỂM: cụm góc máy cũ (`semanticrt_may.json`) **trộn nhiều địa điểm**; cùng con phố được quay lúc ngày/đêm/zoom khác
     (xem mắt: test cụm 10 ban ngày = train cụm 4 ban đêm; cụm 7 = 15/6; cụm 2 = 15/18). Hợp mọi bằng chứng
     (liên kết cạnh cos > 0,8; ORB theo cụm ≥ 8; xem mắt) → thành phần lớn G gồm 18 cụm. TEST = 4 cụm không nối với cụm nào.
   - Bỏ gần trùng trong cùng cụm (cos đặc trưng cạnh ≥ 0,95) → test ứng viên 342, train 6.982.
   - Kiểm rò TỪNG ảnh test: so ORB với 50 ảnh train giống nhất → bỏ 13 ảnh (12 ảnh 155–319 inlier = ảnh "lạc cụm"; 1 ảnh 18 inlier, xem mắt không phân định được nên bỏ cho chắc). 329 ảnh còn lại < 10 inlier.
   - Lấy mẫu ngẫu nhiên seed 0: 1.000 / 300.
4. Các bản chia đã bỏ (đều có ảnh test trùng địa điểm train): test {2, 6, 10, 12} (thêm: lớp 8 = 0 ảnh), test {2, 7, 10, 12}.

## Ô kiểm (đều OK)
- giao train ∩ test = rỗng (id) và (nhóm địa điểm)
- không id trùng trong mỗi file; mọi file tồn tại; đủ 1.000 / 300
- mọi ảnh test: cos đặc trưng cạnh với train < 0,8 (max thực tế 0,53) và ORB inlier < 15

## Giới hạn còn lại
- Kiểm ORB dựa trên 50 ứng viên theo đặc trưng cạnh. Lần kiểm độ phủ (bản 2) có 1/20 ảnh trùng ngày–đêm nằm ở hạng 97. Phần sót còn lại, nếu có, là cùng địa điểm nhưng khác giờ hoặc góc. Nó ảnh hưởng **như nhau** tới mọi phương pháp được so (cùng mô hình chấm).
- 1.000 ảnh train lấy từ thành phần G. Không cần tách địa điểm bên trong train.
