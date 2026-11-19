# Failure Analysis — SAGE / FMB scene_v1 (Train-only)

- Run ID: `20261119T043701Z`.
- Nguồn: FMB VIS–IR–Label official Train + Q02B scene_v1 + frozen SAGE Train PNG.
- 992 Train IDs, 228 Dev IDs; chỉ đọc 992 Train pairs có SAGE. Không dùng Dev/Test.
- Đã khóa 21 SHA256 bundle; kiểm MD5 10 mẫu Train đầu tiên.
- Xuất 30 trường hợp nghi vấn theo tiêu chí có thể tái lập; **chưa phải lỗi định lượng segmentation đã chứng minh**.

## Tổng quan chỉ số mô tả (median trên tập đã phân tích)
- Mean VIS luminance: 115.371 (đơn vị 0–255).
- Strong VIS edge retention = mean(Sobel(SAGE)|top20%-VIS-edge)/mean(Sobel(VIS)|top20%-VIS-edge): 0.905.
- Thermal contrast retention ở vùng GT person (Train-only): 0.854.
- Thermal contrast retention ở vùng GT vehicle (Train-only): 0.763.

## Hai giả thuyết có thể bác bỏ (cần Quân kiểm bằng classwise IoU)
1. H1 — SAGE có thể suy giảm các cấu trúc biên/chi tiết VIS ở một số cảnh, làm tổn hại phân đoạn lớp phụ thuộc biên.
   Bằng chứng cần: vùng có edge retention thấp phải trùng với suy giảm IoU theo lớp trong phân tích Train/Dev đúng protocol; kiểm đối chứng ảnh lỗi/ảnh không lỗi; không chọn gate theo Dev.
   Thử nghiệm tương lai: residual high-frequency VIS được giới hạn theo cue cạnh tại inference, so với zero-residual / fixed mixing, và kiểm trade-off thermal target.
2. H2 — Ở một số đối tượng nóng trong cảnh tối, luminance SAGE có thể làm suy giảm tương phản mục tiêu-vùng nền so với IR, đặc biệt tại người/xe.
   Bằng chứng cần: vùng người/xe có thermal contrast retention thấp đi cùng suy giảm IoU (đối chiếu GT chỉ trong phân tích và training); kiểm trường hợp IR không có lợi.
   Thử nghiệm tương lai: low-frequency IR injection/gate theo độ tin cậy cục bộ từ VIS/IR, không dùng GT lúc inference, kèm các đối chứng TB, fixed IR, global gate đã thử trước đó.

## Bộ ví dụ (ảnh Train-only)
Xem `selected_20_30_train_cases.csv` và `contact_sheet_*.png`.
Các cột `selection_reason`, `edge_retention`, `person_ir_retention`, `vehicle_ir_retention` mô tả tín hiệu quan sát, KHÔNG tự kết luận nguyên nhân của lỗi segmentation.

## Giới hạn
- Chỉ số pixel không suy ra mIoU; không khẳng định novelty. Phải xem ảnh và đối chiếu confusion/predictions của Quân.
- Frozen SAGE PNG source contract phải được kiểm đối chiếu pipeline B3; ảnh source khác contract không được trộn chung khi so điểm.
- Train-probe (nếu có) chỉ là tập con được kiểm ID; nếu được dùng để lọc trước thì phải ghi sai lệch chọn mẫu.
- Nhóm khóa giả thuyết và ablation trên Train; chỉ sau đó mới đánh giá Dev 228 một lần theo protocol đã đăng ký.
